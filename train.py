"""
Shared training and evaluation module.

"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from tqdm import tqdm


# ============================================================
# Reproducibility function
# ============================================================
def set_seed(seed: int = 42) -> None:
    """
    Seed every random number generator that touches training.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False    


# ============================================================
# Model factory
# ============================================================
def build_model(name: str, num_classes: int = 2) -> nn.Module:
    """
    Build a pretrained model with a 2-class classification head.
    """
    valid_names = {
        "resnet50",
        "vit_base_patch16_384",
        "tiny_vit_21m_384",
    }
    assert name in valid_names, f"Unknown model: {name}. Use one of {valid_names}"

    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    return model


# ============================================================
# Class weights - handles the NORMAL:PNEUMONIA imbalance
# ============================================================
def compute_class_weights(train_samples: List, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a fold's training samples.
    """
    n_normal = sum(1 for _, label in train_samples if label == 0)
    n_pneumonia = sum(1 for _, label in train_samples if label == 1)
    total = n_normal + n_pneumonia

    weight_normal = total / (2 * n_normal)
    weight_pneumonia = total / (2 * n_pneumonia)

    return torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32).to(device)


# ============================================================
# Single epoch training
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_num: int,
    scaler: torch.amp.GradScaler = None,
) -> float:
    """Run one epoch of training. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch_num} [train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_samples


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
    criterion: nn.Module = None,
) -> Dict[str, float]:
    """
    Evaluate model on a loader. Returns dict with accuracy, f1, precision, recall.
    Uses the exact same metric definitions as the base paper.

    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_samples = 0

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        labels_dev = labels.to(device)
        logits = model(imgs)

        if criterion is not None:
            loss = criterion(logits, labels_dev)
            total_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_normal":    recall_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0),
        "recall_pneumonia": recall_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0),
    }

    if criterion is not None and n_samples > 0:
        metrics["loss"] = total_loss / n_samples

    cm = confusion_matrix(all_labels, all_preds).tolist()
    metrics["confusion_matrix"] = cm

    return metrics


# ============================================================
# Train ONE fold of k-fold cross-validation
# ============================================================
def train_one_fold(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_index: int,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = None,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 42,
    class_weights: torch.Tensor = None,
) -> Dict:
    """
    Train a single fold and save its best checkpoint (by validation F1).

    """
    # Seed everything before any randomness .
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    # Each fold gets its own checkpoint, e.g. resnet50_fold0_best.pth
    best_ckpt_path = checkpoint_dir / f"{model_name}_fold{fold_index}_best.pth"

    model = build_model(model_name).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Weighted Cross-Entropy: the class_weights make the rarer class count.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Fold {fold_index}] {model_name} ({n_params:,} params) | "
          f"AdamW lr={lr} | weighted CE | epochs={epochs} | device={device}")
    if scaler is not None:
        print(f"[Fold {fold_index}] Mixed precision (AMP/FP16) enabled")
    print(f"[Fold {fold_index}] train batches={len(train_loader)}, val batches={len(val_loader)}")
    print("-" * 60)

    history = {"train_loss": [], "val_metrics": []}
    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler=scaler
        )
    
        val_metrics = evaluate(model, val_loader, device,
                               desc=f"Fold {fold_index} Epoch {epoch} [val]",
                               criterion=criterion)

        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - t0
        print(
            f"[Fold {fold_index}] Epoch {epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save the best checkpoint for this fold (highest validation F1).
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "model_name": model_name,
                "fold_index": fold_index,
            }, best_ckpt_path)
            print(f"  -> [Fold {fold_index}] New best F1, saved: {best_ckpt_path.name}")

    print("-" * 60)
    print(f"[Fold {fold_index}] Done. Best val F1 = {best_val_f1:.4f} (epoch {best_epoch})")

    return {
        "model_name": model_name,
        "fold_index": fold_index,
        "history": history,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "checkpoint_path": str(best_ckpt_path),
    }


# ============================================================
# ORIGINAL single-split training loop
# ============================================================
def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = None,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 42,
) -> Dict:
    
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    best_ckpt_path = checkpoint_dir / f"{model_name}_best.pth"

    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler is not None:
        print("Mixed precision (AMP/FP16) enabled")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} parameters)")
    print(f"Optimizer: Adam, lr={lr}")
    print(f"Loss: CrossEntropy (unweighted, matching paper)")
    print(f"Epochs: {epochs}")
    print(f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print("-" * 60)

    history = {"train_loss": [], "val_metrics": []}
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=scaler)
        val_metrics = evaluate(model, val_loader, device, desc=f"Epoch {epoch} [val]")

        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "model_name": model_name,
            }, best_ckpt_path)
            print(f"  -> New best F1, checkpoint saved: {best_ckpt_path.name}")

    print("-" * 60)
    print("Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, desc="test")

    print("\nFinal test results:")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):   {test_metrics['f1']:.4f}")
    print(f"  Precision:       {test_metrics['precision']:.4f}")
    print(f"  Recall:          {test_metrics['recall']:.4f}")
    print(f"  Recall NORMAL:   {test_metrics['recall_normal']:.4f}")
    print(f"  Recall PNEUMONIA:{test_metrics['recall_pneumonia']:.4f}")
    print(f"  Confusion matrix: {test_metrics['confusion_matrix']}")

    return {
        "model_name": model_name,
        "history": history,
        "test_metrics": test_metrics,
        "best_val_f1": best_val_f1,
        "best_epoch": ckpt["epoch"],
    }