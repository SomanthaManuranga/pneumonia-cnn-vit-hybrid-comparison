"""
Ensemble evaluation module.

"""

from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from tqdm import tqdm

from train import build_model


@torch.no_grad()
def _get_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "predict",
) -> torch.Tensor:
    """
    Run one model over the loader and return its predicted probabilities.

    """
    model.eval()
    all_probs = []

    for imgs, _labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)   
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)



def evaluate_ensemble(
    model_name: str,
    fold_checkpoints: List[str],
    test_loader: DataLoader,
    device: str = None,
) -> Dict:
    """
    Build a soft-voting ensemble from the fold checkpoints and evaluate it
    on the test set.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Building ensemble for {model_name} from {len(fold_checkpoints)} folds")
    print(f"Device: {device}")

    # Build the model once; we will load different fold-weights into it.
    model = build_model(model_name).to(device)

    # This will hold the running sum of probabilities from all folds.
    summed_probs = None

    for i, ckpt_path in enumerate(fold_checkpoints):
        ckpt_path = Path(ckpt_path)
        print(f"  Loading fold {i}: {ckpt_path.name}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # Get this fold-model's probabilities on the test set.
        probs = _get_probabilities(model, test_loader, device, desc=f"fold {i} predict")

        # Add them to the running total (we average at the end).
        if summed_probs is None:
            summed_probs = probs
        else:
            summed_probs = summed_probs + probs

    # Average the probabilities across all folds (soft voting).
    avg_probs = summed_probs / len(fold_checkpoints)

    # Final prediction = the class with the higher averaged probability.
    ensemble_preds = avg_probs.argmax(dim=1).tolist()

    # Ground-truth labels — read directly from dataset (same order, no extra disk I/O).
    true_labels = [label for _, label in test_loader.dataset.samples]

    # Compute the same metrics we used everywhere else.
    metrics = {
        "accuracy":  accuracy_score(true_labels, ensemble_preds),
        "f1":        f1_score(true_labels, ensemble_preds, average="weighted"),
        "precision": precision_score(true_labels, ensemble_preds, average="weighted", zero_division=0),
        "recall":    recall_score(true_labels, ensemble_preds, average="weighted", zero_division=0),
        "recall_normal":    recall_score(true_labels, ensemble_preds, pos_label=0, average="binary", zero_division=0),
        "recall_pneumonia": recall_score(true_labels, ensemble_preds, pos_label=1, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, ensemble_preds).tolist(),
    }

    print("\nEnsemble test results:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  Recall NORMAL:   {metrics['recall_normal']:.4f}")
    print(f"  Recall PNEUMONIA:{metrics['recall_pneumonia']:.4f}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")

    return metrics