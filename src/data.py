"""
This module is imported by all three training notebooks (ResNet, ViT, TinyViT).
"""

import random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold


def _worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker's random state deterministically.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================
# Configuration — matches paper's setup
# ============================================================
DATA_ROOT = Path(__file__).resolve().parent.parent / "Dataset" / "chest_xray"
IMG_SIZE = 384  # paper used 384
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL, label 1 = PNEUMONIA

# ImageNet normalization, required because pretrained models were trained on it
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Base dataset — shared loading logic lives here once
# ============================================================
class ListDataset(Dataset):
    """
    Base dataset built from a pre-made list of (image_path, label) pairs.

    Used directly for k-fold folds (each fold is a custom subset of paths).
    Also serves as the base class for PneumoniaDataset.
    """

    def __init__(self, samples: List, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Open and convert to RGB (handles grayscale X-rays by duplicating channel)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============================================================
# Folder-scanning dataset — inherits all logic from ListDataset
# ============================================================
class PneumoniaDataset(ListDataset):
    """
    Loads chest X-ray images by scanning the Kermany dataset folders.
    Builds self.samples from disk, then delegates to ListDataset.

    Folder structure expected:
        root/
            NORMAL/      (label 0)
            PNEUMONIA/   (label 1)
    """

    def __init__(self, split: str, transform=None):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        root = DATA_ROOT / split

        # Collect all (image_path, label) tuples by scanning folders
        samples = []
        for label, cls_name in enumerate(CLASS_NAMES):
            cls_folder = root / cls_name
            for f in cls_folder.iterdir():
                if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    samples.append((f, label))

        super().__init__(samples, transform)


# ============================================================
# Transform pipelines — paper-matching
# ============================================================
def get_train_transform():
    """Training transforms with augmentation, matching paper's description."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """no augmentation, only resize + normalize."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# K-fold cross-validation with a held-out test set
# ============================================================
def get_test_loader(batch_size: int = 16, num_workers: int = 2) -> DataLoader:
    test_ds = PneumoniaDataset("test", transform=get_eval_transform())
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return test_loader


def _get_pooled_training_samples() -> List:
    """
    Pool all training images for cross-validation.

    The paper provides train/ (5216 images) and val/ (16 images) folders.
    Since the standard val/ set is far too small (only 16 images) to be
    useful on its own, we combine train/ and val/ into one pool. K-fold
    will then create proper validation sets from this pool.
    """
    pooled = []
    pooled += PneumoniaDataset("train", transform=None).samples
    pooled += PneumoniaDataset("val", transform=None).samples
    return pooled


def get_fold_loaders(
    fold_index: int,
    n_splits: int = 5,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    assert 0 <= fold_index < n_splits, "fold_index must be between 0 and n_splits-1"


    torch.manual_seed(seed)

    pooled_samples = _get_pooled_training_samples()


    labels = [label for (_, label) in pooled_samples]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,   
    )

    
    all_folds = list(skf.split(X=pooled_samples, y=labels))
    train_idx, val_idx = all_folds[fold_index]

    train_samples = [pooled_samples[i] for i in train_idx]
    val_samples = [pooled_samples[i] for i in val_idx]

   
    train_ds = ListDataset(train_samples, transform=get_train_transform())
    val_ds = ListDataset(val_samples, transform=get_eval_transform())

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=g, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader


# ============================================================
# Helpers for printing dataset / fold information
# ============================================================
def print_dataset_info():
    """Print summary of dataset sizes for the original fixed split."""
    for split in ["train", "val", "test"]:
        ds = PneumoniaDataset(split, transform=None)
        n_normal = sum(1 for _, l in ds.samples if l == 0)
        n_pneumonia = sum(1 for _, l in ds.samples if l == 1)
        print(f"  {split}: {len(ds)} total ({n_normal} NORMAL, {n_pneumonia} PNEUMONIA)")


def print_fold_info(n_splits: int = 5, seed: int = 42):
    """
    Print the size and class balance of each fold, plus the held-out test set.

    Useful as a one-time check to confirm:
      - the test set is the expected size (624) and never inside a fold,
      - each fold preserves the NORMAL:PNEUMONIA ratio (stratification works).
    """
    pooled_samples = _get_pooled_training_samples()
    labels = [label for (_, label) in pooled_samples]

    n_normal = sum(1 for l in labels if l == 0)
    n_pneumonia = sum(1 for l in labels if l == 1)
    print(f"Pooled training images (train + val): {len(pooled_samples)} "
          f"({n_normal} NORMAL, {n_pneumonia} PNEUMONIA)")

    test_ds = PneumoniaDataset("test", transform=None)
    t_normal = sum(1 for _, l in test_ds.samples if l == 0)
    t_pneumonia = sum(1 for _, l in test_ds.samples if l == 1)
    print(f"Held-out TEST set (never used in training): {len(test_ds)} "
          f"({t_normal} NORMAL, {t_pneumonia} PNEUMONIA)")
    print()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (train_idx, val_idx) in enumerate(skf.split(pooled_samples, labels)):
        val_labels = [labels[i] for i in val_idx]
        v_normal = sum(1 for l in val_labels if l == 0)
        v_pneumonia = sum(1 for l in val_labels if l == 1)
        print(f"  Fold {fold_index}: train={len(train_idx)}, "
              f"val={len(val_idx)} ({v_normal} NORMAL, {v_pneumonia} PNEUMONIA)")
