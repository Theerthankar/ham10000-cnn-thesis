"""Class-imbalance strategies (proposal Section 5.3)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .constants import NUM_CLASSES


def class_counts(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    return np.maximum(counts, 1.0)


def inverse_frequency_weights(train_labels: np.ndarray) -> torch.Tensor:
    """w_c = N / (K * n_c) on the training set."""
    n = len(train_labels)
    counts = class_counts(train_labels)
    weights = n / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights_for_oversampling(train_labels: np.ndarray) -> torch.Tensor:
    counts = class_counts(train_labels)
    per_class = 1.0 / counts
    return torch.tensor([per_class[int(y)] for y in train_labels], dtype=torch.float64)


def make_train_loader(
    dataset,
    train_labels: np.ndarray,
    strategy: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, torch.nn.CrossEntropyLoss]:
    strategy = strategy.lower()
    if strategy == "augmentation":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return loader, torch.nn.CrossEntropyLoss()

    if strategy == "weighted_ce":
        class_weights = inverse_frequency_weights(train_labels)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return loader, torch.nn.CrossEntropyLoss(weight=class_weights)

    if strategy == "oversampling":
        weights = sample_weights_for_oversampling(train_labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return loader, torch.nn.CrossEntropyLoss()

    raise ValueError(f"Unknown imbalance strategy: {strategy}")
