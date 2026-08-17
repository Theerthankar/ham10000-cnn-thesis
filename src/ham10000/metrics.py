"""Evaluation metrics (proposal Chapter 7)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def predict_all(model, loader, device):
    """Raw per-sample predictions, for McNemar's paired test."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def evaluate_model(model, loader, device, num_classes: int = 7) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
        # labels= is load-bearing here. Without it sklearn silently drops any
        # class absent from both y_true and y_pred, and anything downstream that
        # indexes the report by class position then dies with a KeyError. All
        # seven are present in the full test set, but smoke subsets are not, and
        # the resulting crash is a long way from its cause.
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(num_classes)),
            zero_division=0, output_dict=True
        ),
    }
    try:
        metrics["macro_auc_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes)))
        )
    except ValueError:
        metrics["macro_auc_ovr"] = None
    return metrics
