"""Training loop: early stopping on validation loss, resumable from checkpoint.

Defaults are the ones the final runs actually used (50 epochs, patience 10),
not the 30/5 budget the proposal originally registered. The change was made
after piloting showed ResNet-50 was still improving when patience 5 cut it off,
and it is applied to every experiment so the comparison stays controlled.

Every run writes five files into its output directory:
  config.json           the full resolved config, including the split hash
  history.csv           per-epoch train/val curves
  test_metrics.json     final test metrics from the best checkpoint
  test_predictions.csv  per-image predictions, needed for McNemar's test
  checkpoint_best.pt    weights at lowest validation loss
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .constants import CLASS_NAMES, DEFAULT_IMAGE_SIZE, EXPERIMENTS, NUM_CLASSES
from .dataset import HAM10000Dataset
from .imbalance import make_train_loader
from .metrics import evaluate_model, predict_all
from .models import build_model
from .split import load_or_create_split, split_hash
from .transforms import eval_transforms, train_transforms


# Peak GPU memory for a real training step at batch size 32, measured on an
# RTX A4500 (see the methodology chapter). Used only for the preflight check.
_VRAM_NEEDED = {
    ("mobilenet_v2", 224): 2.6,
    ("resnet50", 224): 3.1,
    ("efficientnet_b3", 224): 5.8,
    ("efficientnet_b3", 300): 10.2,
}


@dataclass
class TrainConfig:
    experiment_id: str = "E1"
    data_root: str = "data"
    output_dir: str = "runs/E1"
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 10
    num_workers: int = 4
    smoke_test: bool = False
    smoke_epochs: int = 2
    smoke_max_train: int = 256
    smoke_max_val: int = 64
    smoke_max_test: int = 64
    skip_if_complete: bool = True
    force_rerun: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _smoke_subsample(frame, max_size: int, seed: int):
    if len(frame) <= max_size:
        return frame
    return frame.sample(n=max_size, random_state=seed).reset_index(drop=True)


def _run_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _validate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / max(total, 1)


def _update_manifest(manifest_path: Path, experiment_id: str, entry: dict) -> None:
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest[experiment_id] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def train_experiment(config: TrainConfig) -> dict:
    if config.experiment_id not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {config.experiment_id}. "
            f"Known: {sorted(EXPERIMENTS)}"
        )
    exp = EXPERIMENTS[config.experiment_id]
    image_size = exp.get("image_size", DEFAULT_IMAGE_SIZE)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "manifest.json"

    metrics_path = output_dir / "test_metrics.json"
    if config.skip_if_complete and not config.force_rerun and metrics_path.exists():
        print(f"{config.experiment_id} is already finished ({metrics_path}).")
        print("Set force_rerun=True if you want to train it again from scratch.")
        config_path = output_dir / "config.json"
        return {
            "config": json.loads(config_path.read_text()) if config_path.exists() else {},
            "test_metrics": json.loads(metrics_path.read_text()),
            "output_dir": str(output_dir),
            "skipped": True,
        }

    set_seed(config.seed)
    data_root = Path(config.data_root)

    # The split file is shared by every experiment. First run creates it, the
    # other eleven load it, so all twelve see byte-identical partitions.
    split_path = output_dir.parent / "splits" / "seed42_lesion_stratified.json"
    df, lesion_splits = load_or_create_split(data_root / "HAM10000_metadata.csv", split_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if config.smoke_test:
        train_df = _smoke_subsample(train_df, config.smoke_max_train, config.seed)
        val_df = _smoke_subsample(val_df, config.smoke_max_val, config.seed + 1)
        test_df = _smoke_subsample(test_df, config.smoke_max_test, config.seed + 2)
        print(f"Smoke subset: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_labels = train_df["label"].to_numpy()
    train_ds = HAM10000Dataset(train_df, data_root, train_transforms(image_size))
    val_ds = HAM10000Dataset(val_df, data_root, eval_transforms(image_size))
    test_ds = HAM10000Dataset(test_df, data_root, eval_transforms(image_size))

    pin_memory = torch.cuda.is_available()
    workers = 0 if config.smoke_test else config.num_workers

    train_loader, criterion = make_train_loader(
        train_ds, train_labels, exp["strategy"],
        batch_size=config.batch_size, num_workers=workers, pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preflight VRAM check. A finished notebook keeps its kernel alive holding a
    # CUDA context and model, so running several in sequence without shutting
    # them down exhausts the card. The resulting OOM surfaces partway through
    # epoch 1, long after the actual mistake, and an idle kernel looks identical
    # to a running one. Fail here instead, naming the cause.
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / 1e9
        need_gb = _VRAM_NEEDED.get((exp["architecture"], image_size), 4.0)
        if free_gb < need_gb * 1.05:
            raise RuntimeError(
                f"{config.experiment_id} needs about {need_gb:.1f} GB of GPU memory at "
                f"{image_size}px, but only {free_gb:.1f} GB of "
                f"{total_b / 1e9:.1f} GB is free.\n"
                f"Another notebook's kernel is most likely still holding it. Shut down "
                f"finished kernels (JupyterLab: Running Terminals and Kernels) and re-run.\n"
                f"Closing the browser tab does not release the memory."
            )
        print(f"vram: {free_gb:.1f} GB free, ~{need_gb:.1f} GB needed")

    model = build_model(exp["architecture"], NUM_CLASSES).to(device)
    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)

    # No weight decay, no LR schedule, no gradient clipping. Held identical
    # across all twelve runs, and disclosed in the methodology chapter.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))

    run_config = {
        **asdict(config), **exp,
        "image_size": image_size,
        "class_names": CLASS_NAMES,
        "split_hash": split_hash(lesion_splits),
        "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    max_epochs = config.smoke_epochs if config.smoke_test else config.max_epochs
    best_val_loss = float("inf")
    epochs_without_improve = 0
    history_path = output_dir / "history.csv"
    history_fields = [
        "epoch", "train_loss", "train_acc", "val_loss",
        "val_accuracy", "val_balanced_accuracy", "val_macro_f1",
    ]

    start_epoch = 1
    checkpoint_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "checkpoint_best.pt"

    resuming = checkpoint_path.exists()
    if resuming:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Resuming across a settings change would silently splice two different
        # protocols into one history file, which is the kind of contamination
        # nobody notices until a reviewer asks. Refuse instead.
        _MUST_MATCH = ("architecture", "strategy", "image_size", "batch_size",
                       "lr", "num_workers", "seed", "split_hash")
        old = ckpt.get("config", {})
        drift = {
            k: (old.get(k), run_config.get(k))
            for k in _MUST_MATCH
            if k in old and old.get(k) != run_config.get(k)
        }
        if drift:
            details = "\n".join(f"    {k}: checkpoint={o!r} now={n!r}"
                                for k, (o, n) in drift.items())
            raise RuntimeError(
                f"{config.experiment_id} has a checkpoint from a different configuration:\n"
                f"{details}\n"
                f"Resuming would mix both protocols in one run. Either restore the old\n"
                f"settings, or start this experiment clean:\n"
                f"    rm -rf {output_dir}"
            )

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        epochs_without_improve = ckpt.get("epochs_without_improve", 0)
        print(f"Resuming after epoch {start_epoch - 1}")

    # Append when resuming so earlier epochs survive a disconnect; only a fresh
    # run truncates the history file.
    write_mode = "a" if (resuming and history_path.exists()) else "w"
    last_epoch_run = start_epoch - 1

    with history_path.open(write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        if write_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, max_epochs + 1):
            train_loss, train_acc = _run_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _validate_loss(model, val_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, device, NUM_CLASSES)
            last_epoch_run = epoch

            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
                "val_macro_f1": round(val_metrics["macro_f1"], 6),
            })
            f.flush()
            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} macro-F1 {val_metrics['macro_f1']:.4f}"
            )

            ckpt_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_without_improve": epochs_without_improve,
                "config": run_config,
            }
            torch.save(ckpt_payload, checkpoint_path)
            _update_manifest(manifest_path, config.experiment_id, {
                "status": "in_progress",
                "architecture": exp["architecture"],
                "strategy": exp["strategy"],
                "image_size": image_size,
                "last_epoch": epoch,
                "val_macro_f1": round(val_metrics["macro_f1"], 6),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improve = 0
                torch.save(ckpt_payload, best_path)
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= config.patience:
                    print(f"Early stopping at epoch {epoch} (patience {config.patience})")
                    break

    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model"]
        )

    test_metrics = evaluate_model(model, test_loader, device, NUM_CLASSES)
    metrics_path.write_text(json.dumps(test_metrics, indent=2))

    # Per-image predictions for the paired McNemar tests later. One extra
    # forward pass over ~1k test images, so the cost is seconds.
    y_true, y_pred = predict_all(model, test_loader, device)
    pd.DataFrame({
        "image_id": test_df["image_id"].tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    f1_named = {CLASS_NAMES[i]: round(test_metrics["per_class_f1"][i], 4) for i in range(NUM_CLASSES)}
    print(f"\n=== {config.experiment_id} test metrics (best checkpoint) ===")
    print(f"Accuracy          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1          {test_metrics['macro_f1']:.4f}")
    print(f"Per-class F1      {json.dumps(f1_named)}")

    _update_manifest(manifest_path, config.experiment_id, {
        "status": "complete",
        "architecture": exp["architecture"],
        "strategy": exp["strategy"],
        "image_size": image_size,
        "last_epoch": last_epoch_run,
        "test_macro_f1": round(test_metrics["macro_f1"], 6),
        "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 6),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "config": run_config,
        "test_metrics": test_metrics,
        "output_dir": str(output_dir),
        "skipped": False,
    }
