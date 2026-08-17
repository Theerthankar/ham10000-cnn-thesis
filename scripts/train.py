#!/usr/bin/env python3
"""Command-line entry point for a single experiment.

The notebooks are the primary record of how the results were produced. This
exists for smoke tests and for re-running something from a terminal without
opening Jupyter.

    python scripts/train.py --experiment E7 --data-root data
    python scripts/train.py --experiment E1 --smoke     # 2 epochs, tiny subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham10000.constants import EXPERIMENTS, experiment_label  # noqa: E402
from ham10000.train import TrainConfig, train_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on a few hundred images, to prove the wiring works.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if test_metrics.json already exists.")
    args = parser.parse_args()

    output_dir = args.output_dir or str(ROOT / "runs" / args.experiment)
    print(f"Running {experiment_label(args.experiment)}")

    config = TrainConfig(
        experiment_id=args.experiment,
        data_root=args.data_root,
        output_dir=output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        smoke_test=args.smoke,
        force_rerun=args.force,
    )
    train_experiment(config)


if __name__ == "__main__":
    main()
