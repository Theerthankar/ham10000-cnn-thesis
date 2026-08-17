#!/usr/bin/env python3
"""RQ3: parameter count, on-disk size, and CPU inference latency.

Needs no trained checkpoint and no GPU. These are properties of the
architecture, so the numbers come out the same whether or not training has run.

EfficientNet-B3 is timed at both 224 and 300 because latency scales with input
area, and quoting a single figure for it would hide that. Run this on a quiet
machine: on a shared or virtualised host the standard deviations end up large
enough to make the means hard to defend.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham10000.models import build_model, describe_model  # noqa: E402

# (architecture, input size) pairs matching how each one is actually trained.
CONFIGURATIONS = [
    ("mobilenet_v2", 224),
    ("resnet50", 224),
    ("efficientnet_b3", 224),
    ("efficientnet_b3", 300),
]

WARMUP_RUNS = 10
TIMED_RUNS = 100


def benchmark(architecture: str, image_size: int) -> dict:
    device = torch.device("cpu")
    model = build_model(architecture, num_classes=7).to(device)
    model.eval()

    state_dict_mb = sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 1e6
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    times_ms = []
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(dummy)
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            model(dummy)
            times_ms.append((time.perf_counter() - start) * 1000)

    structure = describe_model(architecture, num_classes=7)
    return {
        "architecture": architecture,
        "image_size": image_size,
        "trainable_params": structure["trainable_params"],
        "weight_layers": structure["weight_layers"],
        "dropout_p": structure["dropout_p"],
        "state_dict_mb": round(state_dict_mb, 3),
        "cpu_latency_ms_mean": round(statistics.mean(times_ms), 3),
        "cpu_latency_ms_median": round(statistics.median(times_ms), 3),
        "cpu_latency_ms_std": round(statistics.stdev(times_ms), 3),
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None,
                        help="Pin torch to N CPU threads. Use 1 for the most repeatable timings.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "efficiency.json")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    print(f"Host      {platform.processor() or platform.machine()}")
    print(f"Threads   {torch.get_num_threads()}")
    print(f"Torch     {torch.__version__}\n")

    results = [benchmark(arch, size) for arch, size in CONFIGURATIONS]

    for r in results:
        print(
            f"{r['architecture']:<18} @{r['image_size']}  "
            f"{r['trainable_params']:>11,} params  "
            f"{r['state_dict_mb']:>7.2f} MB  "
            f"{r['cpu_latency_ms_mean']:>7.1f} +/- {r['cpu_latency_ms_std']:.1f} ms"
        )

    payload = {
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
