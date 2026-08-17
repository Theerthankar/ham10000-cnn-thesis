#!/usr/bin/env python3
"""Build every figure the write-up and the defence slides need.

Reads only what is already on disk under runs/, so it is safe to call at any
point. Experiments that have not finished yet are skipped with a note rather
than crashing the whole run.

Each figure is written as both PNG (for slides) and PDF (vector, for LaTeX).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham10000.constants import (  # noqa: E402
    ABLATION_EXPERIMENTS, ARCHITECTURE_LABELS, CLASS_NAMES,
    CONFIRMATORY_EXPERIMENTS, EXPERIMENTS, STRATEGY_LABELS,
)

RUNS = ROOT / "runs"
FIGDIR = ROOT / "figures"

ARCH_COLOR = {
    "mobilenet_v2": "#D54407",
    "resnet50": "#0D1B2A",
    "efficientnet_b3": "#2E7D32",
}
STRATEGY_HATCH = {"augmentation": "", "weighted_ce": "//", "oversampling": ".."}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .pdf")


def available() -> list[str]:
    return [e for e in EXPERIMENTS if (RUNS / e / "test_metrics.json").exists()]


def load_metrics(exp: str) -> dict:
    return json.loads((RUNS / exp / "test_metrics.json").read_text())


def per_class(exp: str, metric: str) -> dict:
    """metric is 'precision', 'recall' or 'f1-score'.

    Missing classes fall back to 0.0. See the matching note in build_tables.py:
    only affects metrics files written before classification_report was given an
    explicit label list, but a KeyError here would kill every figure at once.
    """
    report = load_metrics(exp)["classification_report"]
    return {
        name: float(report[str(i)][metric]) if str(i) in report else 0.0
        for i, name in enumerate(CLASS_NAMES)
    }


def short(exp: str) -> str:
    e = EXPERIMENTS[exp]
    arch = {"mobilenet_v2": "MNv2", "resnet50": "RN50", "efficientnet_b3": "ENb3"}[e["architecture"]]
    size = f"@{e['image_size']}" if e["architecture"] == "efficientnet_b3" else ""
    return f"{exp}\n{arch}{size}"


def rq2_score(exp: str) -> float:
    r = per_class(exp, "recall")
    return (r["mel"] + r["bcc"]) / 2


# --------------------------------------------------------------------------
# 1. Dataset composition
# --------------------------------------------------------------------------
def fig_class_distribution(exps: list[str]) -> None:
    split_file = RUNS / "splits" / "seed42_lesion_stratified.json"
    meta = ROOT / "data" / "HAM10000_metadata.csv"
    if not meta.exists():
        print("  skip class distribution: data/HAM10000_metadata.csv not found")
        return
    df = pd.read_csv(meta)
    counts = df["dx"].value_counts().reindex(CLASS_NAMES).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([c.upper() for c in CLASS_NAMES], counts.values, color="#0D1B2A")
    bars[CLASS_NAMES.index("mel")].set_color("#D54407")
    bars[CLASS_NAMES.index("bcc")].set_color("#D54407")
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{int(v)}\n{v / total * 100:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title(f"HAM10000 class distribution (n={int(total):,})")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "01_class_distribution")


# --------------------------------------------------------------------------
# 2. Training curves, one panel per architecture
# --------------------------------------------------------------------------
def fig_training_curves(exps: list[str]) -> None:
    archs = ["mobilenet_v2", "resnet50", "efficientnet_b3"]
    groups = {a: [e for e in exps if EXPERIMENTS[e]["architecture"] == a] for a in archs}
    groups = {a: v for a, v in groups.items() if v}
    if not groups:
        return

    fig, axes = plt.subplots(2, len(groups), figsize=(5.2 * len(groups), 7), squeeze=False)
    for col, (arch, members) in enumerate(groups.items()):
        for exp in members:
            hist_path = RUNS / exp / "history.csv"
            if not hist_path.exists():
                continue
            h = pd.read_csv(hist_path)
            style = {"augmentation": "-", "weighted_ce": "--", "oversampling": ":"}[
                EXPERIMENTS[exp]["strategy"]]
            label = f"{exp} {STRATEGY_LABELS[EXPERIMENTS[exp]['strategy']]}"
            if EXPERIMENTS[exp]["architecture"] == "efficientnet_b3":
                label += f" @{EXPERIMENTS[exp]['image_size']}"
            axes[0][col].plot(h["epoch"], h["val_loss"], style, label=label,
                              color=ARCH_COLOR[arch], alpha=0.85)
            axes[1][col].plot(h["epoch"], h["val_macro_f1"], style, label=label,
                              color=ARCH_COLOR[arch], alpha=0.85)
        axes[0][col].set_title(ARCHITECTURE_LABELS[arch])
        axes[0][col].set_ylabel("Validation loss")
        axes[1][col].set_ylabel("Validation macro F1")
        axes[1][col].set_xlabel("Epoch")
        axes[0][col].legend(fontsize=7)
    fig.suptitle("Optimisation behaviour by architecture", y=0.98)
    save(fig, "02_training_curves")


# --------------------------------------------------------------------------
# 3. Per-class F1 heatmap across every finished run
# --------------------------------------------------------------------------
def fig_per_class_heatmap(exps: list[str]) -> None:
    if not exps:
        return
    mat = np.array([[per_class(e, "f1-score")[c] for c in CLASS_NAMES] for e in exps])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(exps) + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES)), [c.upper() for c in CLASS_NAMES])
    ax.set_yticks(range(len(exps)), [short(e).replace("\n", " ") for e in exps], fontsize=8)
    for i in range(len(exps)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title("Per-class F1")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="F1")
    save(fig, "03_per_class_f1_heatmap")


# --------------------------------------------------------------------------
# 4. Confusion matrices
# --------------------------------------------------------------------------
def fig_confusion_matrices(exps: list[str]) -> None:
    if not exps:
        return
    cols = 3
    rows = int(np.ceil(len(exps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.9 * rows), squeeze=False)
    for idx, exp in enumerate(exps):
        ax = axes[idx // cols][idx % cols]
        cm = np.array(load_metrics(exp)["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(7), [c.upper() for c in CLASS_NAMES], rotation=45, fontsize=6)
        ax.set_yticks(range(7), [c.upper() for c in CLASS_NAMES], fontsize=6)
        ax.set_title(short(exp).replace("\n", "  "), fontsize=9)
        ax.grid(False)
    for k in range(len(exps), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Confusion matrices (row-normalised, counts annotated)", y=1.0)
    save(fig, "04_confusion_matrices")


# --------------------------------------------------------------------------
# 5. Melanoma precision/recall trade-off
# --------------------------------------------------------------------------
def fig_mel_tradeoff(exps: list[str]) -> None:
    if not exps:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for exp in exps:
        p = per_class(exp, "precision")["mel"]
        r = per_class(exp, "recall")["mel"]
        arch = EXPERIMENTS[exp]["architecture"]
        ax.scatter(r, p, s=130, color=ARCH_COLOR[arch], edgecolor="white", zorder=3,
                   marker={"augmentation": "o", "weighted_ce": "s", "oversampling": "^"}[
                       EXPERIMENTS[exp]["strategy"]])
        ax.annotate(exp, (r, p), textcoords="offset points", xytext=(7, 4), fontsize=8)

    for f1 in (0.3, 0.4, 0.5, 0.6):
        r = np.linspace(0.05, 0.99, 200)
        p = (f1 * r) / np.maximum(2 * r - f1, 1e-9)
        ok = (p > 0) & (p <= 1)
        ax.plot(r[ok], p[ok], color="grey", lw=0.6, ls=":", zorder=1)
        if ok.any():
            ax.annotate(f"F1={f1}", (r[ok][-1], p[ok][-1]), fontsize=7, color="grey")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=ARCHITECTURE_LABELS[a])
               for a, c in ARCH_COLOR.items()]
    handles += [
        plt.Line2D([], [], marker=m, ls="", color="grey", label=STRATEGY_LABELS[s])
        for s, m in [("augmentation", "o"), ("weighted_ce", "s"), ("oversampling", "^")]
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlabel("Melanoma recall")
    ax.set_ylabel("Melanoma precision")
    ax.set_title("What each configuration trades away on melanoma")
    save(fig, "05_melanoma_precision_recall")


# --------------------------------------------------------------------------
# 6. RQ2 selection criterion
# --------------------------------------------------------------------------
def fig_rq2_scores(exps: list[str]) -> None:
    if not exps:
        return
    scores = [rq2_score(e) for e in exps]
    colors = [ARCH_COLOR[EXPERIMENTS[e]["architecture"]] for e in exps]
    hatches = [STRATEGY_HATCH[EXPERIMENTS[e]["strategy"]] for e in exps]

    # Best per architecture gets an outline.
    best = {}
    for e, s in zip(exps, scores):
        a = EXPERIMENTS[e]["architecture"]
        if a not in best or s > best[a][1]:
            best[a] = (e, s)
    winners = {e for e, _ in best.values()}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(exps)), scores, color=colors)
    for b, h, e in zip(bars, hatches, exps):
        b.set_hatch(h)
        if e in winners:
            b.set_edgecolor("black")
            b.set_linewidth(2.2)
    for i, (s, e) in enumerate(zip(scores, exps)):
        ax.text(i, s + 0.008, f"{s:.3f}", ha="center", fontsize=8,
                fontweight="bold" if e in winners else "normal")
    ax.set_xticks(range(len(exps)), [short(e) for e in exps], fontsize=7)
    ax.set_ylabel("mean(recall MEL, recall BCC)")
    ax.set_title("RQ2 selection criterion. Outlined bar is the best run for that architecture.")
    ax.set_ylim(0, max(scores) * 1.15)
    save(fig, "06_rq2_criterion")


# --------------------------------------------------------------------------
# 7. Deployability: size against clinical performance
# --------------------------------------------------------------------------
def fig_deployability(exps: list[str]) -> None:
    eff_path = RUNS / "efficiency.json"
    if not eff_path.exists() or not exps:
        print("  skip deployability: run benchmark_efficiency.py first")
        return
    eff = json.loads(eff_path.read_text())["results"]
    size_by = {(r["architecture"], r["image_size"]): r for r in eff}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Runs sharing an architecture also share a size and a latency, so their
    # labels land on the same spot. Fan them out by rank within each column.
    seen: dict[tuple, int] = {}
    for exp in exps:
        e = EXPERIMENTS[exp]
        key = (e["architecture"], e["image_size"])
        if key not in size_by:
            continue
        mb = size_by[key]["state_dict_mb"]
        lat = size_by[key]["cpu_latency_ms_mean"]
        f1_mel = per_class(exp, "f1-score")["mel"]
        marker = {"augmentation": "o", "weighted_ce": "s", "oversampling": "^"}[e["strategy"]]
        rank = seen.get(key, 0)
        seen[key] = rank + 1
        for ax, x in ((axes[0], mb), (axes[1], lat)):
            ax.scatter(x, f1_mel, s=130, marker=marker, color=ARCH_COLOR[e["architecture"]],
                       edgecolor="white", zorder=3)
            ax.annotate(exp, (x, f1_mel), textcoords="offset points",
                        xytext=(8, 5 - 11 * (rank % 2)), fontsize=8)

    axes[0].set_xlabel("Model size on disk (MB)")
    axes[1].set_xlabel("CPU latency (ms/image)")
    for ax in axes:
        ax.set_ylabel("Melanoma F1")
    axes[0].set_title("Smaller is better, higher is better")
    axes[1].set_title("Faster is better, higher is better")
    fig.suptitle("Deployment cost against melanoma detection", y=1.01)
    save(fig, "07_deployability")


# --------------------------------------------------------------------------
# 8. Resolution ablation: B3 at 224 against B3 at 300
# --------------------------------------------------------------------------
def fig_resolution_ablation(exps: list[str]) -> None:
    pairs = [("E7", "E10", "augmentation"), ("E8", "E11", "weighted_ce"),
             ("E9", "E12", "oversampling")]
    pairs = [p for p in pairs if p[0] in exps and p[1] in exps]
    if not pairs:
        print("  skip resolution ablation: needs E7-E12")
        return

    panels = [
        ("macro_f1", "Macro F1", lambda e: load_metrics(e)["macro_f1"]),
        ("balanced_accuracy", "Balanced accuracy", lambda e: load_metrics(e)["balanced_accuracy"]),
        ("rq2", "RQ2 criterion (mean clinical recall)", rq2_score),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    width = 0.35
    x = np.arange(len(pairs))

    for ax, (_, title, getter) in zip(axes, panels):
        v224 = [getter(a) for a, _, _ in pairs]
        v300 = [getter(b) for _, b, _ in pairs]
        ax.bar(x - width / 2, v224, width, label="224px", color="#90A4AE")
        ax.bar(x + width / 2, v300, width, label="300px (native)", color="#2E7D32")
        for i, (a, b) in enumerate(zip(v224, v300)):
            ax.text(i - width / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=7)
            ax.text(i + width / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=7)
            # mark which side of the pair actually won
            better = "300" if b > a else "224"
            ax.text(i, -0.135, f"{better} ahead", ha="center", fontsize=7,
                    color="#2E7D32" if better == "300" else "#546E7A",
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(x, [STRATEGY_LABELS[s] for _, _, s in pairs], fontsize=8)
        ax.set_title(title, fontsize=11)
        # headroom so the legend never sits on top of a bar
        ax.set_ylim(0, max(v224 + v300) * 1.32)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)

    fig.suptitle("EfficientNet-B3: does the native 300px input actually help?", y=1.0)
    fig.tight_layout()
    save(fig, "08_resolution_ablation")


# --------------------------------------------------------------------------
# 9. Frequency tiers
# --------------------------------------------------------------------------
def fig_frequency_tiers(exps: list[str]) -> None:
    if not exps:
        return
    majority, minority = ["nv", "bkl", "mel"], ["akiec", "vasc", "df"]
    maj = [np.mean([per_class(e, "f1-score")[c] for c in majority]) for e in exps]
    mino = [np.mean([per_class(e, "f1-score")[c] for c in minority]) for e in exps]

    x = np.arange(len(exps))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width / 2, maj, width, label="Majority tier (NV/BKL/MEL)", color="#0D1B2A")
    ax.bar(x + width / 2, mino, width, label="Minority tier (AKIEC/VASC/DF)", color="#D54407")
    for i, (a, b) in enumerate(zip(maj, mino)):
        if b > a:
            ax.annotate("minority ahead", (i, max(a, b) + 0.03), ha="center", fontsize=7,
                        color="#2E7D32", fontweight="bold")
    ax.set_xticks(x, [short(e) for e in exps], fontsize=7)
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Where each configuration spends its capacity")
    save(fig, "09_frequency_tiers")


# --------------------------------------------------------------------------
# 10. McNemar results
# --------------------------------------------------------------------------
def fig_mcnemar(exps: list[str]) -> None:
    path = RUNS / "mcnemar_results.csv"
    if not path.exists():
        print("  skip McNemar figure: run statistical_tests.py first")
        return
    df = pd.read_csv(path)
    fams = [f for f in ("confirmatory", "exploratory") if f in set(df.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.5 * len(fams), 5), squeeze=False)

    for ax, fam in zip(axes[0], fams):
        sub = df[df.family == fam].copy()
        sub["pair"] = sub.exp_a + " vs " + sub.exp_b
        sub = sub.sort_values("p_holm_corrected")
        colors = ["#2E7D32" if r else "#C62828" for r in sub["reject_at_0.05"]]
        y = np.arange(len(sub))
        ax.barh(y, -np.log10(np.maximum(sub.p_holm_corrected, 1e-30)), color=colors)
        ax.axvline(-np.log10(0.05), ls="--", color="black", lw=1)
        ax.text(-np.log10(0.05), len(sub) - 0.4, "  p = 0.05", fontsize=7, va="top")
        ax.set_yticks(y, sub["pair"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(Holm-corrected p)")
        ax.set_title(f"{fam.capitalize()} family ({len(sub)} comparisons)")
    fig.suptitle("McNemar's paired test. Green passes at alpha = 0.05.", y=1.02)
    save(fig, "10_mcnemar")


# --------------------------------------------------------------------------
# 11. Training cost
# --------------------------------------------------------------------------
def fig_training_cost(exps: list[str]) -> None:
    rows = []
    for e in exps:
        h = RUNS / e / "history.csv"
        if h.exists():
            rows.append((e, len(pd.read_csv(h))))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([short(e) for e, _ in rows], [n for _, n in rows],
           color=[ARCH_COLOR[EXPERIMENTS[e]["architecture"]] for e, _ in rows])
    for i, (_, n) in enumerate(rows):
        ax.text(i, n + 0.4, str(n), ha="center", fontsize=8)
    ax.set_ylabel("Epochs before early stopping")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("How long each configuration needed (max 50, patience 10)")
    save(fig, "11_training_cost")


def main() -> None:
    exps = available()
    if not exps:
        print("No finished experiments under runs/. Train something first.")
        return
    print(f"Found {len(exps)} finished experiments: {', '.join(exps)}\n")

    fig_class_distribution(exps)
    fig_training_curves(exps)
    fig_per_class_heatmap(exps)
    fig_confusion_matrices(exps)
    fig_mel_tradeoff(exps)
    fig_rq2_scores(exps)
    fig_deployability(exps)
    fig_resolution_ablation(exps)
    fig_frequency_tiers(exps)
    fig_mcnemar(exps)
    fig_training_cost(exps)

    missing = [e for e in EXPERIMENTS if e not in exps]
    if missing:
        print(f"\nStill missing: {', '.join(missing)}")
    print(f"\nAll figures are in {FIGDIR}")


if __name__ == "__main__":
    main()
