#!/usr/bin/env python3
"""McNemar's paired test with Holm-Bonferroni correction, in two families.

Dietterich (1998) found McNemar's to be the only test with acceptable Type-I
error when each model is trained once and evaluated on a single test set, which
is exactly this setup.

Two separate families, corrected independently:

  CONFIRMATORY (9 comparisons)  The pre-registered family over E1-E6. Three
      cross-architecture within strategy, six cross-strategy within
      architecture. This is the family the thesis's claims rest on.

  EXPLORATORY (12 comparisons)  Everything involving EfficientNet-B3. Added
      after the original six had already been registered and run, so pooling it
      with the confirmatory family would let a post-hoc addition change whether
      pre-registered results are significant. Corrected on its own instead.

Correcting the two families separately is the honest choice here, but it is a
choice, and it should be stated plainly in the write-up rather than buried.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "src"))

from ham10000.constants import experiment_label  # noqa: E402

CONFIRMATORY = [
    # architecture held constant within strategy
    ("E1", "E2"), ("E3", "E4"), ("E5", "E6"),
    # strategy varied within architecture
    ("E1", "E3"), ("E1", "E5"), ("E3", "E5"),
    ("E2", "E4"), ("E2", "E6"), ("E4", "E6"),
]

EXPLORATORY = [
    # B3 at 224 against each incumbent, strategy held constant
    ("E7", "E1"), ("E7", "E2"),
    ("E8", "E3"), ("E8", "E4"),
    ("E9", "E5"), ("E9", "E6"),
    # strategy varied within B3 at 224
    ("E7", "E8"), ("E7", "E9"), ("E8", "E9"),
    # resolution: same architecture, same strategy, 224 against 300
    ("E7", "E10"), ("E8", "E11"), ("E9", "E12"),
]


def load_predictions(exp_id: str) -> pd.DataFrame:
    path = RUNS / exp_id / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the notebook for {exp_id} before this one."
        )
    return pd.read_csv(path)


def compare(exp_a: str, exp_b: str) -> dict:
    a = load_predictions(exp_a)
    b = load_predictions(exp_b)

    # Join on image_id rather than trusting row order.
    merged = a.merge(b, on="image_id", suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(f"{exp_a} and {exp_b} do not cover the same test images.")
    if not (merged["y_true_a"] == merged["y_true_b"]).all():
        raise ValueError(f"{exp_a} and {exp_b} disagree on ground truth. Different splits?")

    a_ok = merged["y_pred_a"] == merged["y_true_a"]
    b_ok = merged["y_pred_b"] == merged["y_true_b"]

    both_right = int((a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    both_wrong = int((~a_ok & ~b_ok).sum())

    discordant = a_only + b_only
    # Exact binomial when the discordant count is small; the chi-squared
    # approximation is unreliable there.
    use_exact = discordant < 25
    result = mcnemar([[both_right, a_only], [b_only, both_wrong]],
                     exact=use_exact, correction=not use_exact)

    return {
        "exp_a": exp_a,
        "exp_b": exp_b,
        "label_a": experiment_label(exp_a),
        "label_b": experiment_label(exp_b),
        "n_test": len(merged),
        "both_correct": both_right,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "accuracy_a": round(float(a_ok.mean()), 6),
        "accuracy_b": round(float(b_ok.mean()), 6),
        "test_variant": "exact_binomial" if use_exact else "chi2_continuity_corrected",
        "statistic": float(result.statistic) if result.statistic is not None else None,
        "p_raw": float(result.pvalue),
    }


def have(exp: str) -> bool:
    return (RUNS / exp / "test_predictions.csv").exists()


def run_family(name: str, pairs: list[tuple[str, str]]) -> pd.DataFrame | None:
    """Holm-correct within one family.

    Skips the family entirely if any member is missing rather than silently
    correcting over a subset: dropping comparisons changes the correction and
    would quietly make the surviving ones look more significant than they are.
    """
    absent = sorted({e for pair in pairs for e in pair if not have(e)})
    if absent:
        print(f"  {name} family skipped, no predictions yet for: {', '.join(absent)}")
        return None

    rows = [compare(a, b) for a, b in pairs]
    corrected = multipletests([r["p_raw"] for r in rows], alpha=0.05, method="holm")
    for row, reject, p_adj in zip(rows, corrected[0], corrected[1]):
        row["family"] = name
        row["p_holm_corrected"] = float(p_adj)
        row["reject_at_0.05"] = bool(reject)
        row["better"] = (
            row["exp_a"] if row["a_only_correct"] > row["b_only_correct"] else row["exp_b"]
        )
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Confirmatory family: {len(CONFIRMATORY)} comparisons over E1-E6")
    print(f"Exploratory family:  {len(EXPLORATORY)} comparisons involving EfficientNet-B3")
    print()

    frames = [
        f for f in (run_family("confirmatory", CONFIRMATORY),
                    run_family("exploratory", EXPLORATORY))
        if f is not None
    ]
    if not frames:
        print("\nNothing to test yet. Finish the training notebooks first.")
        return

    out = pd.concat(frames, ignore_index=True)
    columns = [
        "family", "exp_a", "exp_b", "label_a", "label_b", "n_test",
        "both_correct", "a_only_correct", "b_only_correct", "both_wrong", "discordant",
        "accuracy_a", "accuracy_b", "better",
        "test_variant", "statistic", "p_raw", "p_holm_corrected", "reject_at_0.05",
    ]
    out = out[columns]

    out_path = RUNS / "mcnemar_results.csv"
    out.to_csv(out_path, index=False)

    print()
    for _, r in out.iterrows():
        verdict = "significant" if r["reject_at_0.05"] else "not significant"
        print(
            f"[{r['family'][:4]}] {r['exp_a']:>3} vs {r['exp_b']:<3}  "
            f"p={r['p_holm_corrected']:.4g}  {verdict:<15} "
            f"(better: {r['better']})"
        )

    print(f"\nWrote {out_path}")
    summary = {
        "confirmatory_significant": int(
            out[(out.family == "confirmatory") & out["reject_at_0.05"]].shape[0]
        ),
        "confirmatory_total": int(out[out.family == "confirmatory"].shape[0]),
        "exploratory_significant": int(
            out[(out.family == "exploratory") & out["reject_at_0.05"]].shape[0]
        ),
        "exploratory_total": int(out[out.family == "exploratory"].shape[0]),
    }
    (RUNS / "mcnemar_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
