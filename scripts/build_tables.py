#!/usr/bin/env python3
"""Turn runs/ into the CSV and LaTeX tables the write-up needs.

Everything here is derived from test_metrics.json, so the tables and the figures
can never drift apart from each other or from what actually ran. Nothing is
typed in by hand.

Outputs land in tables/:
    headline.csv / .tex          accuracy, balanced acc, macro F1, macro AUC
    clinical.csv / .tex          precision, recall, F1 for MEL and BCC
    per_class_f1.csv / .tex      all seven classes
    rq_answers.json              RQ1/RQ2 resolved from the numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ham10000.constants import (  # noqa: E402
    ARCHITECTURE_LABELS, CLASS_NAMES, CONFIRMATORY_EXPERIMENTS,
    EXPERIMENTS, STRATEGY_LABELS,
)

RUNS = ROOT / "runs"
TABLES = ROOT / "tables"
RQ1_THRESHOLD = 0.05


def finished() -> list[str]:
    return [e for e in EXPERIMENTS if (RUNS / e / "test_metrics.json").exists()]


def metrics(exp: str) -> dict:
    return json.loads((RUNS / exp / "test_metrics.json").read_text())


def report(exp: str, cls: str, field: str) -> float:
    """Per-class precision/recall/F1 from the stored classification report.

    Falls back to 0.0 for a class the report omits. That only happens for
    metrics files written before `labels=` was passed to classification_report,
    and only when a class was absent from the evaluated subset, but crashing the
    whole table build over one missing key is worse than reporting the zero.
    """
    entry = metrics(exp)["classification_report"].get(str(CLASS_NAMES.index(cls)))
    return float(entry[field]) if entry else 0.0


def rq2_score(exp: str) -> float:
    return (report(exp, "mel", "recall") + report(exp, "bcc", "recall")) / 2


def epochs_run(exp: str) -> int:
    path = RUNS / exp / "history.csv"
    return len(pd.read_csv(path)) if path.exists() else 0


def write(df: pd.DataFrame, name: str, caption: str, float_fmt: str = "%.3f") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(
        df.to_latex(index=False, float_format=float_fmt, caption=caption,
                    label=f"tab:{name}", escape=True)
    )
    print(f"  tables/{name}.csv and .tex")


def main() -> None:
    exps = finished()
    if not exps:
        print("Nothing finished under runs/ yet.")
        return
    print(f"Building tables from {len(exps)} runs\n")

    headline = pd.DataFrame([{
        "Exp": e,
        "Architecture": ARCHITECTURE_LABELS[EXPERIMENTS[e]["architecture"]],
        "Input": EXPERIMENTS[e]["image_size"],
        "Strategy": STRATEGY_LABELS[EXPERIMENTS[e]["strategy"]],
        "Epochs": epochs_run(e),
        "Accuracy": metrics(e)["accuracy"],
        "Balanced acc": metrics(e)["balanced_accuracy"],
        "Macro F1": metrics(e)["macro_f1"],
        "Macro AUC": metrics(e).get("macro_auc_ovr"),
    } for e in exps])
    write(headline, "headline", "Headline test-set metrics for every run.")

    clinical = pd.DataFrame([{
        "Exp": e,
        "Architecture": ARCHITECTURE_LABELS[EXPERIMENTS[e]["architecture"]],
        "Input": EXPERIMENTS[e]["image_size"],
        "Strategy": STRATEGY_LABELS[EXPERIMENTS[e]["strategy"]],
        "P (MEL)": report(e, "mel", "precision"),
        "R (MEL)": report(e, "mel", "recall"),
        "F1 (MEL)": report(e, "mel", "f1-score"),
        "P (BCC)": report(e, "bcc", "precision"),
        "R (BCC)": report(e, "bcc", "recall"),
        "F1 (BCC)": report(e, "bcc", "f1-score"),
        "RQ2 score": rq2_score(e),
    } for e in exps])
    write(clinical, "clinical",
          "Precision, recall and F1 on the two clinical classes. "
          "The final column is the RQ2 selection criterion.")

    per_class = pd.DataFrame([
        {"Exp": e, **{c.upper(): report(e, c, "f1-score") for c in CLASS_NAMES}}
        for e in exps
    ])
    write(per_class, "per_class_f1", "Per-class F1 for all seven classes.")

    # --- RQ answers, computed rather than asserted ---
    answers: dict = {"rq1": {}, "rq2": {}, "notes": []}

    by_arch: dict[str, list[str]] = {}
    for e in exps:
        by_arch.setdefault(EXPERIMENTS[e]["architecture"], []).append(e)

    answers["rq2"] = {
        "criterion": "mean(recall_MEL, recall_BCC)",
        "scores": {e: round(rq2_score(e), 6) for e in exps},
        "best_per_architecture": {
            arch: max(members, key=rq2_score) for arch, members in by_arch.items()
        },
    }

    # RQ1 is defined over the confirmatory pair only. B3 is an ablation and does
    # not replace either incumbent in the pre-registered question.
    conf = [e for e in exps if e in CONFIRMATORY_EXPERIMENTS]
    mob = [e for e in conf if EXPERIMENTS[e]["architecture"] == "mobilenet_v2"]
    res = [e for e in conf if EXPERIMENTS[e]["architecture"] == "resnet50"]
    if mob and res:
        best_m, best_r = max(mob, key=rq2_score), max(res, key=rq2_score)
        gap = abs(report(best_m, "mel", "f1-score") - report(best_r, "mel", "f1-score"))
        answers["rq1"] = {
            "best_mobilenet": best_m,
            "best_resnet": best_r,
            "f1_mel_mobilenet": round(report(best_m, "mel", "f1-score"), 6),
            "f1_mel_resnet": round(report(best_r, "mel", "f1-score"), 6),
            "gap": round(gap, 6),
            "threshold": RQ1_THRESHOLD,
            "passes": bool(gap <= RQ1_THRESHOLD),
        }
        # Robustness check against the alternative reading of "best".
        alt_m = max(mob, key=lambda e: metrics(e)["macro_f1"])
        alt_r = max(res, key=lambda e: metrics(e)["macro_f1"])
        alt_gap = abs(report(alt_m, "mel", "f1-score") - report(alt_r, "mel", "f1-score"))
        answers["rq1"]["macro_f1_best_alternative"] = {
            "best_mobilenet": alt_m, "best_resnet": alt_r,
            "gap": round(alt_gap, 6), "passes": bool(alt_gap <= RQ1_THRESHOLD),
        }

    (TABLES / "rq_answers.json").write_text(json.dumps(answers, indent=2))
    print("  tables/rq_answers.json")
    print("\n" + json.dumps(answers.get("rq1", {}), indent=2))


if __name__ == "__main__":
    main()
