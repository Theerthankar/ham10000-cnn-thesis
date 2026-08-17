"""Lesion-level stratified 80/10/10 split (proposal Chapter 4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import CLASS_NAMES, DX_TO_IDX

SPLIT_SEED = 42


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    df = df[df["dx"].isin(CLASS_NAMES)].copy()
    df["label"] = df["dx"].map(DX_TO_IDX)
    return df


def make_lesion_level_split(
    df: pd.DataFrame,
    seed: int = SPLIT_SEED,
    test_size: float = 0.10,
    val_ratio_of_remainder: float = 0.10 / 0.90,
) -> dict[str, list[str]]:
    """Split by lesion_id; all images of a lesion stay in one partition."""
    lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]].reset_index(drop=True)
    train_lesions, test_lesions = train_test_split(
        lesions,
        test_size=test_size,
        stratify=lesions["dx"],
        random_state=seed,
    )
    train_lesions, val_lesions = train_test_split(
        train_lesions,
        test_size=val_ratio_of_remainder,
        stratify=train_lesions["dx"],
        random_state=seed,
    )
    return {
        "train": train_lesions["lesion_id"].tolist(),
        "val": val_lesions["lesion_id"].tolist(),
        "test": test_lesions["lesion_id"].tolist(),
    }


def assign_split_column(df: pd.DataFrame, lesion_splits: dict[str, list[str]]) -> pd.DataFrame:
    lesion_to_split = {}
    for split_name, lesion_ids in lesion_splits.items():
        for lid in lesion_ids:
            lesion_to_split[lid] = split_name
    out = df.copy()
    out["split"] = out["lesion_id"].map(lesion_to_split)
    if out["split"].isna().any():
        raise ValueError("Some lesions were not assigned to a split.")
    return out


def split_hash(lesion_splits: dict[str, list[str]]) -> str:
    payload = json.dumps(lesion_splits, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_split(lesion_splits: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SPLIT_SEED,
        "protocol": "lesion_level_stratified_80_10_10",
        "hash": split_hash(lesion_splits),
        "lesion_ids": lesion_splits,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_or_create_split(metadata_path: Path, split_path: Path) -> tuple[pd.DataFrame, dict]:
    df = load_metadata(metadata_path)
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        lesion_splits = payload["lesion_ids"]
    else:
        lesion_splits = make_lesion_level_split(df)
        save_split(lesion_splits, split_path)
    df = assign_split_column(df, lesion_splits)
    return df, lesion_splits
