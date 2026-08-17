"""Image path resolution and PyTorch Dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def resolve_image_path(data_root: Path, image_id: str) -> Path:
    for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        path = data_root / folder / f"{image_id}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for {image_id} under {data_root}")


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = resolve_image_path(self.data_root, row["image_id"])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label
