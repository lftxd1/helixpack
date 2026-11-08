"""Local DNALongBench CMP dataset wrapper."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


CMP_SUBSETS = {
    "HFF": 0,
    "H1hESC": 1,
    "GM12878": 2,
    "IMR90": 3,
    "HCT116": 4,
}


def normalize_cmp_subset(subset: str) -> str:
    value = str(subset).strip()
    for name in CMP_SUBSETS:
        if value.lower() == name.lower():
            return name
    choices = ", ".join(CMP_SUBSETS)
    raise ValueError(f"unknown CMP subset {subset!r}; expected one of: {choices}")


class CMPDataset(Dataset):
    """Convert the public Arrow CMP split to VeriComp token IDs on demand."""

    def __init__(self, split, target_index: int, max_samples: int | None = None):
        self.split = split
        self.target_index = int(target_index)
        self.length = len(split) if not max_samples or max_samples <= 0 else min(len(split), int(max_samples))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.split[int(index)]
        sequence = torch.as_tensor(row["sequence"], dtype=torch.long)
        target = torch.as_tensor(row["label_ut"], dtype=torch.float32)
        target = target[:, self.target_index]
        mask = torch.as_tensor(row["mask_ut"], dtype=torch.bool)
        mask = mask[:, self.target_index]
        return {
            "sample_id": str(row["sample_id"]),
            "input_ids": sequence,
            "target": target,
            "mask": mask,
        }


def load_cmp_splits(data_root: str, subset: str = "HFF",
                    max_train: int | None = None, max_valid: int | None = None,
                    max_test: int | None = None,
                    columns: tuple[str, ...] | None = None):
    subset = normalize_cmp_subset(subset)
    target_index = CMP_SUBSETS[subset]
    root = Path(data_root).expanduser().resolve() / "cmp"
    if not root.exists():
        raise FileNotFoundError(f"CMP data not found at {root}; run tools/download_dnalongbench.py")
    dataset = load_from_disk(str(root))
    if not hasattr(dataset, "keys"):
        raise ValueError(f"Expected DatasetDict at {root}")
    def get(*names):
        for name in names:
            if name in dataset:
                split = dataset[name]
                return split.select_columns(list(columns)) if columns else split
        raise KeyError(f"Missing CMP split {names}; available={list(dataset.keys())}")
    return (
        CMPDataset(get("train"), target_index, max_train),
        CMPDataset(get("validation", "valid"), target_index, max_valid),
        CMPDataset(get("test"), target_index, max_test),
    )
