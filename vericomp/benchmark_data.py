"""Local data loaders for JanusDNA's NT and GenomicBenchmarks tasks."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


NT_SPECS = {
    "H3K79me3": (500, 2, "mcc"), "enhancers": (200, 2, "mcc"),
    "enhancers_types": (200, 3, "mcc"), "H3": (500, 2, "mcc"),
    "H3K4me1": (500, 2, "mcc"), "H3K4me2": (500, 2, "mcc"),
    "H3K4me3": (500, 2, "mcc"), "H3K9ac": (500, 2, "mcc"),
    "H3K14ac": (500, 2, "mcc"), "H3K36me3": (500, 2, "mcc"),
    "H4": (500, 2, "mcc"), "H4ac": (500, 2, "mcc"),
    "promoter_all": (300, 2, "f1_binary"),
    "promoter_no_tata": (300, 2, "f1_binary"),
    "promoter_tata": (300, 2, "f1_binary"),
    "splice_sites_acceptors": (600, 2, "f1_binary"),
    "splice_sites_all": (400, 3, "accuracy"),
    "splice_sites_donors": (600, 2, "f1_binary"),
}

GB_SPECS = {
    "dummy_mouse_enhancers_ensembl": (1024, 2, "mcc"),
    "demo_coding_vs_intergenomic_seqs": (200, 2, "mcc"),
    "demo_human_or_worm": (200, 2, "mcc"),
    "human_enhancers_cohn": (500, 2, "mcc"),
    "human_enhancers_ensembl": (512, 2, "mcc"),
    "human_ensembl_regulatory": (512, 3, "accuracy"),
    "human_nontata_promoters": (251, 2, "f1_binary"),
    "human_ocr_ensembl": (512, 2, "mcc"),
}


_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
class SequenceClassificationDataset(Dataset):
    def __init__(self, records, max_length: int):
        self.records = records
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sequence, label = self.records[int(index)]
        sequence = sequence[: self.max_length].upper()
        sequence = sequence.rjust(self.max_length, "N")
        ids = torch.tensor([_BASE_TO_ID.get(base, 5) for base in sequence], dtype=torch.long)
        return {"input_ids": ids, "labels": torch.tensor(int(label), dtype=torch.long)}


def _limit(records, limit):
    if limit is None or int(limit) <= 0:
        return records
    return records[: int(limit)]


def _split_records(records, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(records), generator=generator).tolist()
    n_valid = max(1, int(round(len(records) * 0.1)))
    valid_ids = set(permutation[:n_valid])
    train = [record for index, record in enumerate(records) if index not in valid_ids]
    valid = [record for index, record in enumerate(records) if index in valid_ids]
    return train, valid


def _kfold_split_records(records, fold: int, num_folds: int):
    """Deterministic k-fold split: fold ``fold`` is validation, the rest is train.

    The shuffle uses a fixed seed so every fold of the same dataset always
    partitions the records identically across runs.
    """
    num_folds = max(2, int(num_folds))
    fold = int(fold) % num_folds
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    folds = [permutation[index::num_folds] for index in range(num_folds)]
    valid_ids = set(folds[fold])
    train = [record for index, record in enumerate(records) if index not in valid_ids]
    valid = [record for index, record in enumerate(records) if index in valid_ids]
    return train, valid


def load_nt_splits(root: str, dataset_name: str, fold: int = 0,
                   max_train: int | None = None, max_valid: int | None = None,
                   max_test: int | None = None, num_folds: int = 5):
    if dataset_name not in NT_SPECS:
        raise KeyError(f"Unknown NT dataset {dataset_name}; available={sorted(NT_SPECS)}")
    max_length, num_classes, metric = NT_SPECS[dataset_name]
    task_root = Path(root).expanduser().resolve() / dataset_name
    train_path, test_path = task_root / "train.parquet", task_root / "test.parquet"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"NT data missing under {task_root}; run tools/download_benchmark_data.py")
    train_table = load_dataset("parquet", data_files=str(train_path), split="train")
    test_table = load_dataset("parquet", data_files=str(test_path), split="train")
    records = [(str(row["sequence"]), int(row["label"])) for row in train_table]
    test_records = [(str(row["sequence"]), int(row["label"])) for row in test_table]
    train_records, valid_records = _kfold_split_records(records, fold, num_folds)
    return (
        SequenceClassificationDataset(_limit(train_records, max_train), max_length),
        SequenceClassificationDataset(_limit(valid_records, max_valid), max_length),
        SequenceClassificationDataset(_limit(test_records, max_test), max_length),
        {"max_length": max_length, "num_classes": num_classes, "metric": metric,
         "fold": int(fold), "num_folds": int(num_folds)},
    )


def _read_gb_split(task_root: Path, split: str):
    split_root = task_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    records = []
    for label, class_dir in enumerate(class_dirs):
        for sequence_path in sorted(class_dir.iterdir()):
            if sequence_path.is_file():
                records.append((sequence_path.read_text().strip(), label))
    return records


def load_gb_splits(root: str, dataset_name: str, fold: int = 0,
                   max_train: int | None = None, max_valid: int | None = None,
                   max_test: int | None = None, num_folds: int = 5):
    if dataset_name not in GB_SPECS:
        raise KeyError(f"Unknown GB dataset {dataset_name}; available={sorted(GB_SPECS)}")
    max_length, num_classes, metric = GB_SPECS[dataset_name]
    task_root = Path(root).expanduser().resolve() / dataset_name
    records = _read_gb_split(task_root, "train")
    test_records = _read_gb_split(task_root, "test")
    train_records, valid_records = _kfold_split_records(records, fold, num_folds)
    return (
        SequenceClassificationDataset(_limit(train_records, max_train), max_length),
        SequenceClassificationDataset(_limit(valid_records, max_valid), max_length),
        SequenceClassificationDataset(_limit(test_records, max_test), max_length),
        {"max_length": max_length, "num_classes": num_classes, "metric": metric,
         "fold": int(fold), "num_folds": int(num_folds)},
    )
