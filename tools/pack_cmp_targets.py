#!/usr/bin/env python3
"""Pack one CMP cell-line target column per split for fast fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


SUBSETS = {"HFF": 0, "H1hESC": 1, "GM12878": 2, "IMR90": 3, "HCT116": 4}


def atomic_save(value: torch.Tensor, path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(value, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def pack_split(split, target_index: int) -> torch.Tensor:
    column = split.data.column("label_ut")
    first = next(column.iterchunks()).to_numpy(zero_copy_only=False)
    output = torch.empty((len(split), first.shape[1]), dtype=torch.float32)
    offset = 0
    for chunk in column.iterchunks():
        values = chunk.to_numpy(zero_copy_only=False)
        selected = np.asarray(values[:, :, target_index], dtype=np.float32).copy()
        tensor = torch.from_numpy(selected)
        output[offset:offset + tensor.shape[0]].copy_(tensor)
        offset += tensor.shape[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subset", choices=tuple(SUBSETS), required=True)
    args = parser.parse_args()

    dataset = load_from_disk(str((args.data_root / "cmp").resolve()))
    target_index = SUBSETS[args.subset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for split_name, source_name in (("train", "train"), ("valid", "validation"), ("test", "test")):
        output_path = args.output_dir / f"targets_{args.subset}_{split_name}.pt"
        if output_path.is_file():
            print(json.dumps({"phase": "target_cache_exists", "path": str(output_path)}), flush=True)
            continue
        values = pack_split(dataset[source_name], target_index)
        atomic_save(values, output_path)
        print(json.dumps({
            "phase": "target_cache_packed",
            "subset": args.subset,
            "split": split_name,
            "path": str(output_path),
            "shape": list(values.shape),
            "size_gib": round(output_path.stat().st_size / 2**30, 3),
        }), flush=True)
    print(json.dumps({"seconds": round(time.perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
