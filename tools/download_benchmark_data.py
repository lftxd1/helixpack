#!/usr/bin/env python
"""Download and verify JanusDNA NT and GenomicBenchmarks data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vericomp.benchmark_data import GB_SPECS, NT_SPECS


def download_nt(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="InstaDeepAI/nucleotide_transformer_downstream_tasks",
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=["*/train.parquet", "*/test.parquet", "*/train.fna", "*/test.fna", "README.md", ".gitattributes"],
        resume_download=True,
    )
    missing = [str(root / name / file)
               for name in NT_SPECS
               for file in ("train.parquet", "test.parquet")
               if not (root / name / file).is_file()]
    if missing:
        raise RuntimeError("NT download missing files:\n" + "\n".join(missing))


def download_gb(root: Path):
    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.data_check import is_downloaded

    root.mkdir(parents=True, exist_ok=True)
    for name in GB_SPECS:
        if not is_downloaded(name, cache_path=str(root)):
            print(f"Downloading GB dataset {name} -> {root}", flush=True)
            download_dataset(name, version=0, dest_path=str(root))
        else:
            print(f"GB dataset already present: {name}", flush=True)
        for split in ("train", "test"):
            split_root = root / name / split
            if not split_root.is_dir() or not any(split_root.rglob("*.txt")):
                raise RuntimeError(f"GB dataset appears incomplete: {split_root}")


def report(name: str, root: Path):
    files = [path for path in root.rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    print(f"{name}: files={len(files)} bytes={size} GiB={size / 2**30:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/benchmark")
    parser.add_argument("--tasks", nargs="+", choices=("nt", "gb"), default=("nt", "gb"))
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    if "nt" in args.tasks:
        print(f"Downloading NT data to {root / 'nt'}", flush=True)
        download_nt(root / "nt")
        report("NT", root / "nt")
    if "gb" in args.tasks:
        download_gb(root / "gb")
        report("GB", root / "gb")
    print("JanusDNA NT/GB benchmark data download and verification completed.", flush=True)


if __name__ == "__main__":
    main()
