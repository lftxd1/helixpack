#!/usr/bin/env python
"""Download and verify DNALongBench data used by VeriComp downstream tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "andyjzhao/dnalongbench"
TASK_PATTERNS = {
    "cmp": ["cmp/**"],
    "etgp": ["etgp/**"],
    "tisp": ["tisp/**"],
    "eqtl": ["eqtl/**"],
}
REQUIRED = {
    "cmp": ["cmp/dataset_dict.json", "cmp/train/dataset_info.json",
            "cmp/validation/dataset_info.json", "cmp/test/dataset_info.json"],
    "etgp": ["etgp/seqs/hg19.fa", "etgp/seqs/hg19.fa.fai",
             "etgp/config/CRISPRi_EPI_K562_hg19.config",
             "etgp/targets/CRISPRi_EPI_K562_hg19.data.tsv"],
    "tisp": ["tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
             "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai",
             "tisp/targets/agg.plus.bw.bedgraph.bw",
             "tisp/targets/agg.minus.bw.bedgraph.bw"],
    "eqtl": ["eqtl/seqs/hg38.fa", "eqtl/seqs/hg38.fa.fai",
             "eqtl/config/gtex_hg38.Whole_Blood.config"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dnalongbench")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_PATTERNS),
                        default=["etgp", "cmp", "tisp"])
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    patterns = [pattern for task in args.tasks for pattern in TASK_PATTERNS[task]]
    print(f"Downloading {args.repo_id}: {', '.join(args.tasks)} -> {root}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
        resume_download=True,
    )
    missing = [item for task in args.tasks for item in REQUIRED[task]
               if not (root / item).is_file()]
    if missing:
        raise SystemExit("Download finished but required files are missing:\n" + "\n".join(missing))
    for task in args.tasks:
        task_root = root / task
        files = sum(1 for path in task_root.rglob("*") if path.is_file())
        size = sum(path.stat().st_size for path in task_root.rglob("*") if path.is_file())
        print(f"{task}: files={files} bytes={size} GiB={size / 2**30:.2f}", flush=True)
    print("DNALongBench download and required-file verification completed.", flush=True)


if __name__ == "__main__":
    main()
