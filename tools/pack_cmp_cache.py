#!/usr/bin/env python3
"""Pack CMP per-example h-bin files into one contiguous tensor file.

The fine-tuning cache is intentionally written as individual files so that
precomputation can resume.  Training, however, is much faster when the same
cache is read once as a large sequential file.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def load_one(path: Path) -> tuple[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return path.name, payload["h_bins"].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", default="h_bins_packed.pt")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output = cache_dir / args.output
    paths = sorted(cache_dir.glob("*.pt"))
    paths = [path for path in paths if path.name != output.name]
    if not paths:
        raise SystemExit(f"no per-example .pt files found in {cache_dir}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(load_one, paths))
    names = [name for name, _ in rows]
    h_bins = torch.stack([tensor for _, tensor in rows]).contiguous()
    index = {name: row for row, name in enumerate(names)}
    payload = {"version": 1, "names": names, "index": index, "h_bins": h_bins}

    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=cache_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output),
        "examples": len(names),
        "shape": list(h_bins.shape),
        "dtype": str(h_bins.dtype),
        "size_gib": round(output.stat().st_size / 2**30, 3),
        "seconds": round(time.perf_counter() - started, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
