#!/usr/bin/env python3
"""Collect the best test AUROC from eQTL training logs.

The eQTL trainer writes periodic records such as::

    {"batch_step": 100, "optimizer_step": 100,
     "test": {"auroc": 0.75, ...}}

This script scans one-line JSON records in the logs, also accepting the final
test record, and reports the highest test AUROC for each task.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/logs/finetune/eQTL") / date.today().strftime("%Y%m%d"),
        help="Directory containing eQTL logs (default: today's dated eQTL directory)",
    )
    parser.add_argument(
        "--pattern",
        default="eqtl-residual-skip-30epoch-full*.log",
        help="Log filename glob (default: %(default)s)",
    )
    parser.add_argument(
        "--default-cell-type",
        default="Artery_Tibial",
        help="Cell type used for a log without a cell-type suffix",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Also write the result as JSON to this path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable table",
    )
    return parser.parse_args()


def task_name(log_path: Path, default_cell_type: str) -> str:
    """Infer a cell type from the standard log filename."""
    name = log_path.name
    prefix = "eqtl-residual-skip-30epoch-full"
    if name == f"{prefix}.log":
        return default_cell_type
    suffix = name.removeprefix(prefix).removesuffix(".log")
    return suffix.lstrip("-") or default_cell_type


def json_objects_from_log(log_path: Path) -> list[dict[str, Any]]:
    """Read one-line JSON objects, ignoring ordinary progress lines."""
    objects: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text.startswith("{") or not text.endswith("}"):
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects.append(value)
    return objects


def test_records(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in objects:
        metrics = obj.get("test")
        if not isinstance(metrics, dict) or "auroc" not in metrics:
            continue
        try:
            auroc = float(metrics["auroc"])
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "auroc": auroc,
                "auprc": metrics.get("auprc"),
                "loss": metrics.get("loss"),
                "examples": metrics.get("examples"),
                "batch_step": obj.get("batch_step"),
                "optimizer_step": obj.get("optimizer_step"),
            }
        )
    return records


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    paths = sorted(args.log_dir.glob(args.pattern))
    for log_path in paths:
        records = test_records(json_objects_from_log(log_path))
        best = max(records, key=lambda record: record["auroc"], default=None)
        results.append(
            {
                "cell_type": task_name(log_path, args.default_cell_type),
                "best_test_auroc": None if best is None else best["auroc"],
                "best_test_auprc": None if best is None else best["auprc"],
                "best_test_loss": None if best is None else best["loss"],
                "batch_step": None if best is None else best["batch_step"],
                "optimizer_step": None if best is None else best["optimizer_step"],
                "test_records": len(records),
                "log": str(log_path),
            }
        )
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ["cell_type", "best_test_auroc", "batch_step", "test_records", "status"]
    rows = []
    for result in results:
        rows.append(
            [
                result["cell_type"],
                "-" if result["best_test_auroc"] is None else f'{result["best_test_auroc"]:.6f}',
                "-" if result["batch_step"] is None else str(result["batch_step"]),
                str(result["test_records"]),
                "no test yet" if result["best_test_auroc"] is None else "ok",
            ]
        )
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    results = collect(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
