"""Minimal DNALongBench eQTL loader compatible with VeriComp token IDs.

The record construction follows GeneZip's ``eqtl.sh`` pipeline: construct a
sequence spanning the gene TSS and variant region, mask blacklist intervals,
replace allele1 with allele2, orient relative to the gene, and pad/crop to a
fixed downstream length.
"""

from __future__ import annotations

from bisect import bisect_left
import csv
import gzip
import os
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .data import IndexedFasta


_RC = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_RC)[::-1]


def parse_config(path: str) -> dict[str, str | int | float | bool]:
    config: dict[str, str | int | float | bool] = {}
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 3 or fields[0].startswith("#"):
                continue
            key, raw, kind = fields[:3]
            if kind == "int":
                value: str | int | float | bool = int(raw)
            elif kind == "float":
                value = float(raw)
            elif kind == "bool":
                value = raw.lower() in {"1", "true", "yes"}
            else:
                value = raw
            config[key] = value
    return config


class IntervalIndex:
    def __init__(self, path: str):
        intervals: dict[str, list[tuple[int, int]]] = {}
        opener = gzip.open if path.endswith((".gz", ".bgz")) else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError:
                    continue
                if end > start:
                    intervals.setdefault(fields[0], []).append((start, end))

        self.intervals: dict[str, list[tuple[int, int]]] = {}
        self.starts: dict[str, list[int]] = {}
        for chrom, values in intervals.items():
            values.sort()
            merged: list[tuple[int, int]] = []
            for start, end in values:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self.intervals[chrom] = merged
            self.starts[chrom] = [start for start, _ in merged]

    def _chrom(self, chrom: str) -> str | None:
        if chrom in self.intervals:
            return chrom
        alternate = chrom.removeprefix("chr") if chrom.startswith("chr") else f"chr{chrom}"
        return alternate if alternate in self.intervals else None

    def overlaps(self, chrom: str, start: int, end: int) -> Iterable[tuple[int, int]]:
        chrom = self._chrom(chrom)
        if chrom is None or end <= start:
            return
        values = self.intervals[chrom]
        starts = self.starts[chrom]
        index = max(0, bisect_left(starts, start) - 1)
        while index < len(values):
            item_start, item_end = values[index]
            if item_start >= end:
                break
            if item_end > start:
                yield max(start, item_start), min(end, item_end)
            index += 1


class PaddedFasta:
    def __init__(self, path: str):
        self.fasta = IndexedFasta(path)

    def _chrom(self, chrom: str) -> str:
        if chrom in self.fasta.index:
            return chrom
        alternate = chrom.removeprefix("chr") if chrom.startswith("chr") else f"chr{chrom}"
        if alternate in self.fasta.index:
            return alternate
        raise KeyError(f"chromosome {chrom!r} is absent from {self.fasta.path}")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        chrom = self._chrom(chrom)
        chrom_size = self.fasta.index[chrom][0]
        if end <= start:
            return ""
        left_pad = "N" * max(0, -start)
        right_pad = "N" * max(0, end - chrom_size)
        return left_pad + self.fasta.fetch(chrom, max(0, start), min(end, chrom_size)) + right_pad


def _mask_sequence(sequence: str, overlaps: Iterable[tuple[int, int]], sequence_start: int) -> str:
    data = bytearray(sequence.encode("ascii"))
    for start, end in overlaps:
        left = max(0, start - sequence_start)
        right = min(len(data), end - sequence_start)
        if right > left:
            data[left:right] = b"N" * (right - left)
    return data.decode("ascii")


def parse_eqtl_records(
    *, root_path: str, config: dict[str, str | int | float | bool], subset: str,
    max_records: int | None = None,
) -> list[tuple[str, int]]:
    fasta = PaddedFasta(os.path.join(root_path, str(config["genome_fa"])))
    blacklist = IntervalIndex(os.path.join(root_path, str(config["eQTL_tabix_file"])))
    records: list[tuple[str, int]] = []
    with open(os.path.join(root_path, str(config["eQTL_file"])), "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("subset") != subset or row.get("gene_chrom") != row.get("region_chrom"):
                continue
            try:
                gene_start, gene_end = int(row["gene_start"]), int(row["gene_end"])
                region_start, region_end = int(row["region_start"]), int(row["region_end"])
            except (KeyError, TypeError, ValueError):
                continue
            strand = row.get("gene_strand")
            if strand == "+":
                tss_anchor = gene_start
            elif strand == "-":
                tss_anchor = gene_end - 1
            else:
                continue

            tss_start = tss_anchor - int(config["tss_flank_upstream"])
            tss_end = tss_anchor + 1 + int(config["tss_flank_downstream"])
            variant_start, variant_end = region_start, region_end
            region_start -= int(config["region_flank_upstream"])
            region_end += int(config["region_flank_downstream"])
            distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
            if distance > int(config.get("seq_len_cutoff", 450000)):
                continue

            sequence_start = min(tss_start, region_start)
            sequence_end = max(tss_end, region_end)
            chrom = str(row["region_chrom"])
            reference = fasta.fetch(chrom, sequence_start, sequence_end)
            allele1, allele2 = str(row.get("allele1", "")), str(row.get("allele2", ""))
            rel_start, rel_end = variant_start - sequence_start, variant_end - sequence_start
            if allele1 and reference[rel_start:rel_end].upper() != allele1.upper():
                continue

            if distance > 0:
                if tss_start > region_end:
                    gap_start, gap_end = region_end, tss_start
                else:
                    gap_start, gap_end = tss_end, region_start
                reference = _mask_sequence(reference, blacklist.overlaps(chrom, gap_start, gap_end), sequence_start)

            variant = f"{reference[:rel_start]}{allele2}{reference[rel_end:]}"
            if gene_start > region_end:
                variant = reverse_complement(variant)
            label = 1 if str(row.get("target", "")).lower() == "positive" else 0
            records.append((variant, label))
            if max_records is not None and len(records) >= max_records:
                break
    return records


class EQTLDataset(Dataset):
    def __init__(self, data_root: str, cell_type: str, subset: str,
                 max_length: int, max_records: int | None = None,
                 dynamic_padding: bool = False):
        self.dynamic_padding = bool(dynamic_padding)
        task_root = os.path.join(os.path.abspath(data_root), "eqtl")
        config_path = os.path.join(task_root, "config", f"gtex_hg38.{cell_type}.config")
        config = parse_config(config_path)
        self.max_length = int(max_length)
        self.records = parse_eqtl_records(
            root_path=task_root, config=config, subset=subset, max_records=max_records
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence, label = self.records[index]
        sequence = sequence[: self.max_length]
        if not self.dynamic_padding and len(sequence) < self.max_length:
            sequence += "N" * (self.max_length - len(sequence))
        ids = torch.tensor([_BASE_TO_ID.get(base, 5) for base in sequence], dtype=torch.long)
        # Keep the canonical sequence alongside the VeriComp token IDs.  The
        # latter are used by the in-tree model, while external DNA backbones
        # (HyenaDNA, Caduceus, NTv3, JanusDNA, and Evo2) use their own
        # tokenizers.  Returning both avoids silently applying one model's
        # vocabulary to another model.
        return {
            "input_ids": ids,
            "sequence": sequence,
            "labels": torch.tensor(label, dtype=torch.long),
        }
