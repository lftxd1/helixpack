"""DNALongBench ETGP records and VeriComp tokenization."""

from __future__ import annotations

import csv
import os

import torch
from torch.utils.data import Dataset

from .eqtl_data import IntervalIndex, PaddedFasta, _mask_sequence, parse_config, reverse_complement


def parse_etgp_records(
    *,
    root_path: str,
    config: dict[str, str | int | float | bool],
    subset: str,
    max_records: int | None = None,
) -> list[tuple[str, int]]:
    """Build ETGP examples using the public GeneZip/DNALongBench convention."""
    fasta = PaddedFasta(os.path.join(root_path, str(config["genome_fa"])))
    blacklist = IntervalIndex(os.path.join(root_path, str(config["enhancer_tabix_file"])))
    records: list[tuple[str, int]] = []
    with open(os.path.join(root_path, str(config["EPI_file"])), "rt", encoding="utf-8") as handle:
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
            region_start_raw, region_end_raw = region_start, region_end
            region_start -= int(config["region_flank_upstream"])
            region_end += int(config["region_flank_downstream"])
            distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
            if distance > int(config.get("seq_len_cutoff", 450000)):
                continue

            sequence_start = min(tss_start, region_start)
            sequence_end = max(tss_end, region_end)
            chrom = str(row["region_chrom"])
            sequence = fasta.fetch(chrom, sequence_start, sequence_end)
            if distance > 0:
                if tss_start > region_end:
                    gap_start, gap_end = region_end, tss_start
                else:
                    gap_start, gap_end = tss_end, region_start
                sequence = _mask_sequence(
                    sequence,
                    blacklist.overlaps(chrom, gap_start, gap_end),
                    sequence_start,
                )
            if gene_start > region_end_raw:
                sequence = reverse_complement(sequence)

            max_length = int(config.get("seq_len_cutoff", 450000))
            sequence = sequence[:max_length].ljust(max_length, "N")
            label = 1 if str(row.get("target", "")).lower() == "positive" else 0
            records.append((sequence, label))
            if max_records is not None and len(records) >= int(max_records):
                break
    return records


class ETGPDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        cell_type: str,
        subset: str,
        max_length: int,
        max_records: int | None = None,
    ) -> None:
        task_root = os.path.join(os.path.abspath(data_root), "etgp")
        config_path = os.path.join(task_root, "config", f"{cell_type}.config")
        config = parse_config(config_path)
        self.max_length = int(max_length)
        self.records = parse_etgp_records(
            root_path=task_root,
            config=config,
            subset=subset,
            max_records=max_records,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence, label = self.records[index]
        sequence = sequence[: self.max_length].ljust(self.max_length, "N")
        ids = torch.tensor(
            [{"A": 0, "C": 1, "G": 2, "T": 3}.get(base, 5) for base in sequence],
            dtype=torch.long,
        )
        return {"input_ids": ids, "labels": torch.tensor(label, dtype=torch.long)}
