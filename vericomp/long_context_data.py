"""Hybrid BED/genome sampling for long-context MLM pre-training.

The existing :class:`vericomp.data.BEDGenomeDataset` deliberately keeps the
original short-context recipe unchanged.  This module adds a separate dataset
for the long-context recipe:

* targets up to 128 KiB are delegated to the existing BED-window sampler;
* longer targets are sampled from the indexed reference genome;
* a sampled long window must overlap a split-specific BED anchor, which keeps
  train/validation/test regions disjoint even though the BED file mixes
  chromosomes between splits;
* windows are always completely inside a chromosome and are rejected when the
  fraction of ambiguous ``N`` bases is too high.

The dataset exposes ``sample_lengths`` so the existing length-aware DDP batch
sampler can keep every global batch at one exact length.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from vericomp.data import BASE_TO_ID, COMPLEMENT, BEDGenomeDataset, IndexedFasta


class HybridLongContextDataset(Dataset):
    """Mix existing BED samples with random, split-safe genomic long windows.

    ``long_samples_per_length`` controls how many placeholder samples are put
    in each long-length bucket.  Long samples are intentionally generated with
    replacement: the actual genomic window is selected in ``__getitem__`` and
    therefore changes whenever the sampler revisits an index.
    """

    SHORT_CONTEXT_LIMIT = 131_072

    def __init__(
        self,
        bed_path: str | Path,
        fasta_path: str | Path,
        *,
        sequence_lengths: List[int],
        split: str,
        max_length: int,
        seed: int = 17,
        random_window_sampling: bool = True,
        long_samples_per_length: int | None = None,
        max_samples_per_length: int | None = None,
        n_max_fraction: float = 0.01,
        max_window_attempts: int = 32,
        boundary_margin: int = 0,
    ):
        lengths = [int(length) for length in sequence_lengths]
        if not lengths or any(length <= 0 or length > max_length for length in lengths):
            raise ValueError("sequence_lengths must be in (0, max_length]")
        if split not in {"train", "valid", "test", None}:
            raise ValueError(f"unsupported split {split!r}")
        if not 0.0 <= float(n_max_fraction) <= 1.0:
            raise ValueError("n_max_fraction must be between 0 and 1")
        if max_window_attempts <= 0:
            raise ValueError("max_window_attempts must be positive")
        if boundary_margin < 0:
            raise ValueError("boundary_margin cannot be negative")

        self.sequence_lengths = lengths
        self.split = split
        self.seed = int(seed)
        self.random_window_sampling = bool(random_window_sampling)
        self.n_max_fraction = float(n_max_fraction)
        self.max_window_attempts = int(max_window_attempts)
        self.boundary_margin = int(boundary_margin)
        self.reader = IndexedFasta(fasta_path)

        self.short_lengths = [length for length in lengths
                              if length <= self.SHORT_CONTEXT_LIMIT]
        self.long_lengths = [length for length in lengths
                             if length > self.SHORT_CONTEXT_LIMIT]

        self.short_dataset = None
        if self.short_lengths:
            self.short_dataset = BEDGenomeDataset(
                bed_path,
                fasta_path,
                max_length=self.SHORT_CONTEXT_LIMIT,
                sequence_lengths=self.short_lengths,
                split=split,
                seed=self.seed,
                random_window_sampling=self.random_window_sampling,
            )

        self.long_records = self._read_anchor_records(bed_path)
        self._eligible_anchors = {
            length: self._anchors_for_length(length) for length in self.long_lengths
        }
        for length, anchors in self._eligible_anchors.items():
            if not anchors:
                raise ValueError(
                    f"no chromosome-safe {length} bp windows are available for split={split!r}; "
                    "check the FASTA index and BED anchors"
                )

        short_samples = []
        if self.short_dataset is not None:
            short_samples = [
                ("short", index, length)
                for index, length in enumerate(self.short_dataset.sample_lengths)
            ]
            if max_samples_per_length is not None:
                short_samples = self._limit_per_length(
                    short_samples, int(max_samples_per_length)
                )

        if long_samples_per_length is None:
            if self.short_dataset is not None:
                counts = {
                    length: sum(1 for value in self.short_dataset.sample_lengths
                                if value == length)
                    for length in self.short_lengths
                }
                fallback = max(counts.values(), default=1024)
            else:
                fallback = 1024
            long_samples_per_length = fallback
        if long_samples_per_length <= 0:
            raise ValueError("long_samples_per_length must be positive")

        long_count = int(long_samples_per_length)
        if max_samples_per_length is not None:
            long_count = min(long_count, int(max_samples_per_length))
        long_samples = [
            ("long", None, length)
            for length in self.long_lengths
            for _ in range(long_count)
        ]
        self.samples = short_samples + long_samples
        if not self.samples:
            raise ValueError("the hybrid dataset contains no samples")

    @staticmethod
    def _limit_per_length(samples, limit):
        if limit <= 0:
            raise ValueError("max_samples_per_length must be positive")
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample[2], []).append(sample)
        result = []
        for length in sorted(grouped):
            result.extend(grouped[length][:limit])
        return result

    def _read_anchor_records(self, bed_path):
        records = []
        with Path(bed_path).open() as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                record_split = fields[3] if len(fields) >= 4 else None
                if self.split is not None and record_split != self.split:
                    continue
                chrom = fields[0]
                if chrom not in self.reader.index:
                    continue
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError:
                    continue
                chrom_length = self.reader.index[chrom][0]
                if 0 <= start < end <= chrom_length:
                    records.append((chrom, start, end))
        return records

    def _anchors_for_length(self, length):
        anchors = []
        for chrom, start, end in self.long_records:
            chrom_length = self.reader.index[chrom][0]
            usable_length = chrom_length - 2 * self.boundary_margin
            if usable_length < length:
                continue
            # A start in this interval makes the long window overlap the full
            # BED anchor.  The window is then still generated from the genome,
            # rather than copied from the original BED sequence.
            low = max(self.boundary_margin, end - length)
            high = min(start, chrom_length - self.boundary_margin - length)
            if low <= high:
                anchors.append((chrom, start, end, low, high))
        return anchors

    def __len__(self):
        return len(self.samples)

    def _fetch_ids(self, chrom, start, length):
        sequence = self.reader.fetch(chrom, start, start + length)
        if len(sequence) != length:
            raise RuntimeError(
                f"FASTA returned {len(sequence)} bases for requested {length} bp "
                f"window {chrom}:{start}-{start + length}"
            )
        return torch.tensor(
            [BASE_TO_ID.get(base, 5) for base in sequence], dtype=torch.long
        )

    def _random_long_item(self, length, index):
        anchors = self._eligible_anchors[length]
        # Each worker gets a different Python random seed from DataLoader.  A
        # per-call RNG also prevents a repeated index from reproducing a fixed
        # long window when the length sampler cycles a bucket.
        rng = random.Random()
        if self.random_window_sampling:
            rng.seed((self.seed << 32) ^ (int(index) << 8) ^ random.getrandbits(64))
        else:
            # Validation windows are fixed across evaluations, while training
            # windows remain fresh when a bucket is revisited.
            rng.seed((self.seed << 32) ^ (int(index) << 8))
        last_fraction = 1.0
        for _ in range(self.max_window_attempts):
            chrom, _, _, low, high = rng.choice(anchors)
            start = rng.randint(low, high)
            pos = self._fetch_ids(chrom, start, length)
            last_fraction = float(pos.eq(5).float().mean().item())
            if last_fraction <= self.n_max_fraction:
                rc = COMPLEMENT[pos.flip(0)]
                return {"input_ids_pos": pos, "input_ids_rc": rc}
        raise RuntimeError(
            f"could not sample a {length} bp window with N fraction <= "
            f"{self.n_max_fraction} after {self.max_window_attempts} attempts "
            f"(last fraction={last_fraction:.4f}, split={self.split!r})"
        )

    def __getitem__(self, index):
        kind, short_index, length = self.samples[int(index)]
        if kind == "short":
            return self.short_dataset[short_index]
        return self._random_long_item(length, index)

    @property
    def sample_lengths(self):
        return [sample[2] for sample in self.samples]
