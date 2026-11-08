"""DNALongBench TISP data loaders without a dependency on GeneZip code."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, get_worker_info

from .data import IndexedFasta


TISP_SEQUENCE_LENGTH = 100_000
TISP_WINDOW_STEP = 50_000
TRACK_FILES = (
    "agg.plus.bw.bedgraph.bw", "agg.encodecage.plus.v2.bedgraph.bw",
    "agg.encoderampage.plus.v2.bedgraph.bw", "agg.plus.grocap.bedgraph.sorted.merged.bw",
    "agg.plus.allprocap.bedgraph.sorted.merged.bw", "agg.minus.allprocap.bedgraph.sorted.merged.bw",
    "agg.minus.grocap.bedgraph.sorted.merged.bw", "agg.encoderampage.minus.v2.bedgraph.bw",
    "agg.encodecage.minus.v2.bedgraph.bw", "agg.minus.bw.bedgraph.bw",
)
BLACKLIST_FILES = ("blacklists/fantom.blacklist8.plus.bed.gz", "blacklists/fantom.blacklist8.minus.bed.gz")
_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
_RC = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


class SimpleIntervals:
    def __init__(self, paths):
        self.values = {}
        for path in paths:
            opener = gzip.open if str(path).endswith(".gz") else open
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
                        self.values.setdefault(fields[0], []).append((start, end))
        for chrom in self.values:
            self.values[chrom].sort()

    def overlaps(self, chrom, start, end):
        candidates = self.values.get(chrom, self.values.get(chrom.removeprefix("chr"), []))
        if not candidates and not chrom.startswith("chr"):
            candidates = self.values.get("chr" + chrom, [])
        return [(max(start, s), min(end, e)) for s, e in candidates if s < end and e > start]


class TISPTracks:
    def __init__(self, root: str):
        import pyBigWig

        self._target_root = Path(root) / "tisp" / "targets"
        self._pid = os.getpid()
        self.handles = [pyBigWig.open(str(self._target_root / name)) for name in TRACK_FILES]
        # The DNALongBench/GeneZip preprocessing has one blacklist per strand.
        # Only the corresponding FANTOM CAGE track is repaired there, using
        # the ENCODE CAGE track as a scaled replacement.  Keep the two interval
        # sets separate so that the replacement cannot affect other assays.
        self.blacklists = [
            SimpleIntervals([self._target_root / name]) for name in BLACKLIST_FILES
        ]
        self.chrom_sizes = {str(k): int(v) for k, v in self.handles[0].chroms().items()}

    def _ensure_process_handles(self) -> None:
        """Reopen BigWig handles after a DataLoader fork.

        libBigWig handles are not fork-safe.  The dataset remains read-only,
        so each worker can cheaply own its own handles while sharing all
        window metadata with the parent process.
        """
        pid = os.getpid()
        if pid == self._pid:
            return
        import pyBigWig
        for handle in self.handles:
            try:
                handle.close()
            except Exception:
                pass
        self.handles = [pyBigWig.open(str(self._target_root / name)) for name in TRACK_FILES]
        self._pid = pid

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self.chrom_sizes:
            return chrom
        alt = chrom.removeprefix("chr") if chrom.startswith("chr") else "chr" + chrom
        if alt in self.chrom_sizes:
            return alt
        raise KeyError(chrom)

    def get(self, chrom: str, start: int, end: int) -> np.ndarray:
        self._ensure_process_handles()
        chrom = self.resolve_chrom(chrom)
        out = np.zeros((len(self.handles), end - start), dtype=np.float32)
        for index, handle in enumerate(self.handles):
            values = handle.values(chrom, start, end, numpy=True)
            if values is not None:
                values = np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)
                out[index, : len(values)] = values
        # Match GeneZip's replacement rule exactly: repair FANTOM CAGE (0, 9)
        # only inside its strand-specific blacklist intervals, using ENCODE
        # CAGE (1, 8) as the replacement track.  Do not overwrite ENCODE CAGE
        # globally and do not zero all ten assays in the blacklist intervals.
        for blacklist, target_index, replacement_index in zip(
            self.blacklists, (0, 9), (1, 8)
        ):
            for left, right in blacklist.overlaps(chrom, start, end):
                lo = max(0, left - start)
                hi = min(end - start, right - start)
                if hi > lo:
                    out[target_index, lo:hi] = (
                        out[replacement_index, lo:hi] * 0.61357
                    )
        return out.T


class TISPRandomDataset(IterableDataset):
    def __init__(self, root, num_samples: int, seed: int = 3):
        self.root = str(root)
        requested_samples = int(num_samples)
        self.seed = int(seed)
        self.fasta = IndexedFasta(Path(root) / "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
        self.tracks = TISPTracks(root)
        all_chroms = [c for c in self.fasta.index if c in self.tracks.chrom_sizes or "chr" + c in self.tracks.chrom_sizes]
        # Match GeneZip: chr8/chr9 are test holdouts and chr10 is validation.
        self.chroms = [c for c in all_chroms if c.removeprefix("chr") not in {"8", "9", "10"}]
        if not self.chroms:
            raise ValueError("No eligible TISP training chromosomes found.")
        if requested_samples > 0:
            self.num_samples = requested_samples
        else:
            self.num_samples = sum(
                max(0, (self.fasta.index[c][0] - TISP_SEQUENCE_LENGTH) // TISP_WINDOW_STEP + 1)
                for c in self.chroms
            )
        self._iterator_index = 0
        self.weights = np.asarray([self.fasta.index[c][0] - 2 * TISP_SEQUENCE_LENGTH for c in self.chroms], dtype=np.float64)
        self.weights = np.maximum(self.weights, 1.0); self.weights /= self.weights.sum()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        info = get_worker_info(); worker_id = info.id if info else 0; workers = info.num_workers if info else 1
        count = self.num_samples // workers + (worker_id < self.num_samples % workers)
        # A new iterator is a new random-window epoch.  Do not replay the
        # exact same stream when the training loop wraps the IterableDataset.
        iterator_index = self._iterator_index
        self._iterator_index += 1
        rng = np.random.default_rng(self.seed + worker_id + 1_000_003 * iterator_index)
        produced = 0
        while produced < count:
            chrom = str(rng.choice(self.chroms, p=self.weights))
            chrom_len = self.fasta.index[chrom][0]
            center = int(rng.integers(TISP_SEQUENCE_LENGTH, chrom_len - TISP_SEQUENCE_LENGTH))
            start, end = center - TISP_SEQUENCE_LENGTH // 2, center + TISP_SEQUENCE_LENGTH // 2
            sequence = self.fasta.fetch(chrom, start, end)
            if len(sequence) != TISP_SEQUENCE_LENGTH or sequence.count("N") / TISP_SEQUENCE_LENGTH > 0.30:
                continue
            ids = torch.tensor([_BASE_TO_ID.get(base, 5) for base in sequence], dtype=torch.long)
            target = torch.from_numpy(self.tracks.get(chrom, start, end))
            produced += 1
            yield {"input_ids": ids, "target": target}


class TISPWindowDataset(Dataset):
    def __init__(self, root, chroms, max_samples: int | None = None):
        self.root = str(root)
        self.fasta = IndexedFasta(Path(root) / "tisp/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
        self.tracks = TISPTracks(root)
        self.windows = []
        for chrom in chroms:
            if chrom not in self.fasta.index:
                continue
            length = self.fasta.index[chrom][0]
            if length < TISP_SEQUENCE_LENGTH:
                continue
            starts = list(range(0, length - TISP_SEQUENCE_LENGTH + 1, TISP_WINDOW_STEP))
            if not starts or starts[-1] != length - TISP_SEQUENCE_LENGTH:
                starts.append(length - TISP_SEQUENCE_LENGTH)
            self.windows.extend((chrom, start) for start in starts)
        if max_samples and max_samples > 0:
            self.windows = self.windows[: int(max_samples)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        chrom, start = self.windows[int(index)]
        sequence = self.fasta.fetch(chrom, start, start + TISP_SEQUENCE_LENGTH)
        ids = torch.tensor([_BASE_TO_ID.get(base, 5) for base in sequence], dtype=torch.long)
        target = torch.from_numpy(self.tracks.get(chrom, start, start + TISP_SEQUENCE_LENGTH))
        return {"input_ids": ids, "target": target}


def build_tisp_datasets(root: str, num_train_samples: int, max_valid_samples=None, max_test_samples=None):
    train = TISPRandomDataset(root, num_train_samples)
    valid = TISPWindowDataset(root, ["chr10", "10"], max_valid_samples)
    test = TISPWindowDataset(root, ["chr8", "8", "chr9", "9"], max_test_samples)
    return train, valid, test
