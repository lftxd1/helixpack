"""FASTA/BED input and 15% span-MLM collation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Sampler


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = torch.tensor([3, 2, 1, 0, 4, 5], dtype=torch.long)


class IndexedFasta:
    def __init__(self, fasta_path: str | Path, fai_path: str | Path | None = None):
        self.path = Path(fasta_path)
        self.fai_path = Path(fai_path or (str(self.path) + ".fai"))
        self.index: Dict[str, tuple[int, int, int, int]] = {}
        with self.fai_path.open() as f:
            for line in f:
                name, length, offset, line_bases, line_width = line.split()[:5]
                self.index[name] = (int(length), int(offset), int(line_bases), int(line_width))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length, offset, line_bases, line_width = self.index[chrom]
        start, end = max(0, start), min(end, length)
        if end <= start:
            return ""
        # FASTA files are wrapped at a fixed line width.  Reading one wrapped
        # line at a time is prohibitively slow for the 128k eQTL windows
        # (roughly 2,500 seeks per record).  Read the contiguous byte span and
        # remove line breaks in memory instead.
        first_line = start // line_bases
        last_line = (end - 1) // line_bases
        file_start = offset + first_line * line_width + (start % line_bases)
        byte_count = (end - start) + (last_line - first_line) * (line_width - line_bases)
        with self.path.open("rb") as f:
            f.seek(file_start)
            data = f.read(byte_count)
        return data.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()[: end - start]


class BEDGenomeDataset(Dataset):
    def __init__(self, bed_path: str | Path, fasta_path: str | Path,
                 max_length: int = 1_048_576,
                 sequence_length: int | None = None,
                 sequence_lengths: List[int] | None = None,
                 split: str | None = None,
                 max_samples: int | None = None,
                 seed: int = 17,
                 random_window_sampling: bool = False):
        self.reader = IndexedFasta(fasta_path)
        if sequence_length is not None and sequence_lengths is not None:
            raise ValueError("set either sequence_length or sequence_lengths, not both")
        self.sequence_lengths = sequence_lengths or ([sequence_length] if sequence_length else None)
        self.split = split
        self.random_window_sampling = random_window_sampling
        self.seed = seed
        if self.sequence_lengths and any(length <= 0 or length > max_length for length in self.sequence_lengths):
            raise ValueError("requested sequence lengths must be in (0, max_length]")
        minimum_length = min(self.sequence_lengths) if self.sequence_lengths else 1
        self.records = []
        with Path(bed_path).open() as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                record_split = fields[3] if len(fields) >= 4 else None
                if split is not None and record_split != split:
                    continue
                start, end = int(fields[1]), int(fields[2])
                if (end > start and end - start <= max_length and end - start >= minimum_length
                        and fields[0] in self.reader.index):
                    self.records.append((fields[0], start, end))

        if self.sequence_lengths:
            # Keep the target length visible to the batch sampler. In random
            # mode, record_index is only a placeholder: __getitem__ chooses a
            # fresh eligible BED interval for every access, with replacement.
            # This is important when BED intervals are exactly max_length and
            # therefore cannot provide an in-interval random crop themselves.
            eligible_by_length = {
                target_length: [
                    record_index
                    for record_index, record in enumerate(self.records)
                    if record[2] - record[1] >= target_length
                ]
                for target_length in self.sequence_lengths
            }
            if random_window_sampling:
                self._eligible_records = eligible_by_length
                self.samples = [
                    (None, target_length)
                    for target_length in self.sequence_lengths
                    for _ in range(len(eligible_by_length[target_length]))
                ]
            else:
                # Expand into deterministic (record, target_length) samples.
                # This lets a batch sampler know the exact L before reading
                # the FASTA.
                self._eligible_records = None
                self.samples = [
                    (record_index, target_length)
                    for target_length in self.sequence_lengths
                    for record_index in eligible_by_length[target_length]
                ]
        else:
            self._eligible_records = None
            self.samples = [(index, None) for index in range(len(self.records))]
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(self.samples), generator=generator)[:max_samples].tolist()
            self.samples = [self.samples[index] for index in order]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_index, target_length = self.samples[index]
        if self.random_window_sampling:
            if target_length is None or self._eligible_records is None:
                raise RuntimeError(
                    "random_window_sampling requires sequence_lengths"
                )
            candidates = self._eligible_records[target_length]
            if not candidates:
                raise IndexError(f"no BED interval can provide length {target_length}")
            record_index = random.choice(candidates)
        chrom, start, end = self.records[record_index]
        if target_length is not None:
            max_start = end - target_length
            if max_start > start:
                start = random.randint(start, max_start)
            end = start + target_length
        seq = self.reader.fetch(chrom, start, end)
        pos = torch.tensor([BASE_TO_ID.get(base, 5) for base in seq], dtype=torch.long)
        rc = COMPLEMENT[pos.flip(0)]
        return {"input_ids_pos": pos, "input_ids_rc": rc}

    @property
    def sample_lengths(self) -> List[int]:
        """Known sequence length for every dataset index."""
        lengths = []
        for record_index, target_length in self.samples:
            if target_length is None:
                record = self.records[record_index]
                lengths.append(record[2] - record[1])
            else:
                lengths.append(target_length)
        return lengths


class DistributedLengthBatchSampler(Sampler[List[int]]):
    """DDP batch sampler with identical L on all ranks for every step.

    A *global* batch is formed inside one length bucket, then split into
    ``world_size`` rank-local lists. Every rank iterates the same global batch
    order and receives its own slice. This is intentionally different from
    DistributedSampler, which shards complete batches and can make rank 0 and
    rank 1 use different sequence lengths at the same DDP step.

    ``per_rank_batch_size`` can be an integer or a length -> local batch-size
    mapping. If ``tokens_per_rank`` is given, local batch size is
    ``max(1, tokens_per_rank // L)``. The resulting global batch size is always
    divisible by world_size.
    """

    def __init__(self, dataset: BEDGenomeDataset, *, world_size: int = 1,
                 rank: int = 0, per_rank_batch_size: int | Dict[int, int] = 1,
                 tokens_per_rank: int | None = None, drop_last: bool = True,
                 shuffle: bool = True, seed: int = 17,
                 max_per_rank_batch_size: int | None = None,
                 equal_length_sampling: bool = False):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid DDP world_size/rank")
        if tokens_per_rank is not None and tokens_per_rank <= 0:
            raise ValueError("tokens_per_rank must be positive")
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.per_rank_batch_size = per_rank_batch_size
        self.tokens_per_rank = tokens_per_rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_per_rank_batch_size = max_per_rank_batch_size
        self.equal_length_sampling = equal_length_sampling
        self.epoch = 0
        self._lengths = dataset.sample_lengths
        self._bucket_keys = sorted(set(self._lengths))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batch_size(self, length: int) -> int:
        if self.tokens_per_rank is not None:
            size = max(1, self.tokens_per_rank // length)
            if self.max_per_rank_batch_size is not None:
                size = min(size, self.max_per_rank_batch_size)
            return size
        if isinstance(self.per_rank_batch_size, dict):
            if length not in self.per_rank_batch_size:
                raise KeyError(f"no batch size configured for length {length}")
            size = self.per_rank_batch_size[length]
        else:
            size = self.per_rank_batch_size
        if self.max_per_rank_batch_size is not None:
            size = min(size, self.max_per_rank_batch_size)
        return size

    def _global_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches_by_length = {}
        for length in self._bucket_keys:
            indices = torch.tensor([i for i, value in enumerate(self._lengths) if value == length])
            if self.shuffle and len(indices):
                indices = indices[torch.randperm(len(indices), generator=generator)]
            local_size = self._batch_size(length)
            if local_size <= 0:
                raise ValueError("batch size must be positive")
            global_size = local_size * self.world_size
            usable = (len(indices) // global_size) * global_size
            if not self.drop_last and usable < len(indices):
                # Keep the tail for evaluation. __iter__ distributes this
                # final global batch as evenly as possible across ranks.
                usable = len(indices)
            length_batches = []
            for start in range(0, usable, global_size):
                length_batches.append(
                    (length, indices[start:start + global_size].tolist(), local_size)
                )
            if length_batches:
                batches_by_length[length] = length_batches

        if self.equal_length_sampling:
            if not batches_by_length:
                return []
            # Keep the same number of batches per epoch as ordinary bucket
            # concatenation, but choose lengths uniformly.  If a bucket has
            # fewer batches than its equal-share allocation, cycle through
            # its reshuffled batches; random_window_sampling makes each access
            # a fresh BED-window sample even when an index recurs.
            total_batches = sum(len(batches) for batches in batches_by_length.values())
            lengths = sorted(batches_by_length)
            choices = (lengths * ((total_batches + len(lengths) - 1) // len(lengths)))[:total_batches]
            if self.shuffle and len(choices) > 1:
                order = torch.randperm(len(choices), generator=generator).tolist()
                choices = [choices[i] for i in order]
            positions = {length: 0 for length in lengths}
            equal_batches = []
            for length in choices:
                batches = batches_by_length[length]
                position = positions[length]
                if position >= len(batches):
                    position = 0
                    if self.shuffle and len(batches) > 1:
                        order = torch.randperm(len(batches), generator=generator).tolist()
                        batches = [batches[i] for i in order]
                        batches_by_length[length] = batches
                positions[length] = position + 1
                equal_batches.append(batches[position])
            return equal_batches

        all_batches = []
        for batches in batches_by_length.values():
            all_batches.extend(batches)
        if self.shuffle and len(all_batches) > 1:
            order = torch.randperm(len(all_batches), generator=generator).tolist()
            all_batches = [all_batches[i] for i in order]
        return all_batches

    def __iter__(self):
        for _, global_indices, local_size in self._global_batches():
            nominal_global_size = local_size * self.world_size
            if len(global_indices) == nominal_global_size:
                start = self.rank * local_size
                yield global_indices[start:start + local_size]
                continue
            # Validation can use a final uneven global batch to evaluate an
            # exact sample count (e.g. 300). DDP does not require equal batch
            # dimensions because gradients are disabled there. Training uses
            # drop_last=True and always takes the equal-size branch above.
            base, remainder = divmod(len(global_indices), self.world_size)
            start = self.rank * base + min(self.rank, remainder)
            stop = start + base + (1 if self.rank < remainder else 0)
            if stop > start:
                yield global_indices[start:stop]

    def __len__(self):
        return len(self._global_batches())

    def batch_lengths(self):
        return [length for length, _, _ in self._global_batches()]


class SpanMLMCollator:
    def __init__(self, mask_prob: float = 0.15, mean_span_length: int = 8,
                 mask_token_id: int = 4, pad_token_id: int = 5):
        self.mask_prob = mask_prob
        self.mean_span_length = mean_span_length
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def _mask(self, ids: torch.Tensor):
        out = ids.clone()
        labels = torch.full_like(ids, -100)
        valid = ids.ne(self.pad_token_id)
        target = max(1, int(valid.sum().item() * self.mask_prob)) if valid.any() else 0
        covered = 0
        while covered < target:
            start = random.randrange(ids.numel())
            if not valid[start] or labels[start] != -100:
                continue
            span = max(1, int(random.expovariate(1.0 / self.mean_span_length)))
            end = min(ids.numel(), start + span)
            positions = torch.arange(start, end)[valid[start:end] & labels[start:end].eq(-100)]
            labels[positions] = ids[positions]
            covered += positions.numel()
        if target:
            masked = labels.ne(-100)
            draw = torch.rand_like(ids, dtype=torch.float)
            out[masked & (draw < 0.8)] = self.mask_token_id
            random_ids = torch.randint(0, 4, ids.shape, dtype=ids.dtype)
            out[masked & (draw >= 0.8) & (draw < 0.9)] = random_ids[masked & (draw >= 0.8) & (draw < 0.9)]
        return out, labels

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        max_len = max(x["input_ids_pos"].numel() for x in batch)
        pos = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        rc = torch.full_like(pos, self.pad_token_id)
        labels = torch.full_like(pos, -100)
        for row, item in enumerate(batch):
            n = item["input_ids_pos"].numel()
            masked, row_labels = self._mask(item["input_ids_pos"])
            pos[row, :n], labels[row, :n] = masked, row_labels
            # Keep the two strands exactly paired after MLM corruption.  The
            # reverse-complement strand mirrors every mask and every random
            # replacement applied to the positive strand, so RC(RC(x)) == x
            # still holds for the actual model inputs (with mask token 4 and
            # padding token 5 self-complementary).
            rc[row, :n] = COMPLEMENT[masked.flip(0)]
        return {"input_ids_pos": pos, "input_ids_rc": rc, "labels": labels}
