"""Four-GPU mixed-length MLM pre-training with genomic long-window sampling.

This is intentionally a separate entrypoint from ``pretrain.train``.  The
original short-context recipe remains unchanged.  For this recipe, lengths up
to 128 KiB use the existing BED sampler; longer lengths are sampled from the
indexed genome through :class:`HybridLongContextDataset`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from pretrain.run_config import print_resolved_config, write_run_config
from pretrain.train import (
    AsyncLogWriter,
    assert_same_length,
    build_optimizer,
    build_scheduler,
    evaluate,
    format_duration,
    make_loader,
    masked_accuracy,
    parse_batch_sizes,
    save_checkpoint,
    setup_distributed,
)
from vericomp.data import DistributedLengthBatchSampler, SpanMLMCollator
from vericomp import VeriCompConfig, VeriCompPretrainModel
from vericomp.long_context_data import HybridLongContextDataset


DEFAULT_LENGTHS = (
    "1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576"
)
DEFAULT_BATCH_SIZES = (
    "1024:128,2048:64,4096:32,8192:32,16384:16,32768:16,"
    "65536:8,131072:4,262144:2,524288:2,1048576:2"
)
DEFAULT_LENGTH_WEIGHTS = (
    "1024:1,2048:1,4096:1,8192:1,16384:1,32768:1,65536:1,131072:1,"
    "262144:1.5,524288:2.25,1048576:3"
)


class WeightedLengthBatchSampler(DistributedLengthBatchSampler):
    """Length-aware DDP sampler with weighted bucket selection.

    This class lives in the long-context entrypoint so the short-context
    training sampler keeps its original equal-probability behavior.  Within a
    selected bucket, batches are shuffled exactly as before; only the choice
    of the next length is weighted.
    """

    def __init__(self, *args, length_weights, **kwargs):
        super().__init__(*args, equal_length_sampling=True, **kwargs)
        self.length_weights = {
            int(length): float(weight)
            for length, weight in length_weights.items()
        }
        missing = sorted(set(self._bucket_keys) - set(self.length_weights))
        if missing:
            raise ValueError(f"missing length weights for buckets {missing}")
        if any(self.length_weights[length] <= 0 for length in self._bucket_keys):
            raise ValueError("all length weights must be positive")

    def _global_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batches_by_length = {}
        for length in self._bucket_keys:
            indices = torch.tensor(
                [i for i, value in enumerate(self._lengths) if value == length]
            )
            if self.shuffle and len(indices):
                indices = indices[torch.randperm(len(indices), generator=generator)]
            local_size = self._batch_size(length)
            if local_size <= 0:
                raise ValueError("batch size must be positive")
            global_size = local_size * self.world_size
            usable = (len(indices) // global_size) * global_size
            if not self.drop_last and usable < len(indices):
                usable = len(indices)
            length_batches = []
            for start in range(0, usable, global_size):
                length_batches.append(
                    (length, indices[start:start + global_size].tolist(), local_size)
                )
            if length_batches:
                batches_by_length[length] = length_batches

        if not batches_by_length:
            return []
        available_lengths = sorted(batches_by_length)
        total_batches = sum(len(batches) for batches in batches_by_length.values())
        weights = torch.tensor(
            [self.length_weights[length] for length in available_lengths],
            dtype=torch.float64,
        )
        probabilities = weights / weights.sum()
        selected = torch.multinomial(
            probabilities,
            total_batches,
            replacement=True,
            generator=generator,
        ).tolist()
        positions = {length: 0 for length in available_lengths}
        selected_batches = []
        for bucket_index in selected:
            length = available_lengths[bucket_index]
            batches = batches_by_length[length]
            position = positions[length]
            if position >= len(batches):
                position = 0
                if self.shuffle and len(batches) > 1:
                    order = torch.randperm(len(batches), generator=generator).tolist()
                    batches = [batches[index] for index in order]
                    batches_by_length[length] = batches
            positions[length] = position + 1
            selected_batches.append(batches[position])
        return selected_batches


def parse_length_weights(spec: str, lengths: list[int]) -> dict[int, float]:
    result = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            length_text, weight_text = item.split(":", 1)
            result[int(length_text)] = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid --length-weights item {item!r}; expected length:weight"
            ) from exc
    missing = sorted(set(lengths) - set(result))
    if missing:
        raise ValueError(f"no length weight configured for {missing}")
    if any(result[length] <= 0 for length in lengths):
        raise ValueError("length weights must be positive")
    return {length: result[length] for length in lengths}


def make_weighted_loader(dataset, *, world_size, rank, batch_sizes, seed,
                         shuffle, drop_last, workers, length_weights):
    sampler = WeightedLengthBatchSampler(
        dataset,
        world_size=world_size,
        rank=rank,
        per_rank_batch_size=batch_sizes,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        length_weights=length_weights,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=SpanMLMCollator(),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fasta", default="/gpfs_data/Projects/data/hg38/hg38.ml.fa")
    parser.add_argument("--bed", default="/gpfs_data/Projects/data/hg38/human-sequences.bed")
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--max-length", type=int, default=1_048_576)
    parser.add_argument("--short-context-limit", type=int, default=131_072)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--length-weights",
        default=DEFAULT_LENGTH_WEIGHTS,
        help="length:weight mapping used only by the long-context bucket sampler",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--total-updates", type=int, default=25_000)
    parser.add_argument("--warmup-updates", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--validation-samples", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--long-samples-per-length",
        type=int,
        default=None,
        help="number of generated samples in each long-length bucket; default matches the largest short bucket",
    )
    parser.add_argument(
        "--n-max-fraction",
        type=float,
        default=0.01,
        help="reject a genomic window when this fraction of bases is N/ambiguous",
    )
    parser.add_argument(
        "--max-window-attempts",
        type=int,
        default=32,
        help="retry budget for N-filtered long windows",
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        default=0,
        help="optional extra bases kept away from both chromosome ends",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path("weights/pretrain")
            / time.strftime("%Y%m%d")
            / "vericomp-long-context-1m"
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="load only model weights from a checkpoint; optimizer and scheduler start fresh",
    )
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument(
        "--allocation-mode",
        choices=("dynamic", "uniform", "length_only", "grid"),
        default="dynamic",
        help="quota evidence for the supplementary compression ablation",
    )
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=32,
        help="bases averaged into one bottleneck token when --allocation-mode grid",
    )
    parser.add_argument(
        "--strict-uncompressed-short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use unit quota for lengths <= max_tokens in this long-context recipe",
    )
    parser.add_argument(
        "--duplex-mode",
        choices=("dual", "forward"),
        default="dual",
        help="whether the MLM allocator uses both strands or only the positive strand",
    )
    parser.add_argument(
        "--no-activation-checkpointing",
        action="store_true",
        help="debug only; activation checkpointing is enabled by default for 1M inputs",
    )
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-non4", action="store_true")
    return parser.parse_args()


def _parse_lengths(text):
    lengths = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("--lengths must contain unique positive integers")
    return lengths


def _load_resume(model, optimizer, scheduler, path, rank):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint_state = checkpoint["model"]
    current_state = raw_model.state_dict()
    compatible = {
        name: value
        for name, value in checkpoint_state.items()
        if name in current_state and current_state[name].shape == value.shape
    }
    missing = sorted(set(current_state) - set(compatible))
    unexpected = sorted(set(checkpoint_state) - set(compatible))
    raw_model.load_state_dict(compatible, strict=False)
    if not missing and not unexpected:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        return int(checkpoint.get("update", 0)), int(checkpoint.get("epoch", 0)), "exact"
    if rank == 0:
        print(
            f"resume={path}: warm-started {len(compatible)} matching tensors; "
            f"skipped {len(missing)} missing and {len(unexpected)} unexpected tensors",
            flush=True,
        )
    return 0, 0, "warm_start"


def _load_model_only(model, path, rank):
    """Load only model weights for a fresh optimizer/scheduler branch."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint_state = checkpoint.get("model", checkpoint)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    current_state = raw_model.state_dict()
    compatible = {
        name: value
        for name, value in checkpoint_state.items()
        if name in current_state and current_state[name].shape == value.shape
    }
    missing = sorted(set(current_state) - set(compatible))
    unexpected = sorted(set(checkpoint_state) - set(compatible))
    raw_model.load_state_dict(compatible, strict=False)
    if rank == 0:
        print(
            f"init-from={path}: loaded {len(compatible)} model tensors; "
            f"skipped {len(missing)} missing and {len(unexpected)} unexpected tensors; "
            "optimizer/scheduler reset",
            flush=True,
        )
    return "model_only"


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed()
    log_writer = AsyncLogWriter(rank == 0)
    try:
        if world_size != 4 and not args.allow_non4:
            raise RuntimeError(
                f"long-context recipe requires four GPUs; got WORLD_SIZE={world_size}. "
                "Use --allow-non4 only for a smoke test."
            )
        if device.type != "cuda":
            raise RuntimeError("long-context BF16 training requires CUDA")
        if not torch.backends.cuda.is_flash_attention_available():
            raise RuntimeError("PyTorch Flash Attention backend is unavailable")
        if args.grad_accumulation < 1 or args.total_updates < 1:
            raise ValueError("grad accumulation and total updates must be positive")
        if args.warmup_updates > args.total_updates:
            raise ValueError("warmup updates cannot exceed total updates")
        if args.short_context_limit != HybridLongContextDataset.SHORT_CONTEXT_LIMIT:
            raise ValueError(
                "short-context limit is fixed at 131072 so the existing BED recipe "
                "and the long-context split remain unambiguous"
            )

        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        lengths = _parse_lengths(args.lengths)
        if any(length <= 0 or length > args.max_length for length in lengths):
            raise ValueError("all --lengths must be in (0, --max-length]")
        batch_sizes = parse_batch_sizes(args.batch_sizes)
        if args.batch_size is not None:
            batch_sizes = {length: args.batch_size for length in lengths}
        missing = sorted(set(lengths) - set(batch_sizes))
        if missing:
            raise ValueError(f"missing batch sizes for lengths {missing}")
        length_weights = parse_length_weights(args.length_weights, lengths)

        train_ds = HybridLongContextDataset(
            args.bed,
            args.fasta,
            sequence_lengths=lengths,
            split="train",
            max_length=args.max_length,
            seed=args.seed,
            random_window_sampling=True,
            long_samples_per_length=args.long_samples_per_length,
            n_max_fraction=args.n_max_fraction,
            max_window_attempts=args.max_window_attempts,
            boundary_margin=args.boundary_margin,
        )
        # Keep the requested validation total approximately constant while
        # giving every exact-length bucket a validation representation.
        validation_per_length = max(1, (args.validation_samples + len(lengths) - 1) // len(lengths))
        valid_ds = HybridLongContextDataset(
            args.bed,
            args.fasta,
            sequence_lengths=lengths,
            split="valid",
            max_length=args.max_length,
            seed=args.seed + 10_000,
            random_window_sampling=False,
            long_samples_per_length=validation_per_length,
            max_samples_per_length=validation_per_length,
            n_max_fraction=args.n_max_fraction,
            max_window_attempts=args.max_window_attempts,
            boundary_margin=args.boundary_margin,
        )
        train_loader, train_sampler = make_weighted_loader(
            train_ds,
            world_size=world_size,
            rank=rank,
            batch_sizes=batch_sizes,
            seed=args.seed,
            shuffle=True,
            drop_last=True,
            workers=args.num_workers,
            length_weights=length_weights,
        )
        valid_loader, valid_sampler = make_loader(
            valid_ds,
            world_size=world_size,
            rank=rank,
            batch_sizes=batch_sizes,
            seed=args.seed + 1,
            shuffle=False,
            drop_last=False,
            workers=args.num_workers,
            equal_length_sampling=False,
        )

        model_config = VeriCompConfig(
            num_layers=args.num_layers,
            allocation_mode=args.allocation_mode,
            grid_stride=args.grid_stride,
            strict_uncompressed_short=args.strict_uncompressed_short,
            duplex_mode=args.duplex_mode,
            activation_checkpointing=not args.no_activation_checkpointing,
        )
        model = VeriCompPretrainModel(model_config).to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                gradient_as_bucket_view=True,
                find_unused_parameters=args.allocation_mode != "dynamic",
            )
        optimizer = build_optimizer(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=not args.no_fused_optimizer,
        )
        scheduler = build_scheduler(optimizer, args.warmup_updates, args.total_updates)
        start_update, epoch, resume_mode = 0, 0, "fresh"
        if args.resume and args.init_from:
            raise ValueError("--resume and --init-from are mutually exclusive")
        if args.init_from:
            resume_mode = _load_model_only(model, args.init_from, rank)
        if args.resume:
            start_update, epoch, resume_mode = _load_resume(
                model, optimizer, scheduler, args.resume, rank
            )

        raw_model = model.module if isinstance(model, DDP) else model
        output_dir = Path(args.output_dir)
        if rank == 0:
            metadata = {
                "entrypoint": "pretrain.train_long_context",
                "world_size": world_size,
                "device": torch.cuda.get_device_name(device),
                "parameters": raw_model.num_parameters(),
                "lengths": lengths,
                "short_lengths": train_ds.short_lengths,
                "long_lengths": train_ds.long_lengths,
                "per_rank_batch_sizes": batch_sizes,
                "global_batch_sizes": {
                    length: batch * world_size for length, batch in batch_sizes.items()
                },
                "equal_length_sampling": train_sampler.equal_length_sampling,
                "weighted_length_sampling": True,
                "length_weights": length_weights,
                "length_sampling_probabilities": {
                    str(length): length_weights[length] / sum(length_weights.values())
                    for length in lengths
                },
                "same_length_global_batch": True,
                "short_sampling": "existing BED windows",
                "long_sampling": "random genome windows overlapping split-specific BED anchors",
                "chromosome_boundary_protection": True,
                "boundary_margin": args.boundary_margin,
                "n_max_fraction": args.n_max_fraction,
                "long_samples_per_length": args.long_samples_per_length,
                "train_samples": len(train_ds),
                "validation_samples": len(valid_ds),
                "activation_checkpointing": model_config.activation_checkpointing,
                "allocation_mode": model_config.allocation_mode,
                "strict_uncompressed_short": model_config.strict_uncompressed_short,
                "duplex_mode": model_config.duplex_mode,
                "resume": args.resume,
                "init_from": args.init_from,
                "resume_mode": resume_mode,
                "torch_version": torch.__version__,
            }
            print_resolved_config(
                entrypoint="pretrain.train_long_context", args=args,
                model_config=model_config, runtime=metadata,
                extra={
                    "data": {
                        "fasta": args.fasta,
                        "bed": args.bed,
                        "split_safe_long_sampling": True,
                    }
                },
            )
            log_writer.log(json.dumps(metadata, indent=2, ensure_ascii=False))
            write_run_config(
                output_dir,
                entrypoint="pretrain.train_long_context",
                model_path=Path(__file__).resolve().parents[1] / "vericomp/modeling.py",
                model_class="VeriCompPretrainModel",
                model_config=model_config,
                args=args,
                runtime=metadata,
                extra={
                    "data": {
                        "fasta": args.fasta,
                        "bed": args.bed,
                        "split_safe_long_sampling": True,
                    }
                },
            )

        best_val_loss = float("inf")
        best_path = output_dir / "best.pt"
        if best_path.exists() and rank == 0:
            try:
                previous_best = torch.load(best_path, map_location="cpu", weights_only=False)
                best_val_loss = float(previous_best.get("best_val_loss", "inf"))
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
        if distributed:
            best_tensor = torch.tensor([best_val_loss], device=device)
            dist.broadcast(best_tensor, src=0)
            best_val_loss = float(best_tensor.item())

        train_sampler.set_epoch(epoch)
        iterator = iter(train_loader)
        update = start_update
        last_log = time.perf_counter()
        eta_start = None
        ema_step_seconds = None
        measured_updates = 0
        total_samples_seen = 0
        total_tokens_seen = 0
        total_compute_seconds = 0.0
        while update < args.total_updates:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            last_length = None
            update_samples = 0
            update_tokens = 0
            last_global_batch = 0
            loss_for_log = mlm_for_log = strand_for_log = router_for_log = 0.0
            correct_for_log = masked_for_log = 0.0
            for micro_step in range(args.grad_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    iterator = iter(train_loader)
                    batch = next(iterator)
                assert_same_length(batch, device)
                last_length = batch["input_ids_pos"].shape[1]
                global_samples = batch["input_ids_pos"].shape[0] * world_size
                last_global_batch = global_samples
                update_samples += global_samples
                update_tokens += global_samples * last_length
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DDP) and micro_step < args.grad_accumulation - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(**batch)
                        loss = out["loss"] / args.grad_accumulation
                    loss.backward()
                loss_for_log += out["loss"].detach().float().item()
                mlm_for_log += out["loss_mlm"].detach().float().item()
                strand_for_log += out["loss_strand"].detach().float().item()
                router_for_log += out["loss_router"].detach().float().item()
                local_correct, local_masked = masked_accuracy(
                    out["logits"].detach(), batch["labels"]
                )
                correct_for_log += local_correct
                masked_for_log += local_masked

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            update += 1
            step_seconds = max(1e-9, time.perf_counter() - step_start)
            total_compute_seconds += step_seconds
            total_samples_seen += update_samples
            total_tokens_seen += update_tokens
            if update == start_update + 1:
                eta_start = time.perf_counter()
            else:
                measured_updates += 1
                ema_step_seconds = (
                    step_seconds
                    if ema_step_seconds is None
                    else 0.9 * ema_step_seconds + 0.1 * step_seconds
                )

            if update % args.eval_every == 0 or update == args.total_updates:
                metrics = evaluate(
                    model,
                    valid_loader,
                    valid_sampler,
                    device=device,
                    world_size=world_size,
                )
                if rank == 0:
                    print(
                        f"validation update={update} examples={metrics['examples']} "
                        f"loss={metrics['loss']:.5f} mlm={metrics['loss_mlm']:.5f} "
                        f"mlm_acc={metrics['mlm_acc']:.5f} "
                        f"strand={metrics['loss_strand']:.6f} "
                        f"router={metrics['loss_router']:.6f}",
                        flush=True,
                    )
                    if metrics["loss"] == metrics["loss"] and metrics["loss"] < best_val_loss:
                        best_val_loss = metrics["loss"]
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            update=update,
                            epoch=epoch,
                            args=args,
                            best_val_loss=best_val_loss,
                        )
                if distributed:
                    best_tensor = torch.tensor([best_val_loss], device=device)
                    dist.broadcast(best_tensor, src=0)
                    best_val_loss = float(best_tensor.item())

            if rank == 0 and (update % args.save_every == 0 or update == args.total_updates):
                save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    update=update,
                    epoch=epoch,
                    args=args,
                    best_val_loss=best_val_loss,
                )

            if rank == 0 and (update == 1 or update % 10 == 0 or update == args.total_updates):
                now = time.perf_counter()
                interval_seconds = max(1e-9, now - last_log)
                last_log = now
                current_samples_s = update_samples / step_seconds
                current_tokens_s = update_tokens / step_seconds
                average_tokens_s = total_tokens_seen / max(total_compute_seconds, 1e-9)
                if eta_start is not None and measured_updates > 0:
                    eta_step = max(0.0, (now - eta_start) / measured_updates)
                    remaining = max(0, args.total_updates - update)
                    eta_text = format_duration(remaining * eta_step)
                else:
                    eta_text = "warming_up"
                log_writer.log(
                    f"update={update}/{args.total_updates} L={last_length} "
                    f"local_batch={last_global_batch // world_size} "
                    f"global_batch={last_global_batch} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} "
                    f"loss={loss_for_log / args.grad_accumulation:.5f} "
                    f"mlm={mlm_for_log / args.grad_accumulation:.5f} "
                    f"mlm_acc={correct_for_log / max(1.0, masked_for_log):.5f} "
                    f"strand={strand_for_log / args.grad_accumulation:.6f} "
                    f"router={router_for_log / args.grad_accumulation:.6f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"step_s={step_seconds:.2f} interval_s={interval_seconds:.2f} "
                    f"throughput_samples_s={current_samples_s:.2f} "
                    f"throughput_tokens_s={current_tokens_s:.2f} "
                    f"avg_tokens_s={average_tokens_s:.2f} "
                    f"tokens_seen={total_tokens_seen} eta={eta_text}"
                )
        if distributed:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        log_writer.close()


if __name__ == "__main__":
    main()
