"""Full-backbone 128k eQTL experiments for the external DNA models.

The recipe follows the GeneZip eQTL experiment: one classifier head is trained
for each tissue, with the pretrained sequence encoder fine-tuned at 128,000
bases.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vericomp.eqtl_data import EQTLDataset  # noqa: E402


TISSUES = (
    "Artery_Tibial",
    "Whole_Blood",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Adipose_Subcutaneous",
    "Nerve_Tibial",
    "Muscle_Skeletal",
    "Thyroid",
    "Cells_Cultured_fibroblasts",
)


def materialize_inference_tensors(module: nn.Module) -> None:
    """Make checkpoint-created inference buffers usable by autograd."""
    for child in module.modules():
        for name, parameter in list(child._parameters.items()):
            if parameter is not None and torch.is_inference(parameter):
                child._parameters[name] = nn.Parameter(
                    parameter.detach().clone(), requires_grad=parameter.requires_grad
                )
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and torch.is_inference(buffer):
                child._buffers[name] = buffer.detach().clone()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _last_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("last_hidden_state", "hidden_states"):
            value = output.get(key)
            if value is not None:
                return value[-1] if isinstance(value, (tuple, list)) else value
    for key in ("last_hidden_state", "hidden_states"):
        value = getattr(output, key, None)
        if value is not None:
            return value[-1] if isinstance(value, (tuple, list)) else value
    if isinstance(output, (tuple, list)):
        # HF causal models conventionally return logits first and hidden states
        # later.  The caller should only reach this fallback for custom models.
        for value in reversed(output):
            if torch.is_tensor(value) and value.ndim == 3:
                return value
    raise RuntimeError(f"cannot find hidden states in output of type {type(output)}")


def _hidden_state_tuple(output):
    """Return every hidden state, not just the last one."""
    if isinstance(output, dict):
        value = output.get("hidden_states")
        if isinstance(value, (tuple, list)):
            return tuple(value)
    value = getattr(output, "hidden_states", None)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise RuntimeError(
        f"cannot find hidden states in output of type {type(output)}")


class Backbone:
    def __init__(
        self, name: str, path: str, device: torch.device, *,
        janusdna_bf16_autocast: bool = False,
        ntv3_bf16_compute: bool = False,
        ntv3_feature_layer=None,
    ):
        self.name, self.path, self.device = name, path, device
        self.janusdna_bf16_autocast = bool(janusdna_bf16_autocast)
        self.ntv3_bf16_compute = bool(ntv3_bf16_compute)
        # NTv3 exposes 20 hidden states: 7 down-conv blocks, 6 Transformer
        # layers, 7 deconv blocks.  The released README calls the final
        # deconv output the "final embedding", but the Transformer stack is
        # the backbone representation proper, so feature extraction has to
        # be able to pick either.  None keeps the released convention.
        self.ntv3_feature_layer = ntv3_feature_layer
        self.model: nn.Module
        self.tokenizer: Any = None
        self.kind = name
        self._load()

    def _load(self) -> None:
        if self.name in {"hyenadna", "caduceus_ph", "caduceus_ps"}:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from transformers.modeling_utils import PreTrainedModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.path, trust_remote_code=True, local_files_only=True
            )
            if self.name == "hyenadna":
                model_cls = AutoModelForCausalLM
            else:
                model_cls = AutoModelForMaskedLM
            # Caduceus checkpoints ship a tie_weights(self) override, while
            # newer Transformers calls tie_weights(recompute_mapping=False)
            # during model construction.  Resolve the remote class first and
            # add a temporary compatibility wrapper only while loading.
            restore_init_weights = None
            restore_tie_weights = None
            if self.name != "hyenadna":
                # Transformers 5.x invokes init_weights(), which in turn
                # passes recompute_mapping=False to custom Caduceus
                # tie_weights().  The released Caduceus class predates that
                # keyword.  Keep the normal initialization path and call its
                # tie_weights method without the new keyword during loading.
                original_init_weights = PreTrainedModel.init_weights

                def _init_weights_compat(instance, *args, **kwargs):
                    # Transformers 5.x renamed initialize_weights() to the
                    # private _initialize_weights() hook.  Support both
                    # versions while retaining the released Caduceus class.
                    if hasattr(instance, "initialize_weights"):
                        instance.initialize_weights()
                    else:
                        instance.apply(instance._initialize_weights)
                    instance.tie_weights()

                PreTrainedModel.init_weights = _init_weights_compat
                restore_init_weights = original_init_weights
                config = AutoConfig.from_pretrained(
                    self.path, trust_remote_code=True, local_files_only=True
                )
                resolved_cls = get_class_from_dynamic_module(
                    config.auto_map["AutoModelForMaskedLM"], self.path, local_files_only=True
                )
                original_tie_weights = resolved_cls.tie_weights

                def _tie_weights_compat(instance, *args, **kwargs):
                    return original_tie_weights(instance)

                resolved_cls.tie_weights = _tie_weights_compat
                restore_tie_weights = (resolved_cls, original_tie_weights)
            try:
                self.model = model_cls.from_pretrained(
                    self.path, trust_remote_code=True, local_files_only=True,
                    torch_dtype=torch.bfloat16,
                )
            finally:
                if restore_init_weights is not None:
                    PreTrainedModel.init_weights = restore_init_weights
                if restore_tie_weights is not None:
                    restore_tie_weights[0].tie_weights = restore_tie_weights[1]
            self.model.to(self.device).eval()
            materialize_inference_tensors(self.model)
            return

        if self.name == "ntv3":
            src = "/gpfs_data/Projects/GenomeModelProject/sources/ntv3_base_model"
            sys.path.insert(0, src)
            from configuration_ntv3_pretrained import Ntv3PreTrainedConfig
            from modeling_ntv3_pretrained import NTv3PreTrained
            from tokenization_ntv3 import NTv3Tokenizer

            config = Ntv3PreTrainedConfig.from_pretrained(self.path, local_files_only=True)
            # The released config pins every block to float32 compute, and the
            # port's ``_autocast_to`` honours that by switching the outer
            # autocast off, so a plain bf16 autocast wrapper never reaches the
            # model.  Repin the compute dtypes to bfloat16 for fp32 parameters
            # plus bf16 autocast, the precision convention used everywhere
            # else.  Parameter dtypes are not touched.
            if self.ntv3_bf16_compute:
                for _field in [f for f in dir(config) if f.endswith("_compute_dtype")]:
                    setattr(config, _field, "bfloat16")
            self.tokenizer = NTv3Tokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = NTv3PreTrained.from_pretrained(
                self.path, config=config, local_files_only=True,
            )
            # The released NTv3 PyTorch port keeps its convolution stem in
            # float32; casting the whole module to bf16 makes the embedding
            # output and first convolution incompatible.  Autocast below
            # still provides bf16 compute where the implementation supports it.
            self.model.to(self.device).eval()
            materialize_inference_tensors(self.model)
            return

        if self.name == "janusdna":
            source_root = "/gpfs_data/Projects/GenomeModelProject/sources/JanusDNA"
            src = f"{source_root}/src"
            sys.path.insert(0, source_root)
            sys.path.insert(0, src)
            # The released JanusDNA module wraps FlexAttention with
            # ``torch.compile``.  This compilation is required at 128k:
            # bypassing it makes PyTorch fall back to dense math attention.
            from janusdna.configuration_janusdna import JanusDNAConfig
            from janusdna.modeling_janusdna import (
                JanusDNAForCausalLM,
                JanusDNAModel,
                JanusDNAPreTrainedModel,
            )

            # JanusDNA implements its own FlexAttention module, but the
            # generic Transformers capability check does not know about this
            # custom architecture.  Mark the class as FlexAttention-capable
            # so the requested backend is not rejected during construction.
            JanusDNAPreTrainedModel._supports_flex_attn = True
            JanusDNAModel._supports_flex_attn = True
            JanusDNAForCausalLM._supports_flex_attn = True

            config_file = next(Path(self.path).glob("*/model_config.json"), None)
            ckpt_file = next(Path(self.path).glob("*/checkpoints/last.ckpt"), None)
            if config_file is None or ckpt_file is None:
                raise FileNotFoundError(f"JanusDNA checkpoint layout not found under {self.path}")
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            cfg = dict(raw["config"])
            cfg.pop("_target_", None)
            # The pretrained checkpoint uses a pretraining coefficient, but
            # the official eQTL fine-tuning recipe overrides it to 0.02.
            cfg["output_router_logits"] = True
            cfg["router_aux_loss_coef"] = 0.02
            # The requested speed-oriented run disables activation
            # checkpointing and uses batch 1 with gradient accumulation.
            # This trades higher memory for shorter per-step latency.
            cfg["gradient_checkpointing"] = False
            cfg["use_cache"] = False
            # JanusDNA's long-range fusion uses a custom block mask.  The
            # official downstream implementation uses FlexAttention, which
            # also supports the FP32 path used in the paper.  Do not replace
            # this with SDPA: SDPA would materialize a dense 128k x 128k
            # attention mask.  The previous FlashAttention-2 override was a
            # BF16-only workaround and is not suitable for the FP32
            # reproduction.
            cfg["attn_implementation"] = "flex_attention"
            cfg["final_attention_class"] = "flex_attention"
            config = JanusDNAConfig(**cfg)
            print(
                "[janusdna] torch.compile enabled; attention implementation=flex_attention",
                flush=True,
            )
            self.model = JanusDNAForCausalLM(config)
            checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            # A FlashAttention-2 conversion uses ``o_proj`` while the
            # released FlexAttention checkpoint stores ``o_projs.0``.  Keep
            # the original key for the FP32 FlexAttention path.
            if cfg["final_attention_class"] == "flash_attention_2" and "model.final_attention.self_attn.o_projs.0.weight" in state:
                state["model.final_attention.self_attn.o_proj.weight"] = state.pop(
                    "model.final_attention.self_attn.o_projs.0.weight"
                )
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"JanusDNA unexpected checkpoint keys: {unexpected[:5]}")
            del checkpoint, state
            # The JanusDNA paper evaluates the eQTL fine-tuning protocol in
            # FP32.  Keep this branch outside the BF16 path used by the other
            # external backbones.
            self.model.to(self.device, dtype=torch.float32).eval()
            materialize_inference_tensors(self.model)
            # Keep the outer CausalLM wrapper for checkpoint compatibility,
            # but use its JanusDNAModel for downstream features. Its
            # last_hidden_state is produced after final_layernorm, final_mlp,
            # and the residual connection. Hooking final_layernorm would
            # silently discard the final MLP block.
            self.model.lm_head.requires_grad_(False)
            return

        if self.name == "ldarnet":
            repo = ROOT / ".cache/ldarnet-official"
            if not repo.exists():
                raise FileNotFoundError(f"LDARNet official repository not found at {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from ldar.models.mixer_seq import LDarForMaskedLM
            from ldar.utils.ckpt import _build_config
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
            config = _build_config(payload, str(self.path))
            self.model = LDarForMaskedLM(config)
            incompatible = self.model.load_state_dict(payload["model"], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"LDARNet checkpoint mismatch: {incompatible}")
            del payload
            return

        raise ValueError(f"unsupported backbone {self.name}")

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        if self.name == "janusdna":
            ids = {"A": 7, "C": 8, "G": 9, "T": 10}
            return torch.tensor(
                [[ids.get(base, 6) for base in sequence] for sequence in sequences],
                dtype=torch.long, device=self.device,
            )
        if self.name in {"hyenadna", "caduceus_ph", "caduceus_ps", "ntv3"}:
            encoded = self.tokenizer(
                sequences, return_tensors="pt", padding=False, truncation=False,
            )
            return encoded["input_ids"].to(self.device)
        if self.name == "ldarnet":
            # Released LDARNet vocabulary is A/C/G/T/N -> 0..4 with PAD=6; the
            # harness feeds equal-length windows, so no padding is added here.
            lut = {"A": 0, "C": 1, "G": 2, "T": 3}
            return torch.tensor(
                [[lut.get(base, 4) for base in sequence] for sequence in sequences],
                dtype=torch.long, device=self.device,
            )
        raise AssertionError(self.name)

    def _ntv3_layer_index(self, states):
        """Resolve the configured NTv3 feature layer to a hidden_states index.

        ``transformer`` selects the final Transformer block, which is the
        backbone representation that lines up with HelixPack bottleneck
        features.  ``deconv``/``final`` keeps the released convention.
        """
        spec = self.ntv3_feature_layer
        total = len(states)
        if spec is None:
            return total - 1
        if isinstance(spec, int):
            return spec if spec >= 0 else total + spec
        name = str(spec).strip().lower()
        if name in {"final", "last", "deconv", "deconv_final"}:
            return total - 1
        if name in {"transformer", "transformer_final", "backbone"}:
            cfg = getattr(self.model, "config", None)
            n_down = int(getattr(cfg, "num_downsamples", 0) or 0)
            n_lay = int(getattr(cfg, "num_layers", 0) or 0)
            if n_down and n_lay:
                return n_down + n_lay - 1
            lengths = [int(s.shape[1]) for s in states]
            lowest = min(lengths)
            return max(i for i, n in enumerate(lengths) if n == lowest)
        raise ValueError(f"unknown ntv3 feature layer {spec!r}")

    def features(
        self,
        sequences: list[str],
        *,
        grad: bool = False,
        output_router_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # Some external checkpoints lazily create long-context rotary caches
        # during the initial inference-only width probe.  Clone those caches
        # before autograd sees them.
        if grad:
            materialize_inference_tensors(self.model)
        ids = self._tokenize(sequences)
        if self.name == "janusdna":
            # Keep the default JanusDNA reproduction in FP32.  ETGP may opt
            # into the established BF16-autocast backbone path while keeping
            # its classifier and objective in FP32.
            with torch.set_grad_enabled(grad), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.janusdna_bf16_autocast,
            ):
                out = self.model.model(
                    input_ids=ids,
                    output_router_logits=output_router_logits,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = _last_hidden(out)
                moe_loss = getattr(out, "router_logits", None) if output_router_logits else None
                if moe_loss is not None:
                    moe_loss = moe_loss.float()
        elif self.name == "ldarnet":
            # LDARNet's backbone asserts that a padding mask is supplied, and its
            # MaskedLMOutput already carries last_hidden_state, so
            # output_hidden_states must not be forwarded (unknown kwargs leak
            # into the mixer stack). The harness feeds equal-length windows, so
            # every position is valid (mask value 1 = keep, as in the released
            # evaluation/eval_nt.py).
            with torch.set_grad_enabled(grad), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
                hidden = _last_hidden(out)
            moe_loss = None
        elif self.name == "ntv3":
            with torch.set_grad_enabled(grad), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=ids, output_hidden_states=True, return_dict=True)
                states = _hidden_state_tuple(out)
                hidden = states[self._ntv3_layer_index(states)]
            moe_loss = None
        else:
            with torch.set_grad_enabled(grad), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=ids, output_hidden_states=True, return_dict=True)
                hidden = _last_hidden(out)
            moe_loss = None
        # Pool immediately so a 128k x hidden-size activation is not retained.
        # Clone out of inference mode before feeding the trainable head; an
        # inference tensor cannot be saved by autograd for the head backward.
        pooled = hidden.float().mean(dim=1).to(device=self.device, dtype=torch.float32)
        pooled = pooled if grad else pooled.clone()
        if output_router_logits:
            return pooled, moe_loss
        return pooled


def _metric(labels: list[int], scores: list[float]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    pred = [int(x >= 0.5) for x in scores]
    result = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
    }
    result["auroc"] = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return result


def _loader(dataset: EQTLDataset, batch_size: int, workers: int, seed: int, shuffle: bool):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, generator=generator,
    )


def _evaluate(
    backbone: Backbone,
    head: nn.Module,
    loader: DataLoader,
) -> dict[str, float]:
    """Evaluate a split without retaining 128k-token activations."""
    backbone_training = backbone.model.training
    head_training = head.training
    backbone.model.eval()
    head.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                head(backbone.features(
                    list(batch["sequence"]), grad=False, output_router_logits=False
                )), dim=-1
            )[:, 1]
            labels.extend(batch["labels"].tolist())
            scores.extend(probs.float().cpu().tolist())
    if backbone_training:
        backbone.model.train()
    if head_training:
        head.train()
    return _metric(labels, scores)


def run_tissue(backbone: Backbone, args: argparse.Namespace, tissue: str, out_dir: Path) -> dict[str, Any]:
    tissue_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_metrics_path = out_dir / "epoch_metrics.jsonl"
    seed_everything(args.seed)
    datasets = {
        split: EQTLDataset(args.data_root, tissue, split, args.context_length, max_records=limit)
        for split, limit in (("train", args.max_train_records), ("valid", args.max_valid_records), ("test", args.max_test_records))
    }
    if min(len(ds) for ds in datasets.values()) == 0:
        raise RuntimeError(f"empty eQTL split for {tissue}: { {k: len(v) for k,v in datasets.items()} }")
    loaders = {
        split: _loader(ds, args.batch_size, args.num_workers, args.seed, split == "train")
        for split, ds in datasets.items()
    }
    print(
        f"[{args.model}] {tissue}: records="
        f"train:{len(datasets['train'])},valid:{len(datasets['valid'])},test:{len(datasets['test'])} "
        f"batches_per_epoch={len(loaders['train'])} optimizer_steps_per_epoch="
        f"{max(1, (len(loaders['train']) + args.grad_accumulation - 1) // args.grad_accumulation)}",
        flush=True,
    )

    # Infer representation width with one real sequence.
    sample = datasets["train"][0]["sequence"]
    # Do not use inference_mode here: FlexAttention caches created during the
    # probe would become inference tensors and cannot later be saved by the
    # training backward graph.  no_grad gives the same memory behavior while
    # keeping subsequent full-backbone training valid.
    with torch.no_grad():
        width = int(backbone.features([sample], grad=False).shape[-1])
    head = nn.Linear(width, 2, bias=True).to(backbone.device, dtype=torch.float32)
    for parameter in backbone.model.parameters():
        parameter.requires_grad_(not args.freeze_backbone)
    if args.freeze_backbone:
        backbone.model.eval()
        trainable = list(head.parameters())
    else:
        backbone.model.train()
        trainable = [p for p in backbone.model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    steps_per_epoch = max(1, (len(loaders["train"]) + args.grad_accumulation - 1) // args.grad_accumulation)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.scheduler_start_factor if warmup_steps else 1.0,
        end_factor=1.0, total_iters=max(1, warmup_steps),
    ) if warmup_steps else None
    loss_fn = nn.CrossEntropyLoss()

    best_score = -float("inf")
    best_state = None
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            feature_output = backbone.features(
                list(batch["sequence"]),
                grad=not args.freeze_backbone,
                output_router_logits=(args.model == "janusdna"),
            )
            if args.model == "janusdna":
                feats, moe_loss = feature_output
            else:
                feats, moe_loss = feature_output, None
            labels = batch["labels"].to(backbone.device)
            logits = head(feats)
            main_loss = loss_fn(logits, labels)
            total_loss = main_loss if moe_loss is None else main_loss + moe_loss
            loss = total_loss / args.grad_accumulation
            loss.backward()
            running_loss += float(total_loss.detach().cpu())
            if step % args.grad_accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None and global_step <= warmup_steps:
                    scheduler.step()
                if global_step % args.log_interval_steps == 0 or global_step == 1 or global_step == total_steps:
                    aux_value = float(moe_loss.detach().cpu()) if moe_loss is not None else 0.0
                    print(
                        f"[{args.model}] {tissue}: optimizer_step={global_step}/{total_steps} "
                        f"epoch={epoch + 1}/{args.epochs} micro_step={step}/{len(loaders['train'])} "
                        f"loss={float(total_loss.detach().cpu()):.6f} "
                        f"main_loss={float(main_loss.detach().cpu()):.6f} moe_loss={aux_value:.6f} "
                        f"elapsed_seconds={time.time() - tissue_start:.1f}",
                        flush=True,
                    )
        # Evaluate both held-out splits once per completed epoch.  The test
        # result is recorded for diagnostics, but checkpoint selection uses
        # validation AUROC only.
        val_metrics = _evaluate(backbone, head, loaders["valid"])
        test_metrics = _evaluate(backbone, head, loaders["test"])
        score = val_metrics["auroc"] if not np.isnan(val_metrics["auroc"]) else val_metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        epoch_record = {
            "model": args.model,
            "tissue": tissue,
            "epoch": epoch + 1,
            "optimizer_step": global_step,
            "validation": val_metrics,
            "test": test_metrics,
            "elapsed_seconds": round(time.time() - tissue_start, 1),
        }
        with epoch_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, allow_nan=True) + "\n")
        print(
            f"[{args.model}] {tissue}: epoch={epoch + 1}/{args.epochs} "
            f"global_step={global_step}/{total_steps} train_loss={running_loss / max(1, len(loaders['train'])):.6f} "
            f"val_auroc={val_metrics['auroc']:.6f} val_balanced_accuracy={val_metrics['balanced_accuracy']:.6f} "
            f"test_auroc={test_metrics['auroc']:.6f} test_balanced_accuracy={test_metrics['balanced_accuracy']:.6f} "
            f"epoch_seconds={time.time() - epoch_start:.1f} tissue_elapsed_seconds={time.time() - tissue_start:.1f}",
            flush=True,
        )

    if best_state is not None:
        head.load_state_dict(best_state)
    test_metrics = _evaluate(backbone, head, loaders["test"])
    result = {
        "model": args.model,
        "tissue": tissue,
        "context_length": args.context_length,
        "frozen_backbone": args.freeze_backbone,
        "settings": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "betas": [args.beta1, args.beta2], "warmup_ratio": args.warmup_ratio,
            "scheduler_start_factor": args.scheduler_start_factor,
            "max_grad_norm": args.max_grad_norm,
            "dtype": (
                "bfloat16_autocast_backbone_float32_head_loss"
                if args.model == "janusdna" and args.janusdna_bf16_autocast
                else ("float32" if args.model == "janusdna" else "bfloat16")
            ),
            "janusdna_bf16_autocast": bool(args.janusdna_bf16_autocast),
            "tf32_enabled": bool(args.model == "janusdna" and args.janusdna_bf16_autocast),
            "output_router_logits_train": args.model == "janusdna",
            "router_aux_loss_coef": 0.02 if args.model == "janusdna" else None,
            "seed": args.seed,
            "log_interval_steps": args.log_interval_steps,
            "eval_interval_epochs": args.eval_interval_epochs,
        },
        "n": {k: len(v) for k, v in datasets.items()},
        "validation": val_metrics,
        "test": test_metrics,
        "representation_dim": width,
    }
    checkpoint = {"classifier": head.state_dict(), "representation_dim": width}
    if not args.freeze_backbone:
        checkpoint["backbone"] = backbone.model.state_dict()
    torch.save(checkpoint, out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(
        f"[{args.model}] {tissue}: completed test_auroc={result['test']['auroc']:.6f} "
        f"test_balanced_accuracy={result['test']['balanced_accuracy']:.6f} "
        f"tissue_elapsed_seconds={time.time() - tissue_start:.1f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["hyenadna", "caduceus_ph", "caduceus_ps", "janusdna", "ntv3", "ldarnet"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default=str(ROOT / "data" / "dnalongbench"))
    parser.add_argument("--output-root", default=str(ROOT / "runs" / "external-eqtl-128k"))
    parser.add_argument("--config", default=str(ROOT / "config" / "external_eqtl_128k.json"))
    parser.add_argument("--cell-type", default="all", choices=["all", *TISSUES])
    parser.add_argument("--context-length", type=int, default=128000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--scheduler-start-factor", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval-steps", type=int, default=1)
    parser.add_argument("--eval-interval-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--janusdna-bf16-autocast", action=argparse.BooleanOptionalAction, default=False,
        help="Use BF16 autocast for JanusDNA backbone operations while keeping parameters, head, and loss FP32.",
    )
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    opt = config["optimization"]
    betas = opt.get("betas", [0.9, 0.95])
    args.beta1 = float(betas[0]) if args.beta1 is None else args.beta1
    args.beta2 = float(betas[1]) if args.beta2 is None else args.beta2
    args.scheduler_start_factor = float(
        opt.get("scheduler_start_factor", 1e-8)
        if args.scheduler_start_factor is None else args.scheduler_start_factor
    )
    expected = {
        "context_length": config["context_length_bp"],
        "epochs": opt["epochs"],
        "learning_rate": opt["learning_rate"],
        "weight_decay": opt["weight_decay"],
        "warmup_ratio": opt["warmup_ratio"],
        "max_grad_norm": opt["max_grad_norm"],
        "seed": opt["seed"],
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise ValueError(f"CLI {field}={getattr(args, field)!r} disagrees with {args.config}: {value!r}")
    effective_batch_size = int(
        opt.get("effective_batch_size", opt["batch_size_per_device"] * opt["gradient_accumulation_steps"])
    )
    if args.batch_size * args.grad_accumulation != effective_batch_size:
        raise ValueError(
            f"batch_size ({args.batch_size}) x grad_accumulation ({args.grad_accumulation}) "
            f"must equal effective batch size {effective_batch_size}"
        )
    logging_cfg = config.get("logging", {})
    if args.log_interval_steps != int(logging_cfg.get("loss_interval_steps", args.log_interval_steps)):
        raise ValueError("CLI log interval disagrees with config")
    if args.eval_interval_epochs != int(logging_cfg.get("validation_interval_epochs", args.eval_interval_epochs)):
        raise ValueError("CLI evaluation interval disagrees with config")
    if args.eval_interval_epochs != int(logging_cfg.get("test_interval_epochs", args.eval_interval_epochs)):
        raise ValueError("validation and test intervals must match")
    if args.eval_interval_epochs != 1:
        raise ValueError("this runner evaluates validation and test once per epoch")
    args.freeze_backbone = bool(config["model_training"]["freeze_backbone"])
    if args.context_length != 128000:
        raise ValueError("this runner intentionally supports the requested 128k context only")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "janusdna" and args.janusdna_bf16_autocast:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    backbone = Backbone(
        args.model, args.model_path, device,
        janusdna_bf16_autocast=(args.model == "janusdna" and args.janusdna_bf16_autocast),
    )
    tissues = TISSUES if args.cell_type == "all" else (args.cell_type,)
    root = Path(args.output_root) / args.model
    root.mkdir(parents=True, exist_ok=True)
    resolved_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    resolved_config["runtime_overrides"] = {
        "model": args.model,
        "model_path": str(Path(args.model_path).resolve()),
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "epochs": args.epochs,
        "janusdna_bf16_autocast": bool(args.janusdna_bf16_autocast),
        "tf32_enabled": bool(args.model == "janusdna" and args.janusdna_bf16_autocast),
        "dtype": (
            "bfloat16_autocast_backbone_float32_head_loss"
            if args.model == "janusdna" and args.janusdna_bf16_autocast
            else ("float32" if args.model == "janusdna" else "bfloat16")
        ),
        "evaluation": "BF16 autocast backbone" if args.model == "janusdna" and args.janusdna_bf16_autocast else "runner default",
    }
    (root / "run_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    all_results = []
    for tissue in tissues:
        print(f"[{args.model}] {tissue}: loading data and fine-tuning backbone plus head", flush=True)
        result = run_tissue(backbone, args, tissue, root / tissue)
        all_results.append(result)
        print(json.dumps(result["test"], sort_keys=True), flush=True)
    (root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(all_results, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
