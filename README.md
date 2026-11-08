# HelixPack

HelixPack is a long-DNA-sequence modeling project. Its core method, VeriComp, is a compression-based Transformer/MoE model. A shallow Mamba2 encoder first extracts high-resolution features, a length-adaptive quota compressor then squeezes the sequence into a bottleneck representation, and a Transformer/MoE bottleneck backbone processes the compressed representation.

The default model has about 100M parameters, with 12 Transformer layers, a 512-dim hidden size, 8 attention heads, 16 MoE experts with Top-2 routing, and up to 4096 bottleneck tokens.

## Core Mechanisms

1. **Length-adaptive compression**: the number of compressed tokens is determined by `N(L)=min(L,4096)`. Short sequences keep per-position representations, while long sequences automatically switch to compressed mode.
2. **Content-aware quota routing**: a boundary-aware router allocates quotas according to changes in neighboring context while strictly preserving the total compression budget.
3. **Sparse and invertible compression mapping**: a three-neighborhood sparse quota aggregation together with a dual dynamic inverse allocation avoids the dense `[B,N,L]` memory cost on very long inputs.
4. **Dual-strand cooperative compression**: the forward and reverse-complement strands propose compression candidates separately, and the final quotas are decided jointly in a unified coordinate system.
5. **Decoupled compressed backbone and decoded features**: downstream tasks can choose the compressed bottleneck features, decoder features, or per-base features.

| Input length | Bottleneck tokens | Compression ratio | Mode |
|---:|---:|---:|---|
| 4,096 | 4,096 | 1x | No compression |
| 32,768 | 4,096 | 8x | Adaptive compression |
| 131,072 | 4,096 | 32x | Adaptive compression |
| 1,048,576 | 4,096 | 256x | Long-sequence inference |

## Repository Layout

```
HelixPack/
├── vericomp/     # Model definitions and data pipeline (modeling.py is the core model)
├── pretrain/     # Pretraining code and launch scripts
├── finetune/     # Downstream finetuning code and launch scripts
└── tools/        # Data download and result aggregation scripts
```

Key files:

- `vericomp/modeling.py`: the VeriComp model, including the compressor, dual-strand mechanism, and MoE backbone
- `vericomp/config.py`: model configuration
- `vericomp/model_presets.py`: preset model sizes
- `vericomp/data.py`: MLM pretraining dataset and samplers
- `pretrain/train.py`: main entry point for mixed-length MLM pretraining
- `pretrain/train_long_context.py`: long-context continued pretraining
- `pretrain/next_token.py`: next-token prediction objective
- `pretrain/multitask_mlm_genomic.py`, `pretrain/multitask_mlm_tisp.py`: multitask MLM with auxiliary genomic tasks
- `pretrain/eval_checkpoints.py`, `pretrain/evaluate_long_context.py`, `pretrain/verify.py`: evaluation and verification
- `finetune/<TASK>/`: finetuning code for the six downstream tasks (eQTL, ETGP, CMP, TISP, NT, GB); `<TASK>.py` is the main entry point, `start_<task>.sh` is the launch script, and `external_*.py` contains baseline implementations

## Environment and Dependencies

The project is developed with PyTorch 2.8 and CUDA 12.8, and requires CUDA with BF16 support. Mamba2 depends on `mamba_ssm`. The main third-party dependencies are `torch`, `mamba_ssm`, `transformers`, `datasets`, `huggingface_hub`, and `numpy`.

## Pretraining

`pretrain/start_train.sh` is the entry point for mixed-length MLM pretraining. By default it covers length buckets from 1K to 131K, and every global batch is length-homogeneous. All hyperparameters can be overridden through environment variables:

```bash
CUDA_DEVICES=0,1,2,3 bash pretrain/start_train.sh
```

Other entry points:

```bash
bash pretrain/start_next_token.sh       # Next-token prediction
bash pretrain/start_long_context.sh     # 128K long-context continued pretraining
bash pretrain/start_train_1b.sh         # 1B-scale pretraining
```

## Data

Pretraining data uses a masked-fragment pipeline over the hg38 reference genome; see the command-line arguments of `pretrain/train.py` for the configuration. Downstream benchmark data is downloaded with:

```bash
python tools/download_dnalongbench.py --tasks etgp cmp tisp    # Downloads to data/dnalongbench/
python tools/download_benchmark_data.py --tasks nt gb          # Downloads to data/benchmark/
```

## Downstream Finetuning

Each task lives in its own directory. After loading a pretrained checkpoint, the default setting is full-parameter finetuning with periodic validation and testing, saving `best.pt`, `last.pt`, and `metrics.json`. Launch examples:

```bash
CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/NT/start_nt.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/GB/start_gb.sh

CHECKPOINT=/path/to/checkpoint.pt \
  CELL_TYPE=Artery_Tibial \
  bash finetune/eQTL/start_eqtl.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/TISP/start_tisp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/ETGP/start_etgp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  CELL_TYPE=CRISPRi_EPI_K562_hg19 \
  bash finetune/CMP/start_cmp.sh
```

Evaluation protocol: NT and GB use 5-fold cross-validation (`FOLD` in 0-4, `NUM_FOLDS` defaults to 5); the other tasks (eQTL, ETGP, CMP, TISP) repeat training over three seeds by default (`SEEDS="1 2 3"`, overridable).

Supported model features and metrics for each task are listed in `finetune/README.md`. Baseline evaluation code (NTv3, HyenaDNA, Caduceus, JanusDNA, Evo2, etc.) lives in each task's `external_*.py` files and the same-named baseline files.
