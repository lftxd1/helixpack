# HelixPack

HelixPack 是长 DNA 序列建模项目，核心方法 VeriComp 是一个压缩式 Transformer/MoE 模型。模型先通过浅层 Mamba2 编码器提取高分辨率特征，再使用长度自适应配额压缩器将序列压缩到瓶颈表示，最后由 Transformer/MoE 瓶颈主干网络处理压缩表示。

默认模型约 1 亿参数，核心配置为 12 层 Transformer、512 维隐藏层、8 个注意力头、16 个 MoE 专家、Top-2 路由，以及最多 4096 个瓶颈 token。

## 核心机制

1. **长度自适应压缩**：压缩 token 数由 `N(L)=min(L,4096)` 决定，短序列保持逐位置表示，长序列自动进入压缩模式。
2. **内容感知的配额路由**：边界感知路由器根据相邻上下文变化分配配额，并严格保持总压缩预算不变。
3. **稀疏且可逆的压缩映射**：采用三邻域稀疏配额聚合和双动态逆分配，避免超长输入上的稠密 `[B,N,L]` 内存开销。
4. **双链协同压缩**：正链和反向互补链分别提出压缩建议，在统一坐标系中共同决定最终配额。
5. **压缩主干与解码特征解耦**：下游任务可以选择压缩后的瓶颈特征、解码器特征或逐碱基特征。

| 输入长度 | 瓶颈 token 数 | 压缩倍数 | 工作模式 |
|---:|---:|---:|---|
| 4,096 | 4,096 | 1 倍 | 不压缩 |
| 32,768 | 4,096 | 8 倍 | 自适应压缩 |
| 131,072 | 4,096 | 32 倍 | 自适应压缩 |
| 1,048,576 | 4,096 | 256 倍 | 长序列推理 |

## 目录结构

```
HelixPack/
├── vericomp/     # 模型定义与数据管线（modeling.py 为核心模型文件）
├── pretrain/     # 预训练代码与启动脚本
├── finetune/     # 下游任务微调代码与启动脚本
└── tools/        # 数据下载与结果汇总脚本
```

主要文件说明：

- `vericomp/modeling.py`，VeriComp 模型主体，包含压缩器、双链机制与 MoE 主干
- `vericomp/config.py`，模型配置
- `vericomp/model_presets.py`，预设模型规模
- `vericomp/data.py`，MLM 预训练数据集与采样器
- `pretrain/train.py`，混合长度 MLM 预训练主入口
- `pretrain/train_long_context.py`，长上下文续训
- `pretrain/next_token.py`，下一 token 预测目标
- `pretrain/multitask_mlm_genomic.py`、`pretrain/multitask_mlm_tisp.py`，带辅助基因组任务的多任务 MLM
- `pretrain/eval_checkpoints.py`、`pretrain/evaluate_long_context.py`、`pretrain/verify.py`，评估与校验
- `finetune/<TASK>/`，eQTL、ETGP、CMP、TISP、NT、GB 六个下游任务的微调代码，`<TASK>.py` 为主入口，`start_<task>.sh` 为启动脚本，`external_*.py` 为基线模型对照实现

## 环境与依赖

项目在 PyTorch 2.8、CUDA 12.8 下开发，需要 CUDA 与 BF16 支持，Mamba2 依赖 `mamba_ssm`。主要第三方依赖为 `torch`、`mamba_ssm`、`transformers`、`datasets`、`huggingface_hub`、`numpy`。

## 预训练

`pretrain/start_train.sh` 是混合长度 MLM 预训练入口，默认覆盖 1K 至 131K 的长度桶，每个全局 batch 内保持长度一致。所有超参数均可用环境变量覆盖：

```bash
CUDA_DEVICES=0,1,2,3 bash pretrain/start_train.sh
```

其他入口：

```bash
bash pretrain/start_next_token.sh       # 下一 token 预测
bash pretrain/start_long_context.sh     # 128K 长上下文续训
bash pretrain/start_train_1b.sh         # 1B 规模预训练
```

## 数据

预训练数据为 hg38 参考基因组的片段掩码流程，配置见 `pretrain/train.py` 的命令行参数。下游基准数据通过脚本下载：

```bash
python tools/download_dnalongbench.py --tasks etgp cmp tisp    # 下载到 data/dnalongbench/
python tools/download_benchmark_data.py --tasks nt gb          # 下载到 data/benchmark/
```

## 下游微调

每个任务一个目录，加载预训练检查点后默认全参微调，定期验证与测试，保存 `best.pt`、`last.pt` 与 `metrics.json`。启动示例：

```bash
CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/NT/start_nt.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/GB/start_gb.sh

CHECKPOINT=/path/to/checkpoint.pt \
  DATASET=eqtl \
  bash finetune/eQTL/start_eqtl.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/TISP/start_tisp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/ETGP/start_etgp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  CELL_TYPE=CRISPRi_EPI_K562_hg19 \
  bash finetune/CMP/start_cmp.sh
```

各任务支持的模型特征与指标见 `finetune/README.md`。基线模型（NTv3、HyenaDNA、Caduceus、JanusDNA、Evo2 等）的评测实现位于各任务的 `external_*.py` 与基线同名文件。
