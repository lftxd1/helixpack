# VeriComp 下游任务

这些脚本沿用 eQTL 流程：加载预训练权重、默认全参微调、定期验证和测试、保存 `best.pt`、`last.pt` 与 `metrics.json`，并将无缓冲日志写入 `runs/logs/finetune/<TASK>/YYYYMMDD/`。

## 数据下载

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fa
python tools/download_dnalongbench.py --tasks etgp cmp tisp
python tools/download_benchmark_data.py --tasks nt gb
```

数据目录分别为 `data/dnalongbench/` 和 `data/benchmark/`。目前支持：

| 任务 | VeriComp 特征 | 主要指标 | 默认数据规模 |
|---|---|---|---|
| eQTL | bottleneck mean pooling | AUROC/AUPRC | `data/dnalongbench/eqtl` |
| ETGP | bottleneck mean pooling | AUROC/AUPRC | `data/dnalongbench/etgp` |
| CMP | 五个细胞系分别使用 decoder 特征预计算 448 个 h-bin，再由 2D CNN head 预测接触图 | SCC/Pearson | `data/dnalongbench/cmp` |
| TISP | bottleneck 插值到 100k 输出网格 | 5 个 assay 的 plus/minus 合并 PCC，再取平均 | `data/dnalongbench/tisp` |
| NT | bottleneck mean pooling | MCC/F1/accuracy | `data/benchmark/nt`，18 个 nucleotide-transformer 任务 |
| GB | bottleneck mean pooling | MCC/F1/accuracy | `data/benchmark/gb`，8 个 GenomicBenchmarks 任务 |

## 启动示例

```bash
CHECKPOINT=/path/to/checkpoint.pt \
  CELL_TYPE=CRISPRi_EPI_K562_hg19 \
  bash finetune/ETGP/start_etgp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/CMP/start_cmp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  bash finetune/TISP/start_tisp.sh

CHECKPOINT=/path/to/checkpoint.pt \
  DATASET=H3K79me3 FOLD=0 \
  bash finetune/NT/start_nt.sh

CHECKPOINT=/path/to/checkpoint.pt \
  DATASET=human_enhancers_ensembl FOLD=0 \
  bash finetune/GB/start_gb.sh
```

小规模 smoke run 可以设置 `MAX_*_RECORDS`、`MAX_*_SAMPLES` 或 `NUM_TRAIN_SAMPLES`。CMP 的原始序列为 1,048,576 bp，包含 `HFF`、`H1hESC`、`GM12878`、`IMR90`、`HCT116` 五个 subset；脚本默认按 131,072 bp trunk 预计算 h-bin，使用 `SUBSET` 选择细胞系；TISP 使用 100,000 bp 窗口。

NT/GB 的 `DATASET` 对应 JanusDNA 原始 benchmark 名称；训练、验证、测试全部只输入同一条正向序列，不做 reverse-complement conjoining，训练默认全参微调。NT 默认 20 个 epoch，GB 默认 10 个 epoch，可通过环境变量覆盖。NT/GB 使用 5 折交叉验证（`FOLD` 取 0 到 4，`NUM_FOLDS` 默认 5），每折的输出保存在 `weights/finetune/{NT,GB}/YYYYMMDD/<dataset>/fold-<k>-seed-<s>/`。其余任务（eQTL、ETGP、CMP、TISP）默认重复 3 个种子（`SEEDS="1 2 3"`，可覆盖），每个种子的输出保存在 `seed-<k>/` 子目录，日志保存在对应的 `runs/logs/finetune/` 目录。

所有下游启动脚本都支持 `MODEL_NAME` 和 `FEATURE=backbone|decoder|base`。其中 `backbone` 是压缩后的 bottleneck，`decoder` 是 decoder 融合后的 full-resolution hidden feature，`base` 是与原始序列等长的单碱基特征；`MODEL_NAME=vericomp` 使用 `vericomp/modeling.py`，`MODEL_NAME=causal` 使用 `vericomp/causal_modeling.py`。

所有启动脚本都支持 `DETACH=1`、`LOG_FILE=...` 和 `DRY_RUN=1`，不会生成 `.log.pid` 文件。

批量调度所有下游任务可使用 [tools/downstream_queue/README.md](../tools/downstream_queue/README.md)，支持为每张 GPU 配置并发任务数。
