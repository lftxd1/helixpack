#!/usr/bin/env bash
# Sequential per-GPU worker for the NT/GB external-backbone measurements.
# Usage: run_external_short.sh <gpu> <family> <model> <task> [task ...]
set -u
ROOT=/gpfs_data/Projects/HelixPack
GPU="$1"; FAMILY="$2"; MODEL="$3"; shift 3
MODEL_ROOT=/gpfs_data/Projects/GenomeModelProject/models
case "$MODEL" in
  ntv3)
    PY=/gpfs_data/Projects/HelixPack/.cache/ldarnet-venv/bin/python
    MP="$MODEL_ROOT/ntv3-100m-pre" ;;
  janusdna)
    PY=/gpfs_data/Projects/HelixPack/.cache/ldarnet-venv/bin/python
    MP="$MODEL_ROOT/janusdna" ;;
  evo2)
    PY=/root/miniconda3/envs/evo2_28/bin/python
    MP="$MODEL_ROOT/evo2-1b-base/evo2_1b_base.pt"
    export PYTHONPATH="/gpfs_data/Projects/GenomeModelProject/sources/evo2:$ROOT" ;;
  *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
OUT="$ROOT/reports/ntgb_external_20261021/$FAMILY"
LOGS="$ROOT/reports/ntgb_external_20261021/logs"
mkdir -p "$LOGS"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TASK in "$@"; do
  DEST="$OUT/$MODEL/$TASK"
  if [ -f "$DEST/result.json" ]; then
    echo "$(date -Is) SKIP gpu=$GPU family=$FAMILY model=$MODEL task=$TASK" >> "$LOGS/driver_gpu${GPU}.log"
    continue
  fi
  echo "$(date -Is) START gpu=$GPU family=$FAMILY model=$MODEL task=$TASK" >> "$LOGS/driver_gpu${GPU}.log"
  "$PY" -m finetune.NT.external_short \
    --family "$FAMILY" --dataset "$TASK" --model "$MODEL" --model-path "$MP" \
    --device cuda --out "$DEST" >> "$LOGS/${FAMILY}_${MODEL}_${TASK}.log" 2>&1
  RC=$?
  echo "$(date -Is) EXIT rc=$RC gpu=$GPU family=$FAMILY model=$MODEL task=$TASK" >> "$LOGS/driver_gpu${GPU}.log"
  if [ "$RC" -ne 0 ]; then
    echo "$FAMILY $MODEL $TASK rc=$RC" >> "$LOGS/failures_gpu${GPU}.log"
  fi
done
echo "$(date -Is) DRIVER DONE gpu=$GPU family=$FAMILY model=$MODEL" >> "$LOGS/driver_gpu${GPU}.log"
