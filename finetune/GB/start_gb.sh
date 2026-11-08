#!/usr/bin/env bash
set -euo pipefail

# JanusDNA GenomicBenchmarks port. One invocation trains one dataset/fold.
# Cross-validation: 5-fold; FOLD in {0,1,2,3,4}.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/latest.pt}"
MODEL_NAME="${MODEL_NAME:-vericomp}"
FEATURE="${FEATURE:-backbone}"
POOLING="${POOLING:-mean}"
HEAD="${HEAD:-linear}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/benchmark/gb}"
DATASET="${DATASET:-dummy_mouse_enhancers_ensembl}"
FOLD="${FOLD:-0}"
NUM_FOLDS="${NUM_FOLDS:-5}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SCHEDULER="${SCHEDULER:-linear}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TEST_EVERY="${TEST_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-1}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/finetune/GB/$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/finetune/GB/$(date +%Y%m%d)}"
LOG_FILE="${LOG_FILE:-}"
DETACH="${DETACH:-0}"
MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-}"
MAX_VALID_RECORDS="${MAX_VALID_RECORDS:-}"
MAX_TEST_RECORDS="${MAX_TEST_RECORDS:-}"
DRY_RUN="${DRY_RUN:-0}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"
cmd=(python -m finetune.GB.gb --checkpoint "$CHECKPOINT" --model "$MODEL_NAME" --feature "$FEATURE"
  --pooling "$POOLING" --head "$HEAD" --head-hidden-dim "$HEAD_HIDDEN_DIM" --head-dropout "$HEAD_DROPOUT"
  --data-root "$DATA_ROOT"
  --dataset "$DATASET" --fold "$FOLD" --num-folds "$NUM_FOLDS" --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS" --lr "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
  --scheduler "$SCHEDULER" --plateau-factor "$PLATEAU_FACTOR" --plateau-patience "$PLATEAU_PATIENCE"
  --max-grad-norm "$MAX_GRAD_NORM" --test-every "$TEST_EVERY" --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR")
if [[ "$FREEZE_ENCODER" == "true" ]]; then cmd+=(--freeze-encoder); else cmd+=(--no-freeze-encoder); fi
[[ -n "$MAX_TRAIN_RECORDS" ]] && cmd+=(--max-train-records "$MAX_TRAIN_RECORDS")
[[ -n "$MAX_VALID_RECORDS" ]] && cmd+=(--max-valid-records "$MAX_VALID_RECORDS")
[[ -n "$MAX_TEST_RECORDS" ]] && cmd+=(--max-test-records "$MAX_TEST_RECORDS")
printf 'Running command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "$DRY_RUN" == "1" ]] && exit 0
if [[ "$DETACH" == "1" ]]; then
  mkdir -p "$LOG_DIR"; [[ -z "$LOG_FILE" ]] && LOG_FILE="${LOG_DIR}/${DATASET}-fold-${FOLD}-seed-${SEED}-$(date +%Y%m%d-%H%M%S).log"
  export PYTHONUNBUFFERED=1
  nohup setsid "${cmd[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
  pid=$!; sleep 2
  kill -0 "$pid" 2>/dev/null || { sed -n '1,160p' "$LOG_FILE" >&2 || true; exit 1; }
  echo "Started GB PID ${pid}"; echo "Log: ${LOG_FILE}"; exit 0
fi
exec "${cmd[@]}"
