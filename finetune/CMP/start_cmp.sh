#!/usr/bin/env bash
set -euo pipefail

# VeriComp DNALongBench CMP launcher; no .pid sidecar is created.
# Repeated runs: 3 seeds (override with SEEDS="5 6 7").

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/latest.pt}"
MODEL_NAME="${MODEL_NAME:-vericomp}"
FEATURE="${FEATURE:-decoder}"
SUBSET="${SUBSET:-HFF}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/dnalongbench}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRELOAD_CACHE="${PRELOAD_CACHE:-1}"
SEEDS="${SEEDS:-1 2 3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/finetune/CMP/$(date +%Y%m%d)/${SUBSET}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/finetune/CMP/$(date +%Y%m%d)}"
LOG_FILE="${LOG_FILE:-}"
DETACH="${DETACH:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
CHUNK_LENGTH="${CHUNK_LENGTH:-131072}"
CACHE_DIR="${CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
PRECOMPUTE_ONLY="${PRECOMPUTE_ONLY:-0}"
SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-1}"
PRECOMPUTE_RANK="${PRECOMPUTE_RANK:-0}"
PRECOMPUTE_WORLD_SIZE="${PRECOMPUTE_WORLD_SIZE:-1}"
WRITER_WORKERS="${WRITER_WORKERS:-4}"
WRITER_QUEUE_BATCHES="${WRITER_QUEUE_BATCHES:-16}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

pids=()
for SEED in $SEEDS; do
  cmd=(python -m finetune.CMP.cmp --checkpoint "$CHECKPOINT" --model "$MODEL_NAME" --feature "$FEATURE"
    --data-root "$DATA_ROOT" --subset "$SUBSET"
    --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --grad-accumulation "$GRAD_ACCUMULATION" --train-steps "$TRAIN_STEPS"
    --eval-every "$EVAL_EVERY" --lr "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
    --max-grad-norm "$MAX_GRAD_NORM" --max-train-samples "$MAX_TRAIN_SAMPLES"
    --max-valid-samples "$MAX_VALID_SAMPLES" --max-test-samples "$MAX_TEST_SAMPLES"
    --trunk-length "$CHUNK_LENGTH" --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --output-dir "${OUTPUT_DIR}/seed-${SEED}")
  [[ -n "$CACHE_DIR" ]] && cmd+=(--cache-dir "$CACHE_DIR")
  cmd+=(--seed "$SEED")
  cmd+=(--precompute-batch-size "$PRECOMPUTE_BATCH_SIZE")
  cmd+=(--precompute-rank "$PRECOMPUTE_RANK" --precompute-world-size "$PRECOMPUTE_WORLD_SIZE")
  cmd+=(--writer-workers "$WRITER_WORKERS" --writer-queue-batches "$WRITER_QUEUE_BATCHES")
  [[ "$PRECOMPUTE_ONLY" == "1" ]] && cmd+=(--precompute-only)
  [[ "$SKIP_PRECOMPUTE" == "1" ]] && cmd+=(--skip-precompute)
  [[ "$PRELOAD_CACHE" == "1" ]] && cmd+=(--preload-cache)
  printf 'Running command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  [[ "$DRY_RUN" == "1" ]] && continue
  if [[ "$DETACH" == "1" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ -z "$LOG_FILE" ]]; then
      log_file="${LOG_DIR}/cmp-${SUBSET}-seed-${SEED}-$(date +%Y%m%d-%H%M%S).log"
    else
      log_file="${LOG_FILE%.log}-seed-${SEED}.log"
    fi
    export PYTHONUNBUFFERED=1
    nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    kill -0 "$pid" 2>/dev/null || { sed -n '1,160p' "$log_file" >&2 || true; exit 1; }
    echo "Started CMP seed ${SEED} PID ${pid}"
    echo "Log: ${log_file}"
    pids+=("$pid")
  else
    "${cmd[@]}"
  fi
done

[[ "$DRY_RUN" == "1" || "$DETACH" != "1" ]] && exit 0
echo "All CMP seed runs started (${pids[*]})"
