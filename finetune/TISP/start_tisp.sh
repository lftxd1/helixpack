#!/usr/bin/env bash
set -euo pipefail

# VeriComp DNALongBench TISP launcher; no .pid sidecar is created.
# Repeated runs: 3 seeds (override with SEEDS="5 6 7").

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fa}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/weights/pretrain/20260814/vericomp-residual-skip-30k/latest.pt}"
MODEL_NAME="${MODEL_NAME:-vericomp}"
FEATURE="${FEATURE:-decoder}"
DUAL_STRAND="${DUAL_STRAND:-false}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/dnalongbench}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-25000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
NUM_TRAIN_SAMPLES="${NUM_TRAIN_SAMPLES:-0}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ROUTER_LOSS_COEF="${ROUTER_LOSS_COEF:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEEDS="${SEEDS:-1 2 3}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/weights/finetune/TISP/$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/runs/logs/finetune/TISP/$(date +%Y%m%d)}"
LOG_FILE="${LOG_FILE:-}"
DETACH="${DETACH:-0}"
DRY_RUN="${DRY_RUN:-0}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

pids=()
for SEED in $SEEDS; do
  cmd=(python -m finetune.TISP.tisp --checkpoint "$CHECKPOINT" --model "$MODEL_NAME" --feature "$FEATURE"
    --data-root "$DATA_ROOT"
    --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --grad-accumulation "$GRAD_ACCUMULATION" --train-steps "$TRAIN_STEPS"
    --eval-every "$EVAL_EVERY" --num-train-samples "$NUM_TRAIN_SAMPLES"
    --max-valid-samples "$MAX_VALID_SAMPLES" --max-test-samples "$MAX_TEST_SAMPLES"
    --lr "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
    --router-loss-coef "$ROUTER_LOSS_COEF" --max-grad-norm "$MAX_GRAD_NORM"
    --seed "$SEED" --output-dir "${OUTPUT_DIR}/seed-${SEED}")
  if [[ "$DUAL_STRAND" == "true" ]]; then cmd+=(--dual-strand); else cmd+=(--no-dual-strand); fi
  if [[ "$FREEZE_ENCODER" == "true" ]]; then cmd+=(--freeze-encoder); else cmd+=(--no-freeze-encoder); fi
  printf 'Running command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  [[ "$DRY_RUN" == "1" ]] && continue
  if [[ "$DETACH" == "1" ]]; then
    mkdir -p "$LOG_DIR"
    if [[ -z "$LOG_FILE" ]]; then
      log_file="${LOG_DIR}/tisp-seed-${SEED}-$(date +%Y%m%d-%H%M%S).log"
    else
      log_file="${LOG_FILE%.log}-seed-${SEED}.log"
    fi
    export PYTHONUNBUFFERED=1
    nohup setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    kill -0 "$pid" 2>/dev/null || { sed -n '1,160p' "$log_file" >&2 || true; exit 1; }
    echo "Started TISP seed ${SEED} PID ${pid}"
    echo "Log: ${log_file}"
    pids+=("$pid")
  else
    "${cmd[@]}"
  fi
done

[[ "$DRY_RUN" == "1" || "$DETACH" != "1" ]] && exit 0
echo "All TISP seed runs started (${pids[*]})"
