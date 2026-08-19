#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/guozuyu/HTP-multiview-independent
SOURCE_ROOT=/workspace/guozuyu/HTP-main
DATA="$SOURCE_ROOT/reproduction/runs/multiview_vq/porto/data"
ROAD_EMBEDDINGS="$SOURCE_ROOT/traj_dataset/porto/road_emb_128d.pt"
RUN="$ROOT/stage_1/runs/independent_tokenizers/porto"
ANALYSIS="$RUN/analysis"
LOG="$RUN/formal.log"
SEEDS=(42 314159 20260819)
FAMILIES=(transformer gru)
VIEWS=(gps grid road)

mkdir -p "$RUN"
exec >>"$LOG" 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

echo "[$(timestamp)] branch=$(git -C "$ROOT" branch --show-current) revision=$(git -C "$ROOT" rev-parse HEAD)"
echo "[$(timestamp)] uv sync frozen"
cd "$ROOT"
uv sync --frozen
uv run python --version
PYTHONPATH=stage_1/src uv run pytest stage_1/tests/test_multiview_pattern.py -q

test -f "$DATA/metadata.json"
test -f "$ROAD_EMBEDDINGS"

wait_for_idle_gpu() {
  local gpu=$1
  while true; do
    local used util
    IFS=',' read -r used util < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    used=${used// /}
    util=${util// /}
    if (( used < 1000 && util < 10 )); then
      sleep 20
      local used2 util2
      IFS=',' read -r used2 util2 < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
      used2=${used2// /}
      util2=${util2// /}
      if (( used2 < 1000 && util2 < 10 )); then
        return
      fi
    fi
    echo "[$(timestamp)] gpu=$gpu busy used=${used}MiB util=${util}%; waiting"
    sleep 30
  done
}

run_one() {
  local family=$1
  local seed=$2
  local view=$3
  local gpu=$4
  local output="$RUN/$family/seed_$seed/$view"
  if [[ -f "$output/metrics.json" ]]; then
    echo "[$(timestamp)] skip completed family=$family seed=$seed view=$view"
    return
  fi
  wait_for_idle_gpu "$gpu"
  echo "[$(timestamp)] start family=$family seed=$seed view=$view gpu=$gpu"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=stage_1/src uv run --frozen python -m multiview_pattern.train_tokenizer \
    --data-dir "$DATA" \
    --road-embeddings "$ROAD_EMBEDDINGS" \
    --output-dir "$output" \
    --view "$view" \
    --encoder-family "$family" \
    --epochs 3 \
    --batch-size 256 \
    --learning-rate 0.0002 \
    --weight-decay 0.0001 \
    --hidden-dim 64 \
    --latent-dim 64 \
    --codebook-size 256 \
    --beta 0.25 \
    --dropout 0.1 \
    --usage-weight 0.05 \
    --seed "$seed" \
    --num-workers 4 \
    --probe-train-trajectories 20000 \
    --device cuda:0
  echo "[$(timestamp)] complete family=$family seed=$seed view=$view gpu=$gpu"
}

jobs=()
job_index=0
for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for view in "${VIEWS[@]}"; do
      gpu=$((job_index % 4))
      run_one "$family" "$seed" "$view" "$gpu" &
      jobs+=("$!")
      job_index=$((job_index + 1))
      if (( ${#jobs[@]} == 4 )); then
        for job in "${jobs[@]}"; do
          wait "$job"
        done
        jobs=()
      fi
    done
  done
done
for job in "${jobs[@]}"; do
  wait "$job"
done

wait_for_idle_gpu 0
echo "[$(timestamp)] start aggregate analysis"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=stage_1/src uv run --frozen python -m multiview_pattern.analyze \
  --run-dir "$RUN" \
  --data-dir "$DATA" \
  --output-dir "$ANALYSIS" \
  --seeds 42 314159 20260819 \
  --codebook-size 256 \
  --bootstrap-replicates 200 \
  --probe-device cuda:0
echo "[$(timestamp)] INDEPENDENT TOKENIZER FORMAL COMPLETE"
