#!/usr/bin/env bash
set -euo pipefail

cd /DATA_2/guest/custom-gopt

DATA_DIR="${DATA_DIR:-/DATA_2/guest/custom-gopt/data/streaming_pcn_gopt_v2_stateful}"
TEACHER_GPUS="${TEACHER_GPUS:-5,6}"
TRAIN_GPU="${TRAIN_GPU:-6}"
RESERVE_GB="${RESERVE_GB:-10}"
POLL_SEC="${POLL_SEC:-60}"

TRAIN_TOTAL="${TRAIN_TOTAL:-2500}"
VAL_TOTAL="${VAL_TOTAL:-1260}"
TEST_TOTAL="${TEST_TOTAL:-1240}"

LOG_PREFIX="[reserve-teacher-watch]"

reserve_pids=()

cleanup_reserve() {
  for pid in "${reserve_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "$(date '+%F %T') $LOG_PREFIX release reserve pid=$pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup_reserve EXIT

start_reserve_one_gpu() {
  local gpu="$1"
  echo "$(date '+%F %T') $LOG_PREFIX reserve ${RESERVE_GB}GB on physical GPU ${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" /DATA_2/guest/custom-gopt/.multipa_env/bin/python - <<PY &
import os, time, torch

gb = float(os.environ.get("RESERVE_GB", "$RESERVE_GB"))
chunks = []
chunk_gb = 1.0

assert torch.cuda.is_available(), "CUDA not available"
dev = torch.device("cuda:0")

n_chunks = int(gb // chunk_gb)
remain = gb - n_chunks * chunk_gb

def alloc(size_gb):
    # fp16: 2 bytes per element
    n = int(size_gb * (1024 ** 3) / 2)
    x = torch.empty(n, dtype=torch.float16, device=dev)
    x.fill_(1.0)
    return x

for _ in range(n_chunks):
    chunks.append(alloc(chunk_gb))
if remain > 0.05:
    chunks.append(alloc(remain))

torch.cuda.synchronize()
print(f"[reserve] gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} reserved≈{gb}GB", flush=True)

while True:
    time.sleep(3600)
PY

  reserve_pids+=("$!")
}

IFS=',' read -ra GPU_LIST <<< "$TEACHER_GPUS"
for gpu in "${GPU_LIST[@]}"; do
  start_reserve_one_gpu "$gpu"
done

progress_count() {
  local split="$1"
  local dir="$DATA_DIR/progress/$split"
  if [ ! -d "$dir" ]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*.pkl' 2>/dev/null | wc -l
}

npz_ready() {
  [ -s "$DATA_DIR/train_chunks.npz" ] && \
  [ -s "$DATA_DIR/val_chunks.npz" ] && \
  [ -s "$DATA_DIR/test_chunks.npz" ] && \
  [ -s "$DATA_DIR/train_manifest.jsonl" ] && \
  [ -s "$DATA_DIR/val_manifest.jsonl" ] && \
  [ -s "$DATA_DIR/test_manifest.jsonl" ]
}

echo "$(date '+%F %T') $LOG_PREFIX watching data generation: $DATA_DIR"
echo "$(date '+%F %T') $LOG_PREFIX teacher_gpus=$TEACHER_GPUS train_gpu=$TRAIN_GPU reserve_gb=$RESERVE_GB"

while true; do
  train_count="$(progress_count train)"
  val_count="$(progress_count val)"
  test_count="$(progress_count test)"
  generator_count="$(pgrep -fc 'build_streaming_pcn_gopt_data.py.*streaming_pcn_gopt_v2_stateful' || true)"

  echo "$(date '+%F %T') $LOG_PREFIX progress train=${train_count}/${TRAIN_TOTAL} val=${val_count}/${VAL_TOTAL} test=${test_count}/${TEST_TOTAL} generators=${generator_count}"

  if npz_ready; then
    echo "$(date '+%F %T') $LOG_PREFIX finalized NPZ/manifest found"
    break
  fi

  sleep "$POLL_SEC"
done

echo "$(date '+%F %T') $LOG_PREFIX data finalized; releasing reserved GPU memory before teacher"
cleanup_reserve
trap - EXIT

sleep 5

echo "$(date '+%F %T') $LOG_PREFIX starting teacher pipeline"
exec env \
  TEACHER_GPUS="$TEACHER_GPUS" \
  TRAIN_GPU="$TRAIN_GPU" \
  POLL_SEC="$POLL_SEC" \
  ./scripts/server/run_pcn_teacher_after_generation.sh
