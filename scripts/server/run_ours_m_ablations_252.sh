#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
PY="${PY:-${ROOT}/.conda_env/bin/python}"
BAD_GPU="${BAD_GPU:-3}"
ALLOWED_GPUS="${ALLOWED_GPUS:-0,1,2,4,5,6,7}"
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"

ABL_ROOT="${ROOT}/paper_experiments/ablations"
CKPT_ROOT="${ABL_ROOT}/checkpoints"
LOG_DIR="${ABL_ROOT}/logs"
PID_DIR="${ABL_ROOT}/pids"
QUEUE_LOG="${LOG_DIR}/ours_m_ablations_queue.log"
QUEUE_PID="${PID_DIR}/ours_m_ablations_queue.pid"

mkdir -p "$CKPT_ROOT" "$LOG_DIR" "$PID_DIR"

die() {
  echo "$*" >&2
  exit 1
}

primary_seed() {
  "$PY" - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('/DATA_2/guest/custom-gopt/paper_experiments/main_comparison/frozen_main_model.json').read_text())
print(int(d['primary_seed']))
PY
}

official_seeds() {
  "$PY" - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('/DATA_2/guest/custom-gopt/paper_experiments/main_comparison/frozen_main_model.json').read_text())
print(' '.join(str(int(s)) for s in d['official_seed_set']))
PY
}

all_train_experiments() {
  printf '%s\n' \
    M_top1_onehot \
    M_no_acoustic \
    M_no_prosody \
    M_no_uncertainty_stats \
    M_vector_gate \
    M_fixed_average \
    M_no_gru \
    M_no_stability \
    M_no_multipa_teacher \
    M_no_closed_gopt_teacher \
    M_no_teachers \
    M_no_stress_weight \
    M_no_auxiliary
}

all_inference_experiments() {
  printf '%s\n' M_replay_all_committed
}

allowed_gpus_list() {
  echo "$ALLOWED_GPUS" | tr ',' '\n' | while read -r gpu; do
    gpu="$(echo "$gpu" | tr -d ' ')"
    [[ -n "$gpu" && "$gpu" != "$BAD_GPU" ]] && echo "$gpu"
  done
}

gpu_mem_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -n 1 | tr -d ' '
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -n 1 | tr -d ' '
}

gpu_compute_pids() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$gpu" '$1 == gpu && $2 ~ /^[0-9]+$/ && $3 == "C" {print $2}'
}

own_running_jobs_on_gpu() {
  local gpu="$1" gpu_file pid_file pid count=0
  for gpu_file in "${PID_DIR}"/*.gpu; do
    [[ -f "$gpu_file" ]] || continue
    [[ "$(cat "$gpu_file" 2>/dev/null || true)" == "$gpu" ]] || continue
    pid_file="${gpu_file%.gpu}.pid"
    [[ -f "$pid_file" ]] || continue
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

gpu_is_idle() {
  local gpu="$1" mem util pids own
  mem="$(gpu_mem_used_mib "$gpu")"
  util="$(gpu_util "$gpu")"
  pids="$(gpu_compute_pids "$gpu" | tr '\n' ' ')"
  own="$(own_running_jobs_on_gpu "$gpu")"
  [[ "$mem" =~ ^[0-9]+$ ]] || return 1
  [[ "$util" =~ ^[0-9]+$ ]] || return 1
  [[ "$own" -eq 0 ]] || return 1
  [[ -z "${pids// /}" ]] || return 1
  [[ "$mem" -lt 1000 ]] || return 1
  [[ "$util" -lt 10 ]] || return 1
}

wait_for_idle_gpu() {
  local gpu
  while true; do
    while read -r gpu; do
      [[ -n "$gpu" ]] || continue
      if gpu_is_idle "$gpu"; then
        echo "$gpu"
        return 0
      fi
    done < <(allowed_gpus_list)
    echo "[wait] no idle allowed GPU; GPU${BAD_GPU} skipped; sleeping ${SLEEP_SEC}s" >&2
    sleep "$SLEEP_SEC"
  done
}

print_gpu_status() {
  local gpu mem util pids state
  echo "GPU policy: allowed=${ALLOWED_GPUS}; GPU${BAD_GPU}=skipped; idle requires no compute pid, mem<1000MiB, util<10%"
  while read -r gpu; do
    mem="$(gpu_mem_used_mib "$gpu" || echo '?')"
    util="$(gpu_util "$gpu" || echo '?')"
    pids="$(gpu_compute_pids "$gpu" | tr '\n' ',' | sed 's/,$//')"
    own="$(own_running_jobs_on_gpu "$gpu")"
    if gpu_is_idle "$gpu"; then state="idle"; else state="busy"; fi
    echo "gpu=${gpu} mem=${mem}MiB util=${util}% compute_pids=${pids:-none} own_queue_jobs=${own} ${state}"
  done < <(allowed_gpus_list)
  echo "gpu=${BAD_GPU} skipped"
}

prepare() {
  cd "$ROOT"
  "$PY" scripts/paper/prepare_ours_m_ablations.py prepare
}

exp_dir_for() {
  local exp="$1" seed="$2"
  echo "${CKPT_ROOT}/${exp}_seed${seed}"
}

is_started_or_done() {
  local exp_dir="$1"
  [[ -f "${exp_dir}/models/best_audio_model.pth" || -f "${exp_dir}/last_checkpoint.pt" || -f "${exp_dir}/test_metrics.json" ]]
}

print_one() {
  local exp="$1" seed="$2" exp_dir
  exp_dir="$(exp_dir_for "$exp" "$seed")"
  mapfile -t args < <(cd "$ROOT" && "$PY" scripts/paper/prepare_ours_m_ablations.py args --experiment "$exp" --seed "$seed" --exp-dir "$exp_dir")
  printf 'CUDA_VISIBLE_DEVICES=<idle_non3> %q src/train_streaming_pcn.py' "$PY"
  printf ' %q' "${args[@]}"
  printf '\n'
}

print_plan() {
  prepare
  local seed exp
  seed="$(primary_seed)"
  echo "ROOT=$ROOT"
  echo "ABL_ROOT=$ABL_ROOT"
  echo "PRIMARY_SEED=$seed"
  echo "ALLOWED_GPUS=$ALLOWED_GPUS"
  for exp in $(all_train_experiments); do
    echo "[$exp seed=$seed]"
    print_one "$exp" "$seed"
  done
  echo "[inference-only]"
  all_inference_experiments
}

start_exp() {
  local exp="$1" seed="${2:-$(primary_seed)}" exp_dir pid_file gpu
  exp_dir="$(exp_dir_for "$exp" "$seed")"
  pid_file="${PID_DIR}/${exp}_seed${seed}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] ${exp}_seed${seed} already running pid=$(cat "$pid_file")"
    return 0
  fi
  if is_started_or_done "$exp_dir" && [[ "$FORCE" != "1" ]]; then
    echo "[skip] ${exp}_seed${seed} already has checkpoint/result: $exp_dir"
    return 0
  fi
  mkdir -p "$exp_dir" "$LOG_DIR" "$PID_DIR"
  mapfile -t args < <(cd "$ROOT" && "$PY" scripts/paper/prepare_ours_m_ablations.py args --experiment "$exp" --seed "$seed" --exp-dir "$exp_dir")
  gpu="$(wait_for_idle_gpu)"
  echo "[run] ${exp}_seed${seed} on GPU${gpu}; GPU${BAD_GPU} skipped"
  (
    cd "$ROOT"
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/train_streaming_pcn.py "${args[@]}"
  ) > "${LOG_DIR}/${exp}_seed${seed}.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "$gpu" > "${PID_DIR}/${exp}_seed${seed}.gpu"
  echo "[run] pid=$! log=${LOG_DIR}/${exp}_seed${seed}.log exp_dir=${exp_dir}"
  sleep 8
}

run_all() {
  prepare
  local seed exp
  seed="$(primary_seed)"
  for exp in $(all_train_experiments); do
    start_exp "$exp" "$seed"
  done
  echo "[queue] launch attempts complete $(date '+%F %T')"
}

start_queue() {
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue=running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
    return 0
  fi
  local script
  script="$(readlink -f "$0")"
  nohup "$script" run_all >> "$QUEUE_LOG" 2>&1 &
  echo "$!" > "$QUEUE_PID"
  echo "queue_pid=$! log=$QUEUE_LOG"
}

status() {
  echo "ABL_ROOT=$ABL_ROOT"
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue=running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
  else
    echo "queue=not-running log=$QUEUE_LOG"
  fi
  print_gpu_status
  local seed exp exp_dir pid_file state
  seed="$(primary_seed)"
  for exp in $(all_train_experiments); do
    exp_dir="$(exp_dir_for "$exp" "$seed")"
    pid_file="${PID_DIR}/${exp}_seed${seed}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      state="running pid=$(cat "$pid_file") gpu=$(cat "${PID_DIR}/${exp}_seed${seed}.gpu" 2>/dev/null || echo '?')"
    elif [[ -f "${exp_dir}/test_metrics.json" ]]; then
      state="done"
    elif [[ -f "${exp_dir}/models/best_audio_model.pth" ]]; then
      state="best_only"
    elif [[ -f "${exp_dir}/last_checkpoint.pt" ]]; then
      state="checkpoint"
    else
      state="missing"
    fi
    echo "${exp}_seed${seed}: ${state}"
  done
  echo "M_replay_all_committed: inference-only via summarize"
}

summarize() {
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/paper/summarize_ours_m_ablations.py
}

case "${1:-status}" in
  prepare)
    prepare
    ;;
  print)
    print_plan
    ;;
  run)
    [[ $# -ge 2 ]] || die "usage: $0 run EXP [SEED]"
    start_exp "$2" "${3:-$(primary_seed)}"
    ;;
  run_all)
    run_all
    ;;
  start)
    start_queue
    ;;
  status)
    status
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 {prepare|print|run EXP [SEED]|run_all|start|status|summarize}" >&2
    exit 2
    ;;
esac
