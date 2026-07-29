#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
BAD_GPU="${BAD_GPU:-3}"
ALLOWED_GPUS="${ALLOWED_GPUS:-6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-8}"
MAX_GPU_MEM_USED_MIB="${MAX_GPU_MEM_USED_MIB:-22000}"
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"

STRICT_DATA_DIR="${STRICT_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu}"
CORRECT_DATA_DIR="${CORRECT_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa}"
ORACLE_DATA_DIR="${ORACLE_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa_oracle_gopt_full}"
SLOTPROSODY_DATA_DIR="${SLOTPROSODY_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa_oracle_gopt_full_slotprosody}"

RUN_BASE="${RUN_BASE:-${ROOT}/paper_experiments/strict_prefix_am_runs}"
RUN_ROOT="${RUN_ROOT:-${RUN_BASE}/runs}"
ORACLE_RUN_ROOT="${ORACLE_RUN_ROOT:-${RUN_BASE}/oracle_runs}"
STRESS_RUN_ROOT="${STRESS_RUN_ROOT:-${RUN_BASE}/stress_runs}"
LOG_DIR="${LOG_DIR:-${RUN_BASE}/logs}"
PID_DIR="${PID_DIR:-${RUN_BASE}/pids}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/strict_prefix_am_queue.log}"
QUEUE_PID="${QUEUE_PID:-${PID_DIR}/strict_prefix_am_queue.pid}"
PY="${PY:-${ROOT}/.conda_env/bin/python}"

mkdir -p "$RUN_BASE" "$LOG_DIR" "$PID_DIR"

die() {
  echo "$*" >&2
  exit 1
}

allowed_gpus_list() {
  echo "$ALLOWED_GPUS" | tr ',' '\n' | while read -r gpu; do
    gpu="$(echo "$gpu" | tr -d ' ')"
    [[ -n "$gpu" && "$gpu" != "$BAD_GPU" ]] && echo "$gpu"
  done
}

gpu_mem_used_mib() {
  local gpu="$1"
  local mem
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' ')"
  [[ "$mem" =~ ^[0-9]+$ ]] || return 1
  echo "$mem"
}

running_jobs_on_gpu() {
  local gpu="$1"
  local count=0
  local gpu_file pid_file pid
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

gpu_has_slot() {
  local gpu="$1"
  local mem running
  mem="$(gpu_mem_used_mib "$gpu" || echo 999999)"
  running="$(running_jobs_on_gpu "$gpu")"
  [[ "$running" -lt "$PROCS_PER_GPU" ]] || return 1
  [[ "$mem" -lt "$MAX_GPU_MEM_USED_MIB" ]] || return 1
}

wait_for_gpu_slot() {
  local gpu
  while true; do
    while read -r gpu; do
      [[ -n "$gpu" ]] || continue
      if gpu_has_slot "$gpu"; then
        echo "$gpu"
        return 0
      fi
    done < <(allowed_gpus_list)
    echo "[wait] no slot on GPUs ${ALLOWED_GPUS}; GPU${BAD_GPU} skipped; procs_per_gpu=${PROCS_PER_GPU}; max_mem=${MAX_GPU_MEM_USED_MIB}MiB; sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
  done
}

gpu_pmon_pids() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$gpu" '$1 == gpu && $2 ~ /^[0-9]+$/ && $3 == "C" { printf("%s,", $2) }'
}

print_gpu_status() {
  echo "GPU policy: use only GPUs ${ALLOWED_GPUS}; GPU${BAD_GPU}=skipped; per_gpu_slots=${PROCS_PER_GPU}; max_mem=${MAX_GPU_MEM_USED_MIB}MiB"
  local gpu line mem util running pids state
  while read -r gpu; do
    line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' ')"
    mem="${line%,*}"
    util="${line#*,}"
    running="$(running_jobs_on_gpu "$gpu")"
    pids="$(gpu_pmon_pids "$gpu")"
    if gpu_has_slot "$gpu"; then
      state="slot_available"
    else
      state="slot_full_or_mem_high"
    fi
    echo "gpu=${gpu} mem=${mem}MiB util=${util}% queue_running=${running}/${PROCS_PER_GPU} compute_pids=${pids:-none} ${state}"
  done < <(allowed_gpus_list)
  echo "gpu=${BAD_GPU} skipped"
}

validate_data() {
  "$PY" - "$STRICT_DATA_DIR" "$CORRECT_DATA_DIR" "$ORACLE_DATA_DIR" "$SLOTPROSODY_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

strict_dir, correct_dir, oracle_dir, slot_dir = [Path(item) for item in sys.argv[1:]]
for root in [strict_dir, correct_dir, oracle_dir, slot_dir]:
    if not root.exists():
        raise SystemExit(f"missing data dir: {root}")
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"missing metadata.json: {root}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("charsiu_mode") != "prefix_recompute":
        raise SystemExit(f"{root}: expected charsiu_mode=prefix_recompute, got {metadata.get('charsiu_mode')!r}")

summary = {}
for split in ["train", "val", "test"]:
    with np.load(correct_dir / f"{split}_chunks.npz") as arr:
        comp = float(arr["teacher_utt_dim_mask"][:, 1].sum()) if "teacher_utt_dim_mask" in arr.files else 0.0
        teacher_rows = int((arr["teacher_utt_mask"] > 0).sum())
        total = int(arr["cn_post"].shape[0])
        if comp != 0.0:
            raise SystemExit(f"{split}: teacher completeness mask sum must be 0, got {comp}")
        if teacher_rows / max(1, total) < 0.90:
            raise SystemExit(f"{split}: teacher coverage too low {teacher_rows}/{total}")
    with np.load(oracle_dir / f"{split}_chunks.npz") as arr:
        comp = float(arr["oracle_utt_dim_mask"][:, 1].sum())
        if "oracle_prefix_utt_dim_mask" in arr.files:
            comp += float(arr["oracle_prefix_utt_dim_mask"][:, 1].sum())
        if "oracle_final_utt_dim_mask" in arr.files:
            comp += float(arr["oracle_final_utt_dim_mask"][:, 1].sum())
        oracle_rows = int((arr["oracle_utt_mask"] > 0).sum())
        total = int(arr["cn_post"].shape[0])
        if comp != 0.0:
            raise SystemExit(f"{split}: oracle completeness mask sum must be 0, got {comp}")
        if oracle_rows / max(1, total) < 0.90:
            raise SystemExit(f"{split}: oracle coverage too low {oracle_rows}/{total}")
        summary[split] = {
            "rows": total,
            "teacher_rows": teacher_rows,
            "oracle_rows": oracle_rows,
            "has_slot_prosody": "slot_prosody" in arr.files,
        }
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

base_ab_args() {
  local data_dir="$1"
  local policy="$2"
  printf '%s\n' \
    --data-dir "$data_dir" \
    --n-epochs 80 \
    --batch-size 16 \
    --num-workers 4 \
    --utt-dim-weights 1,0,1,1,1 \
    --loss-w-phone 1.0 \
    --loss-w-word 1.0 \
    --loss-w-utt 1.0 \
    --loss-w-teacher-score 1.0 \
    --loss-w-prefix-kd 1.0 \
    --loss-w-rank 0.2 \
    --loss-w-phone-stability 0.01 \
    --loss-w-word-stability 0.01 \
    --loss-w-utt-stability 0.01 \
    --soft-label-policy "$policy"
}

oracle_common_args() {
  printf '%s\n' \
    --data-dir "$ORACLE_DATA_DIR" \
    --n-epochs 80 \
    --batch-size 16 \
    --num-workers 4 \
    --utt-dim-weights 1,0,1,1,1 \
    --loss-w-phone 1.0 \
    --loss-w-word 1.0 \
    --loss-w-utt 1.0 \
    --loss-w-teacher-score 0.8 \
    --loss-w-prefix-kd 1.0 \
    --loss-w-rank 0.2 \
    --loss-w-oracle-phone 0.3 \
    --loss-w-oracle-word 1.0 \
    --loss-w-oracle-utt-prefix 0.0 \
    --loss-w-oracle-utt-final 0.0 \
    --loss-w-phone-stability 0.01 \
    --loss-w-word-stability 0.01 \
    --loss-w-utt-stability 0.01 \
    --utt-pooling-head gru_visible \
    --tf32
}

stress_base_args() {
  local data_dir="$1"
  printf '%s\n' \
    --data-dir "$data_dir" \
    --n-epochs 80 \
    --batch-size 12 \
    --num-workers 4 \
    --utt-dim-weights 1,0,1,1,1 \
    --loss-w-phone 1.0 \
    --loss-w-word 1.0 \
    --loss-w-utt 1.0 \
    --loss-w-teacher-score 0.8 \
    --loss-w-prefix-kd 1.0 \
    --loss-w-rank 0.2 \
    --loss-w-oracle-phone 0.3 \
    --loss-w-oracle-word 1.0 \
    --loss-w-oracle-utt-prefix 0.2 \
    --loss-w-oracle-utt-final 0.3 \
    --loss-w-phone-stability 0.01 \
    --loss-w-word-stability 0.01 \
    --loss-w-utt-stability 0.01 \
    --utt-pooling-head gru_visible \
    --fusion-mode concat_vector_gate \
    --embed-dim 64 \
    --depth 3 \
    --heads 4 \
    --gru-dim 64 \
    --grad-clip-norm 1.0 \
    --tf32
}

stress_weight_args() {
  printf '%s\n' --word-dim-weights 1,3,1 --teacher-word-dim-weights 1,2,1 --oracle-word-dim-weights 1,5,1
}

stress_corr_args() {
  stress_weight_args
  printf '%s\n' \
    --loss-w-stress-pearson 0.2 \
    --loss-w-oracle-stress-pearson 0.3 \
    --loss-w-teacher-stress-pearson 0.1 \
    --loss-w-stress-rank 0.1 \
    --loss-w-oracle-stress-rank 0.1
}

stress_detached_args() {
  stress_corr_args
  printf '%s\n' --stress-branch detached --stress-loss-mask vowel
}

exp_dir_for() {
  local exp="$1"
  case "$exp" in
    A_loss_dimmask|B_relaxed_softlabel) echo "${RUN_ROOT}/${exp}" ;;
    C_oracle_word_phone|D_oracle_sentence_light|E_oracle_sentence_balanced|F_oracle_vector_gate|G_oracle_capacity64) echo "${ORACLE_RUN_ROOT}/${exp}" ;;
    H_stress_weighted_G|I_stress_corr_G|J_stress_detached_branch|K_slot_prosody_stress|L_stress_gradscale_voiced|M_stress_scalar_gate_capacity64) echo "${STRESS_RUN_ROOT}/${exp}" ;;
    *) die "unknown experiment: $exp" ;;
  esac
}

exp_args() {
  local exp="$1"
  case "$exp" in
    A_loss_dimmask) base_ab_args "$CORRECT_DATA_DIR" original ;;
    B_relaxed_softlabel) base_ab_args "$CORRECT_DATA_DIR" relaxed ;;
    C_oracle_word_phone) oracle_common_args ;;
    D_oracle_sentence_light) oracle_common_args; printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3 ;;
    E_oracle_sentence_balanced) oracle_common_args; printf '%s\n' --loss-w-teacher-score 0.5 --loss-w-prefix-kd 0.7 --loss-w-oracle-utt-prefix 0.5 --loss-w-oracle-utt-final 0.7 ;;
    F_oracle_vector_gate) oracle_common_args; printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3 --fusion-mode concat_vector_gate ;;
    G_oracle_capacity64) oracle_common_args; printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3 --fusion-mode concat_vector_gate --embed-dim 64 --depth 3 --heads 4 --gru-dim 64 --batch-size 12 ;;
    H_stress_weighted_G) stress_base_args "$ORACLE_DATA_DIR"; stress_weight_args ;;
    I_stress_corr_G) stress_base_args "$ORACLE_DATA_DIR"; stress_corr_args ;;
    J_stress_detached_branch) stress_base_args "$ORACLE_DATA_DIR"; stress_detached_args ;;
    K_slot_prosody_stress) stress_base_args "$SLOTPROSODY_DATA_DIR"; stress_detached_args ;;
    L_stress_gradscale_voiced) stress_base_args "$SLOTPROSODY_DATA_DIR"; stress_corr_args; printf '%s\n' --stress-branch gradscale --stress-grad-scale 0.2 --stress-loss-mask voiced_or_vowel ;;
    M_stress_scalar_gate_capacity64) stress_base_args "$SLOTPROSODY_DATA_DIR"; stress_detached_args; printf '%s\n' --fusion-mode scalar_gate ;;
    *) die "unknown experiment: $exp" ;;
  esac
}

dedupe_args() {
  "$PY" - "$@" <<'PY'
import sys
replace = {
    "--data-dir", "--n-epochs", "--batch-size", "--num-workers", "--utt-dim-weights",
    "--loss-w-phone", "--loss-w-word", "--loss-w-utt", "--loss-w-teacher-score",
    "--loss-w-prefix-kd", "--loss-w-rank", "--loss-w-oracle-phone", "--loss-w-oracle-word",
    "--loss-w-oracle-utt-prefix", "--loss-w-oracle-utt-final", "--loss-w-phone-stability",
    "--loss-w-word-stability", "--loss-w-utt-stability", "--utt-pooling-head", "--fusion-mode",
    "--embed-dim", "--depth", "--heads", "--gru-dim", "--soft-label-policy",
    "--word-dim-weights", "--teacher-word-dim-weights", "--oracle-word-dim-weights",
    "--loss-w-stress-pearson", "--loss-w-oracle-stress-pearson", "--loss-w-teacher-stress-pearson",
    "--loss-w-stress-rank", "--loss-w-oracle-stress-rank", "--stress-branch",
    "--stress-grad-scale", "--stress-loss-mask", "--grad-clip-norm",
}
args = sys.argv[1:]
out = []
i = 0
while i < len(args):
    item = args[i]
    if item in replace and i + 1 < len(args):
        value = args[i + 1]
        if item in out:
            pos = out.index(item)
            out[pos + 1] = value
        else:
            out.extend([item, value])
        i += 2
    else:
        if item.startswith("--"):
            if item not in out:
                out.append(item)
        else:
            out.append(item)
        i += 1
print("\n".join(out))
PY
}

all_exps() {
  printf '%s\n' \
    A_loss_dimmask B_relaxed_softlabel \
    C_oracle_word_phone D_oracle_sentence_light E_oracle_sentence_balanced F_oracle_vector_gate G_oracle_capacity64 \
    H_stress_weighted_G I_stress_corr_G J_stress_detached_branch K_slot_prosody_stress L_stress_gradscale_voiced M_stress_scalar_gate_capacity64
}

is_done_or_started() {
  local exp_dir="$1"
  [[ -f "${exp_dir}/test_metrics.json" || -f "${exp_dir}/last_checkpoint.pt" || -f "${exp_dir}/models/best_audio_model.pth" ]]
}

start_exp() {
  local exp="$1"
  local exp_dir pid_file gpu
  exp_dir="$(exp_dir_for "$exp")"
  pid_file="${PID_DIR}/${exp}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already running pid=$(cat "$pid_file")"
    return 0
  fi
  if is_done_or_started "$exp_dir" && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already has checkpoint/result: $exp_dir"
    return 0
  fi
  mkdir -p "$(dirname "$exp_dir")" "$LOG_DIR" "$PID_DIR"
  mapfile -t raw < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw[@]}")
  gpu="$(wait_for_gpu_slot)"
  echo "[run] $exp on GPU${gpu}; GPU${BAD_GPU} skipped"
  (
    cd "$ROOT"
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/train_streaming_pcn.py --exp-dir "$exp_dir" "${args[@]}"
  ) > "${LOG_DIR}/${exp}.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "$gpu" > "${PID_DIR}/${exp}.gpu"
  echo "[run] pid=$! log=${LOG_DIR}/${exp}.log exp_dir=${exp_dir}"
  sleep 10
}

run_all() {
  validate_data
  for exp in $(all_exps); do
    start_exp "$exp"
  done
  echo "[queue] all launch attempts complete $(date '+%F %T')"
}

start_queue() {
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue already running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
    return 0
  fi
  mkdir -p "$LOG_DIR" "$PID_DIR"
  local script
  script="$(readlink -f "$0")"
  nohup "$script" run_all >> "$QUEUE_LOG" 2>&1 &
  echo "$!" > "$QUEUE_PID"
  echo "queue_pid=$! log=$QUEUE_LOG"
}

status() {
  echo "RUN_BASE=$RUN_BASE"
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue=running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
  else
    echo "queue=not-running log=$QUEUE_LOG"
  fi
  print_gpu_status
  local exp exp_dir pid_file state
  for exp in $(all_exps); do
    exp_dir="$(exp_dir_for "$exp")"
    pid_file="${PID_DIR}/${exp}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      state="running pid=$(cat "$pid_file") gpu=$(cat "${PID_DIR}/${exp}.gpu" 2>/dev/null || echo '?')"
    elif [[ -f "${exp_dir}/test_metrics.json" ]]; then
      state="done"
    elif [[ -f "${exp_dir}/last_checkpoint.pt" ]]; then
      state="checkpoint"
    elif [[ -f "${exp_dir}/models/best_audio_model.pth" ]]; then
      state="best_only"
    else
      state="missing"
    fi
    echo "${exp}: ${state}"
  done
}

print_plan() {
  echo "STRICT_DATA_DIR=$STRICT_DATA_DIR"
  echo "CORRECT_DATA_DIR=$CORRECT_DATA_DIR"
  echo "ORACLE_DATA_DIR=$ORACLE_DATA_DIR"
  echo "SLOTPROSODY_DATA_DIR=$SLOTPROSODY_DATA_DIR"
  echo "RUN_BASE=$RUN_BASE"
  echo "ALLOWED_GPUS=$ALLOWED_GPUS"
  echo "PROCS_PER_GPU=$PROCS_PER_GPU"
  echo "MAX_GPU_MEM_USED_MIB=$MAX_GPU_MEM_USED_MIB"
  for exp in $(all_exps); do
    echo "[$exp] $(exp_dir_for "$exp")"
  done
}

case "${1:-status}" in
  start)
    start_queue
    ;;
  run_all)
    run_all
    ;;
  validate)
    validate_data
    ;;
  print)
    print_plan
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 {start|run_all|validate|print|status}" >&2
    exit 2
    ;;
esac
