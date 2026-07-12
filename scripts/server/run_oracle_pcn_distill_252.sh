#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/DATA_2/guest/custom-gopt}"
EXP_ROOT="${EXP_ROOT:-${REPO}/exp/pcn_extra_20260704_2130}"
ACTION="${1:-status}"
TARGET="${2:-}"
BAD_GPU=3
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"

PY_TRAIN="${PY_TRAIN:-${REPO}/.conda_env/bin/python}"
PY_ORACLE="${PY_ORACLE:-${REPO}/.multipa_env/bin/python}"
ORACLE_JSONL="${EXP_ROOT}/oracle_gopt_closed_prefix_gt_time_all_splits.jsonl"
ORACLE_DATA_DIR="${EXP_ROOT}/data_streaming_pcn_oracle_gopt_full"
ORACLE_RUN_ROOT="${EXP_ROOT}/oracle_runs"
LOG_DIR="${EXP_ROOT}/logs"
PID_DIR="${EXP_ROOT}/pids"

cd "${REPO}"

pcn_data_dir() {
  "${PY_TRAIN}" - "$EXP_ROOT" "$REPO" <<'PY'
import json
import sys
from pathlib import Path
exp_root = Path(sys.argv[1])
repo = Path(sys.argv[2])
config_path = exp_root / "A_loss_dimmask" / "config.json"
if not config_path.exists():
    raise SystemExit(f"missing A config: {config_path}")
config = json.loads(config_path.read_text(encoding="utf-8"))
data_dir = Path(config["data_dir"])
if not data_dir.is_absolute():
    data_dir = repo / data_dir
print(data_dir)
PY
}

gpu_rows() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v g="$gpu" '$1 == g && $2 != "-" {found=1} END {exit found ? 0 : 1}'
}

gpu_reserved_by_tracked_pid() {
  local gpu="$1"
  local pid_file pid visible
  shopt -s nullglob
  for pid_file in "${PID_DIR}"/*.pid; do
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 "$pid" 2>/dev/null || continue
    visible="$(
      tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null \
        | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2; exit}'
    )"
    [[ "$visible" == "$gpu" ]] && return 0
  done
  return 1
}

find_idle_gpu_once() {
  gpu_rows | while IFS=',' read -r raw_idx raw_mem raw_util; do
    idx="$(echo "$raw_idx" | xargs)"
    mem="$(echo "$raw_mem" | xargs)"
    util="$(echo "$raw_util" | xargs)"
    [[ "$idx" == "$BAD_GPU" ]] && continue
    gpu_reserved_by_tracked_pid "$idx" && continue
    [[ "$mem" =~ ^[0-9]+$ ]] || continue
    [[ "$util" =~ ^[0-9]+$ ]] || continue
    if (( mem < 1000 && util < 10 )); then
      if ! gpu_has_compute_process "$idx"; then
        echo "$idx"
        return 0
      fi
    fi
  done
  return 1
}

wait_for_idle_gpu() {
  local gpu=""
  while true; do
    gpu="$(find_idle_gpu_once || true)"
    if [[ -n "$gpu" ]]; then
      echo "$gpu"
      return 0
    fi
    echo "[wait] no idle GPU under policy; GPU${BAD_GPU} is skipped; sleeping ${SLEEP_SEC}s" >&2
    sleep "$SLEEP_SEC"
  done
}

common_args() {
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

exp_args() {
  local exp="$1"
  case "$exp" in
    C_oracle_word_phone)
      common_args
      ;;
    D_oracle_sentence_light)
      common_args
      printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3
      ;;
    E_oracle_sentence_balanced)
      common_args
      printf '%s\n' --loss-w-teacher-score 0.5 --loss-w-prefix-kd 0.7 --loss-w-oracle-utt-prefix 0.5 --loss-w-oracle-utt-final 0.7
      ;;
    F_oracle_vector_gate)
      common_args
      printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3 --fusion-mode concat_vector_gate
      ;;
    G_oracle_capacity64)
      common_args
      printf '%s\n' --loss-w-oracle-utt-prefix 0.2 --loss-w-oracle-utt-final 0.3 --fusion-mode concat_vector_gate --embed-dim 64 --depth 3 --heads 4 --gru-dim 64 --batch-size 12
      ;;
    *)
      echo "unknown experiment: $exp" >&2
      return 1
      ;;
  esac
}

all_exps() {
  printf '%s\n' C_oracle_word_phone D_oracle_sentence_light E_oracle_sentence_balanced F_oracle_vector_gate G_oracle_capacity64
}

dedupe_args() {
  "${PY_TRAIN}" - "$@" <<'PY'
import sys
replace_with_value = {
    "--data-dir",
    "--n-epochs",
    "--num-workers",
    "--utt-dim-weights",
    "--loss-w-phone",
    "--loss-w-word",
    "--loss-w-utt",
    "--batch-size",
    "--loss-w-rank",
    "--loss-w-oracle-phone",
    "--loss-w-oracle-word",
    "--loss-w-teacher-score",
    "--loss-w-prefix-kd",
    "--loss-w-oracle-utt-prefix",
    "--loss-w-oracle-utt-final",
    "--loss-w-phone-stability",
    "--loss-w-word-stability",
    "--loss-w-utt-stability",
    "--utt-pooling-head",
    "--fusion-mode",
    "--embed-dim",
    "--depth",
    "--heads",
    "--gru-dim",
}
args = sys.argv[1:]
out = []
i = 0
while i < len(args):
    item = args[i]
    if item in replace_with_value and i + 1 < len(args):
        value = args[i + 1]
        if item in out:
            pos = out.index(item)
            out[pos + 1] = value
        else:
            out.extend([item, value])
        i += 2
    else:
        if item.startswith("--") and item not in out:
            out.append(item)
        elif not item.startswith("--"):
            out.append(item)
        i += 1
print("\n".join(out))
PY
}

command_for_exp() {
  local exp="$1"
  local raw_args=()
  mapfile -t raw_args < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw_args[@]}")
  printf '%q ' env CUDA_VISIBLE_DEVICES='${GPU}' "${PY_TRAIN}" src/train_streaming_pcn.py --exp-dir "${ORACLE_RUN_ROOT}/${exp}" "${args[@]}"
  printf '\n'
}

validate_oracle_npz() {
  "${PY_TRAIN}" - "$ORACLE_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
summary = {}
for split in ["train", "val", "test"]:
    path = root / f"{split}_chunks.npz"
    if not path.exists():
        raise SystemExit(f"missing oracle npz: {path}")
    arr = np.load(path)
    utt_rows = int((arr["oracle_utt_mask"] > 0).sum())
    word_slots = int(np.any(arr["oracle_word_dim_mask"] > 0, axis=-1).sum())
    phone_slots = int((arr["oracle_phone_mask"] > 0).sum())
    comp_sum = float(arr["oracle_utt_dim_mask"][:, 1].sum())
    if "oracle_prefix_utt_dim_mask" in arr.files:
        comp_sum += float(arr["oracle_prefix_utt_dim_mask"][:, 1].sum())
    if "oracle_final_utt_dim_mask" in arr.files:
        comp_sum += float(arr["oracle_final_utt_dim_mask"][:, 1].sum())
    if utt_rows <= 0:
        raise SystemExit(f"{split}: oracle_utt_rows is 0")
    if comp_sum != 0.0:
        raise SystemExit(f"{split}: oracle completeness mask sum must be 0, got {comp_sum}")
    summary[split] = {
        "oracle_utt_rows": utt_rows,
        "oracle_word_slots": word_slots,
        "oracle_phone_slots": phone_slots,
        "oracle_completeness_mask_sum": comp_sum,
    }
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

inject_oracle_data() {
  local data_dir="$1"
  "${PY_TRAIN}" scripts/local/inject_oracle_gopt_teacher_pcn.py \
    --data-dir "$data_dir" \
    --oracle-jsonl "$ORACLE_JSONL" \
    --output-dir "$ORACLE_DATA_DIR" \
    --splits train,val,test \
    --drop-completeness \
    --overwrite
}

prepare() {
  mkdir -p "$EXP_ROOT"
  local data_dir
  data_dir="$(pcn_data_dir)"
  if [[ ! -s "$ORACLE_JSONL" || "$FORCE" == "1" ]]; then
    local gpu
    gpu="$(wait_for_idle_gpu)"
    echo "[prepare] using GPU${gpu} for GT-time alignment; GPU${BAD_GPU} skipped"
    CUDA_VISIBLE_DEVICES="$gpu" "${PY_ORACLE}" scripts/local/build_full_oracle_gopt_teacher_pcn.py \
      --pcn-data-dir "$data_dir" \
      --output-jsonl "$ORACLE_JSONL" \
      --splits train,val,test \
      --oracle-source auto \
      --device cuda \
      --align-device cuda:0 \
      --overwrite
  else
    echo "[prepare] oracle jsonl exists: $ORACLE_JSONL"
  fi
  if [[ ! -d "$ORACLE_DATA_DIR" || "$FORCE" == "1" ]]; then
    inject_oracle_data "$data_dir"
  else
    echo "[prepare] oracle data dir exists: $ORACLE_DATA_DIR"
  fi
  local summary_tmp="${EXP_ROOT}/oracle_prepare_summary.json.tmp"
  if ! validate_oracle_npz > "$summary_tmp"; then
    echo "[prepare] oracle data validation failed; rebuilding from $ORACLE_JSONL" >&2
    inject_oracle_data "$data_dir"
    validate_oracle_npz > "$summary_tmp"
  fi
  cat "$summary_tmp" | tee "${EXP_ROOT}/oracle_prepare_summary.json"
  rm -f "$summary_tmp"
}

print_commands() {
  echo "EXP_ROOT=$EXP_ROOT"
  echo "PCN_DATA_DIR=$(pcn_data_dir)"
  echo "ORACLE_JSONL=$ORACLE_JSONL"
  echo "ORACLE_DATA_DIR=$ORACLE_DATA_DIR"
  while IFS= read -r exp; do
    echo "[$exp]"
    command_for_exp "$exp"
  done < <(all_exps)
}

is_done() {
  local exp_dir="$1"
  [[ -f "$exp_dir/models/best_audio_model.pth" || -f "$exp_dir/last_checkpoint.pt" ]]
}

start_exp() {
  local exp="$1"
  mkdir -p "$LOG_DIR" "$PID_DIR" "$ORACLE_RUN_ROOT"
  local exp_dir="${ORACLE_RUN_ROOT}/${exp}"
  local pid_file="${PID_DIR}/${exp}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already running pid=$(cat "$pid_file")"
    return 0
  fi
  if is_done "$exp_dir" && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already has checkpoint/model under $exp_dir"
    return 0
  fi
  if [[ ! -d "$ORACLE_DATA_DIR" ]]; then
    echo "missing ORACLE_DATA_DIR=$ORACLE_DATA_DIR; run prepare first" >&2
    return 1
  fi
  local gpu
  gpu="$(wait_for_idle_gpu)"
  local raw_args=()
  mapfile -t raw_args < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw_args[@]}")
  echo "[run] $exp on GPU${gpu}; GPU${BAD_GPU} skipped"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "${PY_TRAIN}" src/train_streaming_pcn.py \
    --exp-dir "$exp_dir" \
    "${args[@]}" > "${LOG_DIR}/${exp}.log" 2>&1 &
  echo $! > "$pid_file"
  echo "[run] pid=$(cat "$pid_file") log=${LOG_DIR}/${exp}.log"
}

status() {
  echo "EXP_ROOT=$EXP_ROOT"
  echo "GPU policy: idle means no compute process, memory.used < 1000 MiB, utilization.gpu < 10; GPU${BAD_GPU}=skipped"
  gpu_rows | while IFS=',' read -r idx mem util; do
    idx="$(echo "$idx" | xargs)"
    marker=""
    [[ "$idx" == "$BAD_GPU" ]] && marker=" skipped"
    echo "gpu=${idx} mem=$(echo "$mem" | xargs)MiB util=$(echo "$util" | xargs)%${marker}"
  done
  while IFS= read -r exp; do
    local exp_dir="${ORACLE_RUN_ROOT}/${exp}"
    local pid_file="${PID_DIR}/${exp}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "$exp running pid=$(cat "$pid_file")"
    elif is_done "$exp_dir"; then
      echo "$exp done"
    else
      echo "$exp missing"
    fi
  done < <(all_exps)
}

summarize() {
  mkdir -p "$ORACLE_RUN_ROOT"
  local done_exps=()
  while IFS= read -r exp; do
    if [[ -f "${ORACLE_RUN_ROOT}/${exp}/models/best_audio_model.pth" ]]; then
      "${PY_TRAIN}" scripts/local/eval_pcn_coverage_pcc.py \
        --exp-root "$ORACLE_RUN_ROOT" \
        --experiments "$exp" \
        --device cpu \
        --output-csv "${ORACLE_RUN_ROOT}/${exp}/coverage_pcc_phone_word_sentence.csv"
      done_exps+=("$exp")
    fi
  done < <(all_exps)
  local out="${ORACLE_RUN_ROOT}/oracle_coverage_pcc_all_models.csv"
  : > "$out"
  local wrote=0
  local baseline="${EXP_ROOT}/coverage_pcc_phone_word_sentence_all_models.csv"
  if [[ -f "$baseline" ]]; then
    cat "$baseline" >> "$out"
    wrote=1
  fi
  for exp in "${done_exps[@]}"; do
    local csv="${ORACLE_RUN_ROOT}/${exp}/coverage_pcc_phone_word_sentence.csv"
    if [[ "$wrote" == "0" ]]; then
      cat "$csv" >> "$out"
      wrote=1
    else
      tail -n +2 "$csv" >> "$out"
    fi
  done
  echo "[summarize] wrote $out"
}

case "$ACTION" in
  prepare)
    prepare
    ;;
  print)
    print_commands
    ;;
  run)
    if [[ -z "$TARGET" ]]; then
      echo "usage: $0 run EXPERIMENT" >&2
      exit 1
    fi
    start_exp "$TARGET"
    ;;
  run_all)
    while IFS= read -r exp; do
      start_exp "$exp"
    done < <(all_exps)
    ;;
  status)
    status
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 {prepare|print|run EXP|run_all|status|summarize}" >&2
    exit 1
    ;;
esac
