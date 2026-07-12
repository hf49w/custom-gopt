#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/DATA_2/guest/custom-gopt}"
EXP_ROOT="${EXP_ROOT:-${REPO}/exp/pcn_extra_20260704_2130}"
ORACLE_DATA_DIR="${ORACLE_DATA_DIR:-${EXP_ROOT}/data_streaming_pcn_oracle_gopt_full}"
SLOTPROSODY_DATA_DIR="${SLOTPROSODY_DATA_DIR:-${EXP_ROOT}/data_streaming_pcn_oracle_gopt_full_slotprosody}"
RUN_ROOT="${RUN_ROOT:-${EXP_ROOT}/stress_runs}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
PID_DIR="${PID_DIR:-${EXP_ROOT}/pids}"
PY_TRAIN="${PY_TRAIN:-${REPO}/.conda_env/bin/python}"
BAD_GPU=3
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"
ACTION="${1:-status}"
TARGET="${2:-}"

cd "$REPO"

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

all_exps() {
  printf '%s\n' \
    H_stress_weighted_G \
    I_stress_corr_G \
    J_stress_detached_branch \
    K_slot_prosody_stress \
    L_stress_gradscale_voiced \
    M_stress_scalar_gate_capacity64
}

base_args() {
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
  printf '%s\n' \
    --word-dim-weights 1,3,1 \
    --teacher-word-dim-weights 1,2,1 \
    --oracle-word-dim-weights 1,5,1
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

exp_args() {
  local exp="$1"
  case "$exp" in
    H_stress_weighted_G)
      base_args "$ORACLE_DATA_DIR"
      stress_weight_args
      ;;
    I_stress_corr_G)
      base_args "$ORACLE_DATA_DIR"
      stress_corr_args
      ;;
    J_stress_detached_branch)
      base_args "$ORACLE_DATA_DIR"
      stress_detached_args
      ;;
    K_slot_prosody_stress)
      base_args "$SLOTPROSODY_DATA_DIR"
      stress_detached_args
      ;;
    L_stress_gradscale_voiced)
      base_args "$SLOTPROSODY_DATA_DIR"
      stress_corr_args
      printf '%s\n' --stress-branch gradscale --stress-grad-scale 0.2 --stress-loss-mask voiced_or_vowel
      ;;
    M_stress_scalar_gate_capacity64)
      base_args "$SLOTPROSODY_DATA_DIR"
      stress_detached_args
      printf '%s\n' --fusion-mode scalar_gate
      ;;
    *)
      echo "unknown experiment: $exp" >&2
      return 1
      ;;
  esac
}

dedupe_args() {
  "$PY_TRAIN" - "$@" <<'PY'
import sys
with_value = {
    "--data-dir", "--n-epochs", "--batch-size", "--num-workers", "--utt-dim-weights",
    "--loss-w-phone", "--loss-w-word", "--loss-w-utt", "--loss-w-teacher-score",
    "--loss-w-prefix-kd", "--loss-w-rank", "--loss-w-oracle-phone", "--loss-w-oracle-word",
    "--loss-w-oracle-utt-prefix", "--loss-w-oracle-utt-final", "--loss-w-phone-stability",
    "--loss-w-word-stability", "--loss-w-utt-stability", "--utt-pooling-head", "--fusion-mode",
    "--embed-dim", "--depth", "--heads", "--gru-dim", "--word-dim-weights",
    "--teacher-word-dim-weights", "--oracle-word-dim-weights", "--loss-w-stress-pearson",
    "--loss-w-oracle-stress-pearson", "--loss-w-teacher-stress-pearson", "--loss-w-stress-rank",
    "--loss-w-oracle-stress-rank", "--stress-branch", "--stress-grad-scale",
    "--stress-loss-mask", "--stress-voiced-threshold", "--stress-rank-margin",
    "--stress-rank-max-items",
}
args = sys.argv[1:]
out = []
i = 0
while i < len(args):
    item = args[i]
    if item in with_value and i + 1 < len(args):
        value = args[i + 1]
        if item in out:
            out[out.index(item) + 1] = value
        else:
            out.extend([item, value])
        i += 2
    else:
        if item.startswith("--") and item in out:
            i += 1
            continue
        out.append(item)
        i += 1
print("\n".join(out))
PY
}

command_for_exp() {
  local exp="$1"
  local raw_args=()
  local args=()
  mapfile -t raw_args < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw_args[@]}")
  printf '%q ' env CUDA_VISIBLE_DEVICES='${GPU}' "$PY_TRAIN" src/train_streaming_pcn.py --exp-dir "${RUN_ROOT}/${exp}" "${args[@]}"
  printf '\n'
}

validate_oracle_data() {
  "$PY_TRAIN" - "$ORACLE_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
summary = {}
for split in ["train", "val", "test"]:
    path = root / f"{split}_chunks.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with np.load(path) as arr:
        comp_sum = float(arr["oracle_utt_dim_mask"][:, 1].sum()) if "oracle_utt_dim_mask" in arr.files else 0.0
        if "oracle_prefix_utt_dim_mask" in arr.files:
            comp_sum += float(arr["oracle_prefix_utt_dim_mask"][:, 1].sum())
        if "oracle_final_utt_dim_mask" in arr.files:
            comp_sum += float(arr["oracle_final_utt_dim_mask"][:, 1].sum())
        summary[split] = {
            "oracle_utt_rows": int((arr["oracle_utt_mask"] > 0).sum()) if "oracle_utt_mask" in arr.files else 0,
            "oracle_completeness_mask_sum": comp_sum,
        }
        if comp_sum != 0.0:
            raise SystemExit(f"{split}: oracle completeness mask sum must be 0, got {comp_sum}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

validate_slotprosody_data() {
  "$PY_TRAIN" - "$SLOTPROSODY_DATA_DIR" <<'PY'
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
metadata_path = root / "metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
augment_summary = metadata.get("slot_prosody_augment_summary", {})
summary = {}
for split in ["train", "val", "test"]:
    path = root / f"{split}_chunks.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    missing = [f"{name}.npy" for name in ["slot_prosody", "slot_is_vowel", "slot_voiced_ratio"] if f"{name}.npy" not in names]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    with np.load(path) as arr:
        vowel_count = int(arr["slot_is_vowel"].sum())
        voiced_count = int((arr["slot_voiced_ratio"] > 0.3).sum())
    cur = dict(augment_summary.get(split, {}))
    cur.setdefault("slot_prosody_nonzero_ratio", "")
    cur["vowel_slot_count"] = vowel_count
    cur["voiced_slot_count"] = voiced_count
    summary[split] = cur
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

prepare() {
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$PID_DIR"
  if [[ ! -d "$ORACLE_DATA_DIR" ]]; then
    echo "missing ORACLE_DATA_DIR=$ORACLE_DATA_DIR" >&2
    exit 1
  fi
  validate_oracle_data | tee "${EXP_ROOT}/stress_oracle_data_validation.json"
  if [[ ! -d "$SLOTPROSODY_DATA_DIR" || "$FORCE" == "1" ]]; then
    "$PY_TRAIN" scripts/local/augment_slot_prosody_pcn.py \
      --data-dir "$ORACLE_DATA_DIR" \
      --output-dir "$SLOTPROSODY_DATA_DIR" \
      --splits train,val,test \
      --overwrite
  else
    echo "[prepare] slotprosody data dir exists: $SLOTPROSODY_DATA_DIR"
  fi
  if ! validate_slotprosody_data > "${EXP_ROOT}/stress_slotprosody_validation.json.tmp"; then
    echo "[prepare] slotprosody validation failed; rebuilding" >&2
    "$PY_TRAIN" scripts/local/augment_slot_prosody_pcn.py \
      --data-dir "$ORACLE_DATA_DIR" \
      --output-dir "$SLOTPROSODY_DATA_DIR" \
      --splits train,val,test \
      --overwrite
    validate_slotprosody_data > "${EXP_ROOT}/stress_slotprosody_validation.json.tmp"
  fi
  cat "${EXP_ROOT}/stress_slotprosody_validation.json.tmp" | tee "${EXP_ROOT}/stress_slotprosody_validation.json"
  rm -f "${EXP_ROOT}/stress_slotprosody_validation.json.tmp"
}

print_commands() {
  echo "EXP_ROOT=$EXP_ROOT"
  echo "ORACLE_DATA_DIR=$ORACLE_DATA_DIR"
  echo "SLOTPROSODY_DATA_DIR=$SLOTPROSODY_DATA_DIR"
  echo "RUN_ROOT=$RUN_ROOT"
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
  mkdir -p "$RUN_ROOT" "$LOG_DIR" "$PID_DIR"
  local exp_dir="${RUN_ROOT}/${exp}"
  local pid_file="${PID_DIR}/${exp}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already running pid=$(cat "$pid_file")"
    return 0
  fi
  if is_done "$exp_dir" && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already has checkpoint/model under $exp_dir"
    return 0
  fi
  if [[ -d "$exp_dir" && -n "$(find "$exp_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "$FORCE" != "1" ]]; then
    echo "[skip] $exp dir exists but is incomplete; set FORCE=1 to reuse it: $exp_dir" >&2
    return 1
  fi
  local args=()
  local raw_args=()
  mapfile -t raw_args < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw_args[@]}")
  local gpu
  gpu="$(wait_for_idle_gpu)"
  echo "[run] $exp on GPU${gpu}; GPU${BAD_GPU} skipped"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY_TRAIN" src/train_streaming_pcn.py \
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
    local exp_dir="${RUN_ROOT}/${exp}"
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
  mkdir -p "$RUN_ROOT"
  local done_exps=()
  while IFS= read -r exp; do
    if [[ -f "${RUN_ROOT}/${exp}/models/best_audio_model.pth" ]]; then
      "$PY_TRAIN" scripts/local/eval_pcn_coverage_pcc.py \
        --exp-root "$RUN_ROOT" \
        --experiments "$exp" \
        --device cpu \
        --output-csv "${RUN_ROOT}/${exp}/coverage_pcc_phone_word_sentence.csv"
      done_exps+=("$exp")
    fi
  done < <(all_exps)
  "$PY_TRAIN" - "$RUN_ROOT" "$EXP_ROOT" "${done_exps[@]}" <<'PY'
import csv
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
done = sys.argv[3:]
sources = [
    exp_root / "coverage_pcc_phone_word_sentence_all_models.csv",
    exp_root / "oracle_runs" / "oracle_coverage_pcc_all_models.csv",
]
sources += [run_root / exp / "coverage_pcc_phone_word_sentence.csv" for exp in done]
out = run_root / "stress_coverage_pcc_all_models.csv"
rows = []
fieldnames = None
for src in sources:
    if not src.exists():
        continue
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(dict(row) for row in reader)
if fieldnames is None:
    raise SystemExit("no coverage CSV sources found")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[summarize] wrote {out}")

wanted = {
    ("sentence", "total"): "sent_total",
    ("word", "accuracy"): "word_acc",
    ("word", "stress"): "word_stress",
    ("word", "total"): "word_total",
    ("phone", "phone"): "phone",
}
by_model = {}
for row in rows:
    if str(row.get("coverage")) != "100":
        continue
    key = (row.get("level"), row.get("metric"))
    if key not in wanted:
        continue
    model = row.get("experiment", row.get("model", ""))
    by_model.setdefault(model, {})[wanted[key]] = row.get("pcc", "")
print("coverage=100:")
print("model,sent_total,word_acc,word_stress,word_total,phone,stress_gt_multipa,phone_ge_0.32,word_total_ge_0.40,sent_total_ge_0.70")
for model in sorted(by_model):
    vals = by_model[model]
    def f(name):
        try:
            return float(vals.get(name, "nan"))
        except ValueError:
            return float("nan")
    stress = f("word_stress")
    phone = f("phone")
    word_total = f("word_total")
    sent_total = f("sent_total")
    print(
        ",".join(
            [
                model,
                str(vals.get("sent_total", "")),
                str(vals.get("word_acc", "")),
                str(vals.get("word_stress", "")),
                str(vals.get("word_total", "")),
                str(vals.get("phone", "")),
                str(stress > 0.045436),
                str(phone >= 0.32),
                str(word_total >= 0.40),
                str(sent_total >= 0.70),
            ]
        )
    )
for required in [
    "A_loss_dimmask",
    "B_relaxed_softlabel",
    "G_oracle_capacity64",
    "F_oracle_vector_gate",
    "MultiPA",
    "gopt_original",
    "gopt_open_base",
    "gopt_open_medium",
]:
    if required not in by_model:
        print(f"[summarize][warning] missing model in merged coverage: {required}")
PY
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
