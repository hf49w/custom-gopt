#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
MULTIPA_ROOT="${MULTIPA_ROOT:-/DATA_2/MultiPA}"
BAD_GPU="${BAD_GPU:-3}"
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"

BASE_DATA_DIR="${BASE_DATA_DIR:-${ROOT}/data/streaming_pcn_gopt_v2_stateful}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/exp/pcn_extra_correct_multipa_20260710}"
TEACHER_DIR="${TEACHER_DIR:-${EXP_ROOT}/teacher_multipa_correct}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${TEACHER_DIR}/train_val_manifest.jsonl}"
TEACHER_JSONL="${TEACHER_JSONL:-${TEACHER_DIR}/multipa_train_val.jsonl}"
CORRECT_DATA_DIR="${CORRECT_DATA_DIR:-${EXP_ROOT}/data_streaming_pcn_correct_multipa}"
ORACLE_JSONL="${ORACLE_JSONL:-${EXP_ROOT}/oracle_gopt_closed_prefix_gt_time_all_splits.jsonl}"
ORACLE_DATA_DIR="${ORACLE_DATA_DIR:-${EXP_ROOT}/data_streaming_pcn_correct_multipa_oracle_gopt_full}"
SLOTPROSODY_DATA_DIR="${SLOTPROSODY_DATA_DIR:-${EXP_ROOT}/data_streaming_pcn_correct_multipa_oracle_gopt_full_slotprosody}"
RUN_ROOT="${RUN_ROOT:-${EXP_ROOT}/runs}"
ORACLE_RUN_ROOT="${ORACLE_RUN_ROOT:-${EXP_ROOT}/oracle_runs}"
STRESS_RUN_ROOT="${STRESS_RUN_ROOT:-${EXP_ROOT}/stress_runs}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
PID_DIR="${PID_DIR:-${EXP_ROOT}/pids}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/corrected_queue.log}"
QUEUE_PID="${QUEUE_PID:-${PID_DIR}/corrected_queue.pid}"

PY="${PY:-${ROOT}/.conda_env/bin/python}"
MULTIPA_PY="${MULTIPA_PY:-${ROOT}/.multipa_env/bin/python}"
PY_ORACLE="${PY_ORACLE:-${ROOT}/.multipa_env/bin/python}"
ALIGNER="${ALIGNER:-${ROOT}/server_assets/models/charsiu-en_w2v2_fc_10ms}"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$PID_DIR"

die() {
  echo "$*" >&2
  exit 1
}

assert_correct_multipa() {
  local resolved
  resolved="$(readlink -f "$MULTIPA_ROOT")"
  [[ "$resolved" != "/DATA_2/guest/MultiPA_pic" ]] || die "/DATA_2/guest/MultiPA_pic is forbidden; use /DATA_2/MultiPA."
  [[ "$resolved" == "/DATA_2/MultiPA" ]] || die "MULTIPA_ROOT must resolve to /DATA_2/MultiPA, got $resolved"
  for path in \
    "$MULTIPA_ROOT/eval_multipa_prefix.py" \
    "$MULTIPA_ROOT/fairseq_hubert/hubert_base_ls960.pt" \
    "$MULTIPA_ROOT/fairseq_roberta" \
    "$MULTIPA_ROOT/model_assessment"; do
    [[ -e "$path" ]] || die "Missing required MultiPA asset: $path"
  done
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$gpu" '$1 == gpu && $2 ~ /^[0-9]+$/ && $3 == "C" { found = 1 } END { exit(found ? 0 : 1) }'
}

gpu_is_idle() {
  local gpu="$1"
  [[ "$gpu" != "$BAD_GPU" ]] || return 1
  local line mem util
  line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' ')"
  [[ -n "$line" ]] || return 1
  mem="${line%,*}"
  util="${line#*,}"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
  [[ "$mem" -lt 1000 && "$util" -lt 10 ]] || return 1
  if gpu_has_compute_process "$gpu"; then
    return 1
  fi
  return 0
}

idle_gpus_now() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read -r gpu; do
    gpu="$(echo "$gpu" | tr -d ' ')"
    if gpu_is_idle "$gpu"; then
      echo "$gpu"
    fi
  done
}

wait_for_idle_gpu() {
  local gpu
  while true; do
    while read -r gpu; do
      [[ -n "$gpu" ]] || continue
      echo "$gpu"
      return 0
    done < <(idle_gpus_now)
    echo "[wait] no idle GPU; GPU${BAD_GPU} skipped; sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
  done
}

print_gpu_status() {
  echo "GPU policy: idle means no compute process, memory.used < 1000 MiB, utilization.gpu < 10; GPU${BAD_GPU}=skipped"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r gpu mem util; do
    gpu="$(echo "$gpu" | tr -d ' ')"
    mem="$(echo "$mem" | tr -d ' ')"
    util="$(echo "$util" | tr -d ' ')"
    if [[ "$gpu" == "$BAD_GPU" ]]; then
      echo "gpu=${gpu} mem=${mem}MiB util=${util}% skipped"
    else
      echo "gpu=${gpu} mem=${mem}MiB util=${util}%"
    fi
  done
}

teacher_env() {
  export XDG_CACHE_HOME=/DATA_2/MultiPA/.cache
  export CHARSU_TOKENIZER_EN_CMU="${ROOT}/server_assets/src/charsiu_repo/local"
  export CHARSIU_TOKENIZER_EN_CMU="${ROOT}/server_assets/src/charsiu_repo/local"
  export HF_HOME="${ROOT}/server_assets/hf_home"
  export TRANSFORMERS_CACHE="${ROOT}/server_assets/hf_home/transformers"
  export FAIRSEQ_GPT2_ENCODER_JSON="${ROOT}/server_assets/fairseq_bpe/encoder.json"
  export FAIRSEQ_GPT2_VOCAB_BPE="${ROOT}/server_assets/fairseq_bpe/vocab.bpe"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
}

build_teacher_manifest() {
  mkdir -p "$TEACHER_DIR"
  cat "${BASE_DATA_DIR}/train_manifest.jsonl" "${BASE_DATA_DIR}/val_manifest.jsonl" > "$TEACHER_MANIFEST"
}

prepare_teacher() {
  assert_correct_multipa
  build_teacher_manifest
  if [[ -s "$TEACHER_JSONL" && "$FORCE" != "1" ]]; then
    echo "[teacher] exists: $TEACHER_JSONL"
    return 0
  fi
  rm -f "${TEACHER_DIR}"/multipa_train_val.shard_*.jsonl "${TEACHER_DIR}"/multipa_train_val.shard_*.log "$TEACHER_JSONL"
  mapfile -t gpus < <(idle_gpus_now)
  while [[ "${#gpus[@]}" -eq 0 ]]; do
    echo "[teacher] no idle GPU; GPU${BAD_GPU} skipped; sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
    mapfile -t gpus < <(idle_gpus_now)
  done
  local num_shards="${#gpus[@]}"
  echo "[teacher] generating with /DATA_2/MultiPA shards=${num_shards} gpus=${gpus[*]} GPU${BAD_GPU}=skipped"
  teacher_env
  mkdir -p "${TEACHER_DIR}/shards"
  local pids=()
  for shard_index in "${!gpus[@]}"; do
    local gpu="${gpus[$shard_index]}"
    local shard_manifest="${TEACHER_DIR}/shards/train_val_manifest.shard_${shard_index}.jsonl"
    local shard_jsonl="${TEACHER_DIR}/multipa_train_val.shard_${shard_index}.jsonl"
    local shard_log="${TEACHER_DIR}/multipa_train_val.shard_${shard_index}.log"
    "$PY" scripts/shard_jsonl.py \
      --input-jsonl "$TEACHER_MANIFEST" \
      --output-jsonl "$shard_manifest" \
      --num-shards "$num_shards" \
      --shard-index "$shard_index"
    (
      cd "$MULTIPA_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" "$MULTIPA_PY" eval_multipa_prefix.py \
        --prefix-manifest "$shard_manifest" \
        --output-jsonl "$shard_jsonl" \
        --resume \
        --fairseq-base-model "$MULTIPA_ROOT/fairseq_hubert/hubert_base_ls960.pt" \
        --fairseq-roberta "$MULTIPA_ROOT/fairseq_roberta" \
        --ckptdir "$MULTIPA_ROOT/model_assessment" \
        --aligner-model "$ALIGNER" \
        --whisper-sentence-model /DATA_2/guest/custom-whisper/data/models/whisper/medium.en.pt \
        --whisper-word-model /DATA_2/guest/custom-whisper/data/models/whisper/base.en.pt \
        --local-model-cache "$ROOT/server_assets/multipa_model_cache"
    ) > "$shard_log" 2>&1 &
    pids+=("$!")
    echo "${pids[-1]}" > "${PID_DIR}/teacher_shard_${shard_index}.pid"
    echo "[teacher] shard=${shard_index}/${num_shards} gpu=${gpu} pid=${pids[-1]} log=${shard_log}"
    sleep 10
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "[teacher] one or more shards failed"
  : > "$TEACHER_JSONL"
  for shard_index in "${!gpus[@]}"; do
    cat "${TEACHER_DIR}/multipa_train_val.shard_${shard_index}.jsonl" >> "$TEACHER_JSONL"
  done
  "$PY" - "$TEACHER_JSONL" <<'PY'
import json, sys
from collections import Counter
counts = Counter()
with open(sys.argv[1], encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            counts[json.loads(line).get('status', 'unknown')] += 1
print('[teacher-summary]', dict(counts), flush=True)
if counts['ok'] == 0:
    raise SystemExit('teacher export produced no valid rows')
PY
}

inject_multipa_data() {
  if [[ -d "$CORRECT_DATA_DIR" && "$FORCE" != "1" ]]; then
    echo "[inject] exists: $CORRECT_DATA_DIR"
  else
    "$PY" scripts/local/inject_multipa_teacher_pcn.py \
      --data-dir "$BASE_DATA_DIR" \
      --teacher-jsonl "$TEACHER_JSONL" \
      --output-dir "$CORRECT_DATA_DIR" \
      --splits train,val,test \
      --overwrite
  fi
  "$PY" - "$CORRECT_DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
summary = {}
for split in ["train", "val", "test"]:
    arr = np.load(root / f"{split}_chunks.npz")
    teacher_rows = int((arr["teacher_utt_mask"] > 0).sum()) if "teacher_utt_mask" in arr.files else 0
    comp = float(arr["teacher_utt_dim_mask"][:, 1].sum()) if "teacher_utt_dim_mask" in arr.files else 0.0
    state_rows = int((arr["teacher_state_mask"] > 0).sum()) if "teacher_state_mask" in arr.files else 0
    summary[split] = {
        "teacher_utt_rows": teacher_rows,
        "teacher_completeness_mask_sum": comp,
        "teacher_state_rows": state_rows,
    }
    if split in {"train", "val"} and teacher_rows <= 0:
        raise SystemExit(f"{split}: no teacher rows")
    if comp != 0.0:
        raise SystemExit(f"{split}: completeness teacher mask must be 0, got {comp}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

prepare_oracle() {
  if [[ ! -s "$ORACLE_JSONL" || "$FORCE" == "1" ]]; then
    local gpu
    gpu="$(wait_for_idle_gpu)"
    echo "[oracle] build full oracle teacher on GPU${gpu}; GPU${BAD_GPU} skipped"
    MULTIPA_REPO_ROOT="$MULTIPA_ROOT" CUDA_VISIBLE_DEVICES="$gpu" "$PY_ORACLE" scripts/local/build_full_oracle_gopt_teacher_pcn.py \
      --pcn-data-dir "$CORRECT_DATA_DIR" \
      --output-jsonl "$ORACLE_JSONL" \
      --splits train,val,test \
      --oracle-source auto \
      --multipa-repo-root "$MULTIPA_ROOT" \
      --device cuda \
      --align-device cuda:0 \
      --overwrite
  else
    echo "[oracle] exists: $ORACLE_JSONL"
  fi
  if [[ ! -d "$ORACLE_DATA_DIR" || "$FORCE" == "1" ]]; then
    "$PY" scripts/local/inject_oracle_gopt_teacher_pcn.py \
      --data-dir "$CORRECT_DATA_DIR" \
      --oracle-jsonl "$ORACLE_JSONL" \
      --output-dir "$ORACLE_DATA_DIR" \
      --splits train,val,test \
      --drop-completeness \
      --overwrite
  else
    echo "[oracle] data exists: $ORACLE_DATA_DIR"
  fi
  "$PY" - "$ORACLE_DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
summary = {}
for split in ["train", "val", "test"]:
    arr = np.load(root / f"{split}_chunks.npz")
    rows = int((arr["oracle_utt_mask"] > 0).sum())
    comp = float(arr["oracle_utt_dim_mask"][:, 1].sum())
    if "oracle_prefix_utt_dim_mask" in arr.files:
        comp += float(arr["oracle_prefix_utt_dim_mask"][:, 1].sum())
    if "oracle_final_utt_dim_mask" in arr.files:
        comp += float(arr["oracle_final_utt_dim_mask"][:, 1].sum())
    summary[split] = {"oracle_utt_rows": rows, "oracle_completeness_mask_sum": comp}
    if rows <= 0:
        raise SystemExit(f"{split}: no oracle rows")
    if comp != 0.0:
        raise SystemExit(f"{split}: oracle completeness mask must be 0, got {comp}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

prepare_slotprosody() {
  if [[ ! -d "$SLOTPROSODY_DATA_DIR" || "$FORCE" == "1" ]]; then
    "$PY" scripts/local/augment_slot_prosody_pcn.py \
      --data-dir "$ORACLE_DATA_DIR" \
      --output-dir "$SLOTPROSODY_DATA_DIR" \
      --splits train,val,test \
      --overwrite
  else
    echo "[slotprosody] exists: $SLOTPROSODY_DATA_DIR"
  fi
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

is_done() {
  local exp_dir="$1"
  [[ -f "${exp_dir}/test_metrics.json" ]]
}

start_exp() {
  local exp="$1"
  local exp_dir
  exp_dir="$(exp_dir_for "$exp")"
  local pid_file="${PID_DIR}/${exp}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already running pid=$(cat "$pid_file")"
    return 0
  fi
  if is_done "$exp_dir" && [[ "$FORCE" != "1" ]]; then
    echo "[skip] $exp already done"
    return 0
  fi
  mkdir -p "$(dirname "$exp_dir")" "$LOG_DIR" "$PID_DIR"
  mapfile -t raw < <(exp_args "$exp")
  mapfile -t args < <(dedupe_args "${raw[@]}")
  local gpu
  gpu="$(wait_for_idle_gpu)"
  echo "[run] $exp on GPU${gpu}; GPU${BAD_GPU} skipped"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/train_streaming_pcn.py --exp-dir "$exp_dir" "${args[@]}"
  ) > "${LOG_DIR}/${exp}.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "$gpu" > "${PID_DIR}/${exp}.gpu"
  echo "[run] pid=$! log=${LOG_DIR}/${exp}.log"
  sleep 30
}

run_experiment_queue() {
  for exp in $(all_exps); do
    start_exp "$exp"
  done
}

prepare_all() {
  assert_correct_multipa
  prepare_teacher
  inject_multipa_data
  prepare_oracle
  prepare_slotprosody
}

pipeline() {
  echo "[queue] start $(date '+%F %T')"
  echo "[queue] EXP_ROOT=$EXP_ROOT"
  prepare_all
  run_experiment_queue
  echo "[queue] all experiments launched $(date '+%F %T')"
}

start_queue() {
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "[queue] already running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
    return 0
  fi
  mkdir -p "$LOG_DIR" "$PID_DIR"
  local script
  script="$(readlink -f "$0")"
  nohup "$script" pipeline >> "$QUEUE_LOG" 2>&1 &
  echo "$!" > "$QUEUE_PID"
  echo "[queue] pid=$! log=$QUEUE_LOG"
}

stop_existing_old_runs() {
  local old_root="${OLD_EXP_ROOT:-${ROOT}/exp/pcn_extra_20260704_2130}"
  local old_pid_dir="${old_root}/pids"
  for exp in H_stress_weighted_G I_stress_corr_G J_stress_detached_branch K_slot_prosody_stress L_stress_gradscale_voiced M_stress_scalar_gate_capacity64; do
    local pf="${old_pid_dir}/${exp}.pid"
    [[ -f "$pf" ]] || continue
    local pid
    pid="$(cat "$pf")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      local cmd
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      case "$cmd" in
        *"src/train_streaming_pcn.py"*"stress_runs/${exp}"*)
          echo "[stop] old $exp pid=$pid"
          kill "$pid" || true
          ;;
      esac
    fi
    rm -f "$pf"
  done
}

print_plan() {
  echo "EXP_ROOT=$EXP_ROOT"
  echo "MULTIPA_ROOT=$MULTIPA_ROOT"
  echo "BASE_DATA_DIR=$BASE_DATA_DIR"
  echo "TEACHER_JSONL=$TEACHER_JSONL"
  echo "CORRECT_DATA_DIR=$CORRECT_DATA_DIR"
  echo "ORACLE_DATA_DIR=$ORACLE_DATA_DIR"
  echo "SLOTPROSODY_DATA_DIR=$SLOTPROSODY_DATA_DIR"
  for exp in $(all_exps); do
    echo "[$exp] exp_dir=$(exp_dir_for "$exp")"
  done
}

status() {
  echo "EXP_ROOT=$EXP_ROOT"
  echo "MULTIPA_ROOT=$MULTIPA_ROOT"
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
  else
    echo "queue not-running log=$QUEUE_LOG"
  fi
  print_gpu_status
  echo "teacher_jsonl=$([[ -s "$TEACHER_JSONL" ]] && echo present || echo missing) $TEACHER_JSONL"
  for pf in "${PID_DIR}"/teacher_shard_*.pid; do
    [[ -f "$pf" ]] || continue
    local pid
    pid="$(cat "$pf")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "teacher $(basename "$pf" .pid) running pid=$pid"
    fi
  done
  for exp in $(all_exps); do
    local exp_dir pid_file state
    exp_dir="$(exp_dir_for "$exp")"
    pid_file="${PID_DIR}/${exp}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      state="running pid=$(cat "$pid_file") gpu=$(cat "${PID_DIR}/${exp}.gpu" 2>/dev/null || echo '?')"
    elif [[ -f "${exp_dir}/test_metrics.json" ]]; then
      state="done"
    elif [[ -f "${exp_dir}/last_checkpoint.pt" ]]; then
      state="checkpoint"
    else
      state="missing"
    fi
    echo "${exp}: ${state}"
  done
}

case "${1:-status}" in
  start) start_queue ;;
  pipeline) pipeline ;;
  prepare) prepare_all ;;
  run_all) run_experiment_queue ;;
  stop_old) stop_existing_old_runs ;;
  print) print_plan ;;
  status) status ;;
  *)
    echo "usage: $0 {start|pipeline|prepare|run_all|stop_old|print|status}" >&2
    exit 2
    ;;
esac
