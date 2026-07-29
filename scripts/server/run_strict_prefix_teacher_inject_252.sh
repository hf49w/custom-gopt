#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
MULTIPA_ROOT="${MULTIPA_ROOT:-/DATA_2/MultiPA}"
BAD_GPU="${BAD_GPU:-3}"
ALLOWED_GPUS="${ALLOWED_GPUS:-6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"
MAX_GPU_MEM_USED_MIB="${MAX_GPU_MEM_USED_MIB:-22000}"
SLEEP_SEC="${SLEEP_SEC:-60}"
FORCE="${FORCE:-0}"

STRICT_DATA_DIR="${STRICT_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu}"
TEACHER_DIR="${TEACHER_DIR:-${ROOT}/paper_experiments/teacher/strict_prefix_correct_multipa}"
TEACHER_SPLITS="${TEACHER_SPLITS:-train,val,test}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${TEACHER_DIR}/strict_prefix_teacher_manifest.jsonl}"
TEACHER_WORK_MANIFEST="${TEACHER_WORK_MANIFEST:-${TEACHER_DIR}/strict_prefix_teacher_manifest.remaining.jsonl}"
TEACHER_JSONL="${TEACHER_JSONL:-${TEACHER_DIR}/multipa_strict_prefix_all_splits.jsonl}"
MULTIPA_DATA_DIR="${MULTIPA_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa}"
ORACLE_JSONL="${ORACLE_JSONL:-${ROOT}/paper_experiments/teacher/oracle_gopt_closed_prefix_gt_time_strict_prefix_all_splits.jsonl}"
ORACLE_DATA_DIR="${ORACLE_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa_oracle_gopt_full}"
SLOTPROSODY_DATA_DIR="${SLOTPROSODY_DATA_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_correct_multipa_oracle_gopt_full_slotprosody}"
LOG_DIR="${LOG_DIR:-${ROOT}/paper_experiments/logs}"
PID_DIR="${PID_DIR:-${ROOT}/paper_experiments/pids}"
QUEUE_LOG="${QUEUE_LOG:-${LOG_DIR}/strict_prefix_teacher_inject_queue.log}"
QUEUE_PID="${QUEUE_PID:-${PID_DIR}/strict_prefix_teacher_inject_queue.pid}"

PY="${PY:-${ROOT}/.conda_env/bin/python}"
MULTIPA_PY="${MULTIPA_PY:-${ROOT}/.multipa_env/bin/python}"
PY_ORACLE="${PY_ORACLE:-${ROOT}/.multipa_env/bin/python}"
ALIGNER="${ALIGNER:-${ROOT}/server_assets/models/charsiu-en_w2v2_fc_10ms}"
NUM_MULTIPA_SHARDS="${NUM_MULTIPA_SHARDS:-8}"

mkdir -p "$TEACHER_DIR" "$LOG_DIR" "$PID_DIR"

die() {
  echo "$*" >&2
  exit 1
}

split_list() {
  echo "$TEACHER_SPLITS" | tr ',' '\n' | while read -r split; do
    split="$(echo "$split" | tr -d ' ')"
    [[ -n "$split" ]] && echo "$split"
  done
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
  for gpu_file in "${PID_DIR}"/strict_prefix_teacher_*.gpu; do
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

wait_for_available_gpu() {
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

print_gpu_status() {
  echo "GPU policy: use only GPUs ${ALLOWED_GPUS}; GPU${BAD_GPU}=skipped; per_gpu_slots=${PROCS_PER_GPU}; max_mem=${MAX_GPU_MEM_USED_MIB}MiB"
  local gpu line mem util running state
  while read -r gpu; do
    line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' ')"
    mem="${line%,*}"
    util="${line#*,}"
    running="$(running_jobs_on_gpu "$gpu")"
    if gpu_has_slot "$gpu"; then
      state="slot_available"
    else
      state="slot_full_or_mem_high"
    fi
    echo "gpu=${gpu} mem=${mem}MiB util=${util}% queue_running=${running}/${PROCS_PER_GPU} ${state}"
  done < <(allowed_gpus_list)
  echo "gpu=${BAD_GPU} skipped"
}

assert_inputs() {
  [[ -f "${STRICT_DATA_DIR}/metadata.json" ]] || die "strict prefix data is not finalized: missing ${STRICT_DATA_DIR}/metadata.json"
  for split in $(split_list); do
    [[ -s "${STRICT_DATA_DIR}/${split}_manifest.jsonl" ]] || die "missing ${split} strict manifest"
    [[ -s "${STRICT_DATA_DIR}/${split}_chunks.npz" ]] || die "missing ${split} strict chunks"
  done
  local resolved
  resolved="$(readlink -f "$MULTIPA_ROOT")"
  [[ "$resolved" == "/DATA_2/MultiPA" ]] || die "MULTIPA_ROOT must resolve to /DATA_2/MultiPA, got $resolved"
  [[ "$resolved" != "/DATA_2/guest/MultiPA_pic" ]] || die "/DATA_2/guest/MultiPA_pic is forbidden"
  for path in \
    "$MULTIPA_ROOT/eval_multipa_prefix.py" \
    "$MULTIPA_ROOT/fairseq_hubert/hubert_base_ls960.pt" \
    "$MULTIPA_ROOT/fairseq_roberta" \
    "$MULTIPA_ROOT/model_assessment"; do
    [[ -e "$path" ]] || die "Missing required MultiPA asset: $path"
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
  : > "$TEACHER_MANIFEST"
  for split in $(split_list); do
    cat "${STRICT_DATA_DIR}/${split}_manifest.jsonl" >> "$TEACHER_MANIFEST"
  done
  "$PY" - "$TEACHER_MANIFEST" <<'PY'
import json, sys
from collections import Counter
counts = Counter()
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            counts[str(row.get("split", "unknown"))] += 1
print(json.dumps({"teacher_manifest": sys.argv[1], "rows_by_split": dict(counts), "rows": sum(counts.values())}, ensure_ascii=False), flush=True)
if not counts:
    raise SystemExit("teacher manifest is empty")
PY
}

validate_teacher_jsonl() {
  "$PY" - "$STRICT_DATA_DIR" "$TEACHER_JSONL" "$TEACHER_SPLITS" <<'PY'
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

data_dir = Path(sys.argv[1])
teacher_jsonl = Path(sys.argv[2])
splits = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
ok_keys = set()
all_keys = set()
statuses = Counter()
with teacher_jsonl.open(encoding="utf-8-sig") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("utt_id")), int(row.get("chunk_id", -1)))
        all_keys.add(key)
        status = row.get("status", "ok")
        statuses[status] += 1
        if status == "ok":
            ok_keys.add(key)
summary = {"teacher_jsonl": str(teacher_jsonl), "status": dict(statuses), "splits": {}}
for split in splits:
    manifest_keys = []
    with (data_dir / f"{split}_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest_keys.append((str(row.get("utt_id")), int(row.get("chunk_id", -1))))
    total = len(manifest_keys)
    rows_hit = sum(1 for key in manifest_keys if key in all_keys)
    ok_hit = sum(1 for key in manifest_keys if key in ok_keys)
    coverage = ok_hit / max(1, total)
    summary["splits"][split] = {"rows": total, "teacher_rows": rows_hit, "ok_rows": ok_hit, "ok_coverage": coverage}
    if total and coverage < 0.90:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

build_multipa_work_manifest() {
  "$PY" - "$TEACHER_MANIFEST" "$TEACHER_WORK_MANIFEST" "${TEACHER_DIR}/multipa_strict_prefix.shard_" <<'PY'
import glob
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
work_manifest = Path(sys.argv[2])
shard_prefix = sys.argv[3]
ok_keys = set()
all_status = Counter()
files = sorted(glob.glob(shard_prefix + "*.jsonl"))
for path in files:
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(row.get("utt_id")), int(row.get("chunk_id", -1)))
            status = row.get("status", "ok")
            all_status[status] += 1
            if status == "ok":
                ok_keys.add(key)

total = 0
remaining = 0
work_manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open(encoding="utf-8") as src, work_manifest.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        key = (str(row.get("utt_id")), int(row.get("chunk_id", -1)))
        if key in ok_keys:
            continue
        dst.write(line)
        remaining += 1
print(
    json.dumps(
        {
            "existing_shard_files": files,
            "existing_status": dict(all_status),
            "completed_ok_keys": len(ok_keys),
            "manifest_rows": total,
            "remaining_rows": remaining,
            "work_manifest": str(work_manifest),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
PY
}

combine_multipa_teacher_jsonl() {
  "$PY" - "$TEACHER_JSONL" "${TEACHER_DIR}"/multipa_strict_prefix.shard_*.jsonl <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

output_path = Path(sys.argv[1])
input_paths = [Path(item) for item in sys.argv[2:] if "*" not in item]
rows_by_key = {}
order = []
seen_lines = 0
for path in input_paths:
    if not path.exists():
        continue
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            seen_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (str(row.get("utt_id")), int(row.get("chunk_id", -1)))
            if key not in rows_by_key:
                order.append(key)
                rows_by_key[key] = row
                continue
            current_ok = rows_by_key[key].get("status", "ok") == "ok"
            new_ok = row.get("status", "ok") == "ok"
            if new_ok and not current_ok:
                rows_by_key[key] = row

output_path.parent.mkdir(parents=True, exist_ok=True)
status = Counter()
with output_path.open("w", encoding="utf-8") as dst:
    for key in order:
        row = rows_by_key[key]
        status[row.get("status", "ok")] += 1
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
print(
    json.dumps(
        {
            "output_jsonl": str(output_path),
            "input_files": [str(item) for item in input_paths if item.exists()],
            "input_lines": seen_lines,
            "unique_rows": len(order),
            "status": dict(status),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
PY
}

generate_multipa_teacher() {
  assert_inputs
  build_teacher_manifest
  if [[ -s "$TEACHER_JSONL" && "$FORCE" != "1" ]]; then
    echo "[teacher] exists: $TEACHER_JSONL"
    validate_teacher_jsonl
    return 0
  fi
  rm -f "$TEACHER_JSONL"
  mkdir -p "${TEACHER_DIR}/shards"
  build_multipa_work_manifest
  local remaining_rows
  remaining_rows="$(wc -l < "$TEACHER_WORK_MANIFEST" | tr -d ' ')"
  if [[ "$remaining_rows" -eq 0 ]]; then
    combine_multipa_teacher_jsonl
    validate_teacher_jsonl
    return 0
  fi
  teacher_env
  local pids=()
  local shard_index
  local work_shards="$NUM_MULTIPA_SHARDS"
  if [[ "$remaining_rows" -lt "$work_shards" ]]; then
    work_shards="$remaining_rows"
  fi
  for shard_index in $(seq 0 $((work_shards - 1))); do
    local gpu shard_manifest shard_jsonl shard_log pid_file
    gpu="$(wait_for_available_gpu)"
    shard_manifest="${TEACHER_DIR}/shards/strict_prefix_teacher_manifest.shard_${shard_index}.jsonl"
    shard_jsonl="${TEACHER_DIR}/multipa_strict_prefix.shard_${shard_index}.jsonl"
    shard_log="${TEACHER_DIR}/multipa_strict_prefix.shard_${shard_index}.gpu${gpu}.log"
    pid_file="${PID_DIR}/strict_prefix_teacher_multipa_shard_${shard_index}.pid"
    "$PY" scripts/shard_jsonl.py \
      --input-jsonl "$TEACHER_WORK_MANIFEST" \
      --output-jsonl "$shard_manifest" \
      --num-shards "$work_shards" \
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
    echo "$!" > "$pid_file"
    echo "$gpu" > "${PID_DIR}/strict_prefix_teacher_multipa_shard_${shard_index}.gpu"
    pids+=("$!")
    echo "[teacher] shard=${shard_index}/${work_shards} gpu=${gpu} pid=$! log=${shard_log}"
    sleep 5
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "[teacher] one or more MultiPA shards failed; inspect ${TEACHER_DIR}/multipa_strict_prefix.shard_*.log"
  combine_multipa_teacher_jsonl
  validate_teacher_jsonl
}

inject_multipa() {
  if [[ -d "$MULTIPA_DATA_DIR" && "$FORCE" != "1" ]]; then
    echo "[inject-multipa] exists: $MULTIPA_DATA_DIR"
  else
    "$PY" scripts/local/inject_multipa_teacher_pcn.py \
      --data-dir "$STRICT_DATA_DIR" \
      --teacher-jsonl "$TEACHER_JSONL" \
      --output-dir "$MULTIPA_DATA_DIR" \
      --splits "$TEACHER_SPLITS" \
      --overwrite
  fi
  "$PY" - "$MULTIPA_DATA_DIR" "$TEACHER_SPLITS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
splits = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
summary = {}
for split in splits:
    with np.load(root / f"{split}_chunks.npz") as arr:
        rows = int(arr["cn_post"].shape[0])
        teacher_rows = int((arr["teacher_utt_mask"] > 0).sum()) if "teacher_utt_mask" in arr.files else 0
        word_slots = int((arr["teacher_word_mask"] > 0).sum()) if "teacher_word_mask" in arr.files else 0
        comp = float(arr["teacher_utt_dim_mask"][:, 1].sum()) if "teacher_utt_dim_mask" in arr.files else 0.0
        state_rows = int((arr["teacher_state_mask"] > 0).sum()) if "teacher_state_mask" in arr.files else 0
        summary[split] = {
            "rows": rows,
            "teacher_utt_rows": teacher_rows,
            "teacher_word_slots": word_slots,
            "teacher_state_rows": state_rows,
            "teacher_completeness_mask_sum": comp,
        }
        if rows and teacher_rows / rows < 0.90:
            raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
        if comp != 0.0:
            raise SystemExit(f"{split}: teacher completeness mask must be 0, got {comp}")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

generate_oracle_split() {
  local split="$1"
  local gpu="$2"
  local split_jsonl="${TEACHER_DIR}/oracle_${split}.jsonl"
  local log="${TEACHER_DIR}/oracle_${split}.gpu${gpu}.log"
  local pid_file="${PID_DIR}/strict_prefix_teacher_oracle_${split}.pid"
  (
    cd "$ROOT"
    MULTIPA_REPO_ROOT="$MULTIPA_ROOT" CUDA_VISIBLE_DEVICES="$gpu" "$PY_ORACLE" scripts/local/build_full_oracle_gopt_teacher_pcn.py \
      --pcn-data-dir "$MULTIPA_DATA_DIR" \
      --output-jsonl "$split_jsonl" \
      --splits "$split" \
      --oracle-source auto \
      --multipa-repo-root "$MULTIPA_ROOT" \
      --aligner "$ALIGNER" \
      --word-time-cache "${TEACHER_DIR}/gt_word_time_cache_${split}" \
      --device cuda \
      --align-device cuda:0 \
      --overwrite
  ) > "$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "$gpu" > "${PID_DIR}/strict_prefix_teacher_oracle_${split}.gpu"
  echo "[oracle] split=${split} gpu=${gpu} pid=$! log=${log}"
}

validate_oracle_jsonl() {
  "$PY_ORACLE" - "$MULTIPA_DATA_DIR" "$ORACLE_JSONL" "$TEACHER_SPLITS" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
data_dir = Path(sys.argv[1])
oracle_jsonl = Path(sys.argv[2])
splits = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
ok_keys = set()
all_keys = set()
status = Counter()
with oracle_jsonl.open(encoding="utf-8-sig") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("split")), str(row.get("utt_id")), int(row.get("chunk_id", -1)))
        all_keys.add(key)
        status[row.get("status", "ok")] += 1
        if row.get("status", "ok") == "ok":
            ok_keys.add(key)
summary = {"oracle_jsonl": str(oracle_jsonl), "status": dict(status), "splits": {}}
for split in splits:
    manifest_keys = []
    with (data_dir / f"{split}_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest_keys.append((split, str(row.get("utt_id")), int(row.get("chunk_id", -1))))
    total = len(manifest_keys)
    rows_hit = sum(1 for key in manifest_keys if key in all_keys)
    ok_hit = sum(1 for key in manifest_keys if key in ok_keys)
    coverage = ok_hit / max(1, total)
    summary["splits"][split] = {"rows": total, "oracle_rows": rows_hit, "ok_rows": ok_hit, "ok_coverage": coverage}
    if total and rows_hit / total < 0.90:
        raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
}

generate_oracle() {
  if [[ -s "$ORACLE_JSONL" && "$FORCE" != "1" ]]; then
    echo "[oracle] exists: $ORACLE_JSONL"
    validate_oracle_jsonl
    return 0
  fi
  rm -f "${TEACHER_DIR}"/oracle_*.jsonl "${TEACHER_DIR}"/oracle_*.log "$ORACLE_JSONL"
  local pids=()
  local split
  for split in $(split_list); do
    local gpu
    gpu="$(wait_for_available_gpu)"
    generate_oracle_split "$split" "$gpu"
    pids+=("$(cat "${PID_DIR}/strict_prefix_teacher_oracle_${split}.pid")")
    sleep 5
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "[oracle] one or more split jobs failed; inspect ${TEACHER_DIR}/oracle_*.log"
  : > "$ORACLE_JSONL"
  for split in $(split_list); do
    cat "${TEACHER_DIR}/oracle_${split}.jsonl" >> "$ORACLE_JSONL"
  done
  validate_oracle_jsonl
}

inject_oracle() {
  if [[ -d "$ORACLE_DATA_DIR" && "$FORCE" != "1" ]]; then
    echo "[inject-oracle] exists: $ORACLE_DATA_DIR"
  else
    "$PY" scripts/local/inject_oracle_gopt_teacher_pcn.py \
      --data-dir "$MULTIPA_DATA_DIR" \
      --oracle-jsonl "$ORACLE_JSONL" \
      --output-dir "$ORACLE_DATA_DIR" \
      --splits "$TEACHER_SPLITS" \
      --drop-completeness \
      --overwrite
  fi
  "$PY" - "$ORACLE_DATA_DIR" "$TEACHER_SPLITS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
splits = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
summary = {}
for split in splits:
    with np.load(root / f"{split}_chunks.npz") as arr:
        rows = int(arr["cn_post"].shape[0])
        oracle_rows = int((arr["oracle_utt_mask"] > 0).sum())
        word_slots = int(np.any(arr["oracle_word_dim_mask"] > 0, axis=-1).sum())
        phone_slots = int((arr["oracle_phone_mask"] > 0).sum())
        comp = float(arr["oracle_utt_dim_mask"][:, 1].sum())
        if "oracle_prefix_utt_dim_mask" in arr.files:
            comp += float(arr["oracle_prefix_utt_dim_mask"][:, 1].sum())
        if "oracle_final_utt_dim_mask" in arr.files:
            comp += float(arr["oracle_final_utt_dim_mask"][:, 1].sum())
        summary[split] = {
            "rows": rows,
            "oracle_utt_rows": oracle_rows,
            "oracle_word_slots": word_slots,
            "oracle_phone_slots": phone_slots,
            "oracle_completeness_mask_sum": comp,
            "has_slot_prosody": "slot_prosody" in arr.files,
        }
        if rows and oracle_rows / rows < 0.90:
            raise SystemExit(json.dumps(summary, ensure_ascii=False, indent=2))
        if comp != 0.0:
            raise SystemExit(f"{split}: oracle completeness mask must be 0, got {comp}")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
  if [[ ! -e "$SLOTPROSODY_DATA_DIR" ]]; then
    ln -s "$ORACLE_DATA_DIR" "$SLOTPROSODY_DATA_DIR"
    echo "[slotprosody] symlink ${SLOTPROSODY_DATA_DIR} -> ${ORACLE_DATA_DIR}"
  else
    echo "[slotprosody] exists: $SLOTPROSODY_DATA_DIR"
  fi
}

prepare_all() {
  assert_inputs
  generate_multipa_teacher
  inject_multipa
  generate_oracle
  inject_oracle
}

status() {
  print_gpu_status
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue=running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
  else
    echo "queue=not-running log=$QUEUE_LOG"
  fi
  echo "strict_data=$([[ -f "${STRICT_DATA_DIR}/metadata.json" ]] && echo finalized || echo missing) ${STRICT_DATA_DIR}"
  echo "teacher_jsonl=$([[ -s "$TEACHER_JSONL" ]] && echo present || echo missing) ${TEACHER_JSONL}"
  echo "multipa_data=$([[ -d "$MULTIPA_DATA_DIR" ]] && echo present || echo missing) ${MULTIPA_DATA_DIR}"
  echo "oracle_jsonl=$([[ -s "$ORACLE_JSONL" ]] && echo present || echo missing) ${ORACLE_JSONL}"
  echo "oracle_data=$([[ -d "$ORACLE_DATA_DIR" ]] && echo present || echo missing) ${ORACLE_DATA_DIR}"
  local pf pid name
  for pf in "${PID_DIR}"/strict_prefix_teacher_*.pid; do
    [[ -f "$pf" ]] || continue
    pid="$(cat "$pf" 2>/dev/null || true)"
    name="$(basename "$pf" .pid)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "${name}=running pid=${pid} gpu=$(cat "${pf%.pid}.gpu" 2>/dev/null || echo '?')"
    else
      echo "${name}=not-running pid=${pid:-?}"
    fi
  done
}

print_plan() {
  echo "ROOT=$ROOT"
  echo "MULTIPA_ROOT=$MULTIPA_ROOT"
  echo "STRICT_DATA_DIR=$STRICT_DATA_DIR"
  echo "TEACHER_JSONL=$TEACHER_JSONL"
  echo "MULTIPA_DATA_DIR=$MULTIPA_DATA_DIR"
  echo "ORACLE_JSONL=$ORACLE_JSONL"
  echo "ORACLE_DATA_DIR=$ORACLE_DATA_DIR"
  echo "SLOTPROSODY_DATA_DIR=$SLOTPROSODY_DATA_DIR"
  echo "ALLOWED_GPUS=$ALLOWED_GPUS"
  echo "PROCS_PER_GPU=$PROCS_PER_GPU"
  echo "NUM_MULTIPA_SHARDS=$NUM_MULTIPA_SHARDS"
  echo "pipeline: rebuild strict MultiPA JSONL, inject MultiPA, rebuild strict oracle JSONL, inject oracle; GPU${BAD_GPU} skipped"
}

start_queue() {
  if [[ -f "$QUEUE_PID" ]] && kill -0 "$(cat "$QUEUE_PID")" 2>/dev/null; then
    echo "queue already running pid=$(cat "$QUEUE_PID") log=$QUEUE_LOG"
    return 0
  fi
  mkdir -p "$LOG_DIR" "$PID_DIR"
  local script
  script="$(readlink -f "$0")"
  nohup "$script" pipeline >> "$QUEUE_LOG" 2>&1 &
  echo "$!" > "$QUEUE_PID"
  echo "queue_pid=$! log=$QUEUE_LOG"
}

case "${1:-status}" in
  prepare|pipeline)
    prepare_all
    ;;
  start)
    start_queue
    ;;
  print)
    print_plan
    ;;
  status)
    status
    ;;
  validate)
    validate_teacher_jsonl
    validate_oracle_jsonl
    ;;
  *)
    echo "usage: $0 {start|pipeline|prepare|print|status|validate}" >&2
    exit 2
    ;;
esac
