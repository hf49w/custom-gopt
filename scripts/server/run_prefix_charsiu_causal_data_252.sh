#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
BAD_GPU="${BAD_GPU:-3}"
SLEEP_SEC="${SLEEP_SEC:-60}"
ALLOWED_GPUS="${ALLOWED_GPUS:-6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"
MAX_GPU_MEM_USED_MIB="${MAX_GPU_MEM_USED_MIB:-22000}"
PY="${PY:-${ROOT}/.multipa_env/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/server_assets/speechocean762/speechocean762}"
FULL_DATA_DIR="${FULL_DATA_DIR:-${ROOT}/data/streaming_pcn_gopt_v2_stateful}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/paper_experiments/cache/prefix_charsiu}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${ROOT}/paper_experiments/data/streaming_pcn_gopt_v2_prefix_charsiu_smoke}"
SMOKE_CACHE_DIR="${SMOKE_CACHE_DIR:-${ROOT}/paper_experiments/cache/prefix_charsiu_smoke}"
LOG_DIR="${LOG_DIR:-${ROOT}/paper_experiments/logs}"
PID_DIR="${PID_DIR:-${ROOT}/paper_experiments/pids}"
ALIGNER="${ALIGNER:-${ROOT}/server_assets/models/charsiu_en_w2v2_tiny_fc_10ms}"
ASR_MODEL="${ASR_MODEL:-${ROOT}/exp/streaming-whisper-base/best_model}"
NUM_SHARDS="${NUM_SHARDS:-7}"
SMOKE_MARKER="${SMOKE_MARKER:-${SMOKE_OUTPUT_DIR}/SMOKE_OK}"

mkdir -p "$LOG_DIR" "$PID_DIR"

die() {
  echo "$*" >&2
  exit 1
}

common_env() {
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HOME="${HF_HOME:-${ROOT}/server_assets/hf_home}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${ROOT}/server_assets/hf_home/transformers}"
  export CHARSIU_SRC_DIR="${CHARSIU_SRC_DIR:-${ROOT}/server_assets/src/charsiu_repo}"
  local tokenizer_cache="${ROOT}/server_assets/hf_cache/transformers/models--charsiu--tokenizer_en_cmu"
  local tokenizer_dir="${ROOT}/server_assets/src/charsiu_repo/local"
  if [[ -f "${tokenizer_cache}/refs/main" ]]; then
    local revision
    revision="$(cat "${tokenizer_cache}/refs/main")"
    if [[ -f "${tokenizer_cache}/snapshots/${revision}/vocab.json" ]]; then
      tokenizer_dir="${tokenizer_cache}/snapshots/${revision}"
    fi
  fi
  export CHARSU_TOKENIZER_EN_CMU="${CHARSU_TOKENIZER_EN_CMU:-${tokenizer_dir}}"
  export CHARSIU_TOKENIZER_EN_CMU="${CHARSIU_TOKENIZER_EN_CMU:-${tokenizer_dir}}"
}

allowed_gpus_list() {
  echo "$ALLOWED_GPUS" | tr ',' '\n' | while read -r gpu; do
    gpu="$(echo "$gpu" | tr -d ' ')"
    [[ -n "$gpu" && "$gpu" != "$BAD_GPU" ]] && echo "$gpu"
  done
}

gpu_mem_used_mib() {
  local gpu="$1"
  local line mem
  line="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d ' ')"
  [[ -n "$line" ]] || return 1
  mem="$line"
  [[ "$mem" =~ ^[0-9]+$ ]] || return 1
  echo "$mem"
}

running_shards_on_gpu() {
  local gpu="$1"
  local count=0
  local gpu_file pid_file pid
  for gpu_file in "${PID_DIR}"/prefix_charsiu_shard_*.gpu; do
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
  running="$(running_shards_on_gpu "$gpu")"
  [[ "$running" -lt "$PROCS_PER_GPU" ]] || return 1
  [[ "$mem" -lt "$MAX_GPU_MEM_USED_MIB" ]] || return 1
  return 0
}

available_gpus_now() {
  local gpu
  while read -r gpu; do
    if gpu_has_slot "$gpu"; then
      echo "$gpu"
    fi
  done < <(allowed_gpus_list)
}

wait_for_available_gpu() {
  local gpu
  while true; do
    while read -r gpu; do
      [[ -n "$gpu" ]] || continue
      echo "$gpu"
      return 0
    done < <(available_gpus_now)
    echo "[wait] no shard slot on GPUs ${ALLOWED_GPUS}; GPU${BAD_GPU} skipped; procs_per_gpu=${PROCS_PER_GPU}; max_mem=${MAX_GPU_MEM_USED_MIB}MiB; sleeping ${SLEEP_SEC}s" >&2
    sleep "$SLEEP_SEC"
  done
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$gpu" '$1 == gpu && $2 ~ /^[0-9]+$/ && $3 == "C" { found = 1 } END { exit(found ? 0 : 1) }'
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
    running="$(running_shards_on_gpu "$gpu")"
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

gpu_is_available() {
  local gpu="$1"
  gpu_has_slot "$gpu"
}

run_builder() {
  local gpu="$1"
  local output_dir="$2"
  local cache_dir="$3"
  local target_splits="$4"
  shift 4
  common_env
  cd "$ROOT"
  if [[ "$gpu" == "cpu" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PY" src/prep_data/build_streaming_pcn_gopt_data.py \
      --dataset-root "$DATASET_ROOT" \
      --scores-json "${ROOT}/src/prep_data/scores.json" \
      --output-dir "$output_dir" \
      --aligner-model "$ALIGNER" \
      --charsiu-src-dir "${CHARSIU_SRC_DIR:-${ROOT}/server_assets/src/charsiu_repo}" \
      --asr-model "$ASR_MODEL" \
      --charsiu-mode prefix_recompute \
      --prefix-charsiu-cache-dir "$cache_dir" \
      --target-splits "$target_splits" \
      --chunk-sec 0.64 \
      --right-context-sec 0.16 \
      --include-slot-prosody \
      --resume \
      --device cpu \
      "$@"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/prep_data/build_streaming_pcn_gopt_data.py \
      --dataset-root "$DATASET_ROOT" \
      --scores-json "${ROOT}/src/prep_data/scores.json" \
      --output-dir "$output_dir" \
      --aligner-model "$ALIGNER" \
      --charsiu-src-dir "${CHARSIU_SRC_DIR:-${ROOT}/server_assets/src/charsiu_repo}" \
      --asr-model "$ASR_MODEL" \
      --charsiu-mode prefix_recompute \
      --prefix-charsiu-cache-dir "$cache_dir" \
      --target-splits "$target_splits" \
      --chunk-sec 0.64 \
      --right-context-sec 0.16 \
      --include-slot-prosody \
      --resume \
      --device cuda \
      "$@"
  fi
}

launch_shard() {
  local shard_index="$1"
  local num_shards="$2"
  local gpu="$3"
  local log="${LOG_DIR}/prefix_charsiu_shard_${shard_index}_of_${num_shards}.gpu${gpu}.log"
  local pid_file="${PID_DIR}/prefix_charsiu_shard_${shard_index}.pid"
  common_env
  echo "[launch] shard=${shard_index}/${num_shards} gpu=${gpu} log=${log}"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/prep_data/build_streaming_pcn_gopt_data.py \
      --dataset-root "$DATASET_ROOT" \
      --scores-json "${ROOT}/src/prep_data/scores.json" \
      --output-dir "$OUTPUT_DIR" \
      --aligner-model "$ALIGNER" \
      --charsiu-src-dir "${CHARSIU_SRC_DIR:-${ROOT}/server_assets/src/charsiu_repo}" \
      --asr-model "$ASR_MODEL" \
      --charsiu-mode prefix_recompute \
      --prefix-charsiu-cache-dir "$CACHE_DIR" \
      --target-splits train,val,test \
      --chunk-sec 0.64 \
      --right-context-sec 0.16 \
      --include-slot-prosody \
      --resume \
      --skip-finalize \
      --device cuda \
      --num-shards "$num_shards" \
      --shard-index "$shard_index"
  ) > "$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "$gpu" > "${PID_DIR}/prefix_charsiu_shard_${shard_index}.gpu"
}

finalize_data() {
  common_env
  cd "$ROOT"
  "$PY" src/prep_data/build_streaming_pcn_gopt_data.py \
    --dataset-root "$DATASET_ROOT" \
    --scores-json "${ROOT}/src/prep_data/scores.json" \
    --output-dir "$OUTPUT_DIR" \
    --aligner-model "$ALIGNER" \
    --charsiu-src-dir "${CHARSIU_SRC_DIR:-${ROOT}/server_assets/src/charsiu_repo}" \
    --asr-model "$ASR_MODEL" \
    --charsiu-mode prefix_recompute \
    --prefix-charsiu-cache-dir "$CACHE_DIR" \
    --target-splits train,val,test \
    --chunk-sec 0.64 \
    --right-context-sec 0.16 \
    --include-slot-prosody \
    --finalize-only
  "$PY" scripts/paper/audit_charsiu_causality.py \
    --full-data-dir "$FULL_DATA_DIR" \
    --prefix-data-dir "$OUTPUT_DIR" \
    --output-dir "${ROOT}/paper_experiments" \
    --overwrite
}

queue_run_all() {
  if [[ -f "$OUTPUT_DIR/metadata.json" && "${FORCE:-0}" != "1" ]]; then
    die "$OUTPUT_DIR is already finalized; choose a new OUTPUT_DIR to avoid overwriting."
  fi
  if [[ "${SKIP_SMOKE:-0}" != "1" && ! -f "$SMOKE_MARKER" ]]; then
    local smoke_gpu
    smoke_gpu="$(wait_for_available_gpu)"
    echo "[queue] running strict-prefix smoke first on GPU${smoke_gpu}; GPU${BAD_GPU} skipped"
    run_builder "$smoke_gpu" "$SMOKE_OUTPUT_DIR" "$SMOKE_CACHE_DIR" test --num-shards 5000 --shard-index 0 --skip-finalize
    mkdir -p "$(dirname "$SMOKE_MARKER")"
    date -Is > "$SMOKE_MARKER"
    echo "[queue] smoke ok: $SMOKE_MARKER"
  fi
  mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
  local num_shards="$NUM_SHARDS"
  local pids=()
  for shard_index in $(seq 0 $((num_shards - 1))); do
    local gpu
    gpu="$(wait_for_available_gpu)"
    launch_shard "$shard_index" "$num_shards" "$gpu"
    pids+=("$(cat "${PID_DIR}/prefix_charsiu_shard_${shard_index}.pid")")
    sleep 5
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "[queue] at least one shard failed; inspect ${LOG_DIR}/prefix_charsiu_shard_*.log"
  finalize_data
  echo "[queue] finalized: $OUTPUT_DIR"
}

progress_summary() {
  local dir="$1"
  for split in train val test; do
    local count=0
    if [[ -d "${dir}/progress/${split}" ]]; then
      count="$(find "${dir}/progress/${split}" -type f -name '*.pkl' | wc -l)"
    fi
    echo "progress_${split}_utterances=${count}"
  done
  [[ -f "${dir}/metadata.json" ]] && echo "finalized=${dir}" || echo "not_finalized=${dir}"
}

case "${1:-status}" in
  smoke_cpu)
    echo "[smoke_cpu] output=${SMOKE_OUTPUT_DIR}"
    run_builder cpu "$SMOKE_OUTPUT_DIR" "$SMOKE_CACHE_DIR" test --num-shards 5000 --shard-index 0 --skip-finalize
    ;;
  smoke)
    gpu="$(wait_for_available_gpu)"
    echo "[smoke] gpu=${gpu} output=${SMOKE_OUTPUT_DIR}"
    run_builder "$gpu" "$SMOKE_OUTPUT_DIR" "$SMOKE_CACHE_DIR" test --num-shards 5000 --shard-index 0 --skip-finalize
    ;;
  run_all)
    queue_log="${LOG_DIR}/prefix_charsiu_queue.log"
    queue_pid="${PID_DIR}/prefix_charsiu_queue.pid"
    if [[ -f "$queue_pid" ]] && kill -0 "$(cat "$queue_pid")" 2>/dev/null; then
      die "queue already running pid=$(cat "$queue_pid")"
    fi
    nohup bash "$0" _queue > "$queue_log" 2>&1 &
    echo "$!" > "$queue_pid"
    echo "queue_pid=$(cat "$queue_pid") log=${queue_log}"
    ;;
  _queue)
    queue_run_all
    ;;
  finalize)
    finalize_data
    ;;
  print)
    echo "ROOT=$ROOT"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "CACHE_DIR=$CACHE_DIR"
    echo "ALIGNER=$ALIGNER"
    echo "ASR_MODEL=$ASR_MODEL"
    echo "NUM_SHARDS=$NUM_SHARDS"
    echo "ALLOWED_GPUS=$ALLOWED_GPUS"
    echo "PROCS_PER_GPU=$PROCS_PER_GPU"
    echo "MAX_GPU_MEM_USED_MIB=$MAX_GPU_MEM_USED_MIB"
    echo "run_all: uses only GPUs ${ALLOWED_GPUS}, skips GPU${BAD_GPU}, allows multiple prefix data shards per allowed GPU, then finalize-only and causality audit"
    ;;
  status)
    print_gpu_status
    if [[ -f "${PID_DIR}/prefix_charsiu_queue.pid" ]]; then
      pid="$(cat "${PID_DIR}/prefix_charsiu_queue.pid")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "queue=running pid=${pid} log=${LOG_DIR}/prefix_charsiu_queue.log"
      else
        echo "queue=not-running pid=${pid} log=${LOG_DIR}/prefix_charsiu_queue.log"
      fi
    else
      echo "queue=missing"
    fi
    for pid_file in "${PID_DIR}"/prefix_charsiu_shard_*.pid; do
      [[ -f "$pid_file" ]] || continue
      pid="$(cat "$pid_file")"
      name="$(basename "$pid_file" .pid)"
      if kill -0 "$pid" 2>/dev/null; then
        echo "${name}=running pid=${pid}"
      else
        echo "${name}=not-running pid=${pid}"
      fi
    done
    progress_summary "$OUTPUT_DIR"
    if [[ -f "$SMOKE_MARKER" ]]; then
      echo "smoke=ok marker=${SMOKE_MARKER}"
    else
      echo "smoke=missing marker=${SMOKE_MARKER}"
    fi
    ;;
  *)
    die "usage: $0 {smoke_cpu|smoke|run_all|finalize|print|status}"
    ;;
esac
