#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/server_assets}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-all}"
AUTO_RESUME="${AUTO_RESUME:-1}"

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT}/speechocean762/speechocean762}"
SCORES_JSON="${SCORES_JSON:-${REPO_ROOT}/src/prep_data/scores.json}"

ALIGNER_MODEL="${ALIGNER_MODEL:-charsiu/en_w2v2_tiny_fc_10ms}"
if [[ -n "${ALIGNER_MODEL_DIR:-}" ]]; then
  ALIGNER_MODEL="${ALIGNER_MODEL_DIR}"
fi
WHISPER_BASE_MODEL="${WHISPER_BASE_MODEL:-${DATA_ROOT}/models/whisper-base}"
if [[ ! -e "${WHISPER_BASE_MODEL}" ]]; then
  WHISPER_BASE_MODEL="${WHISPER_BASE_MODEL_FALLBACK:-openai/whisper-base}"
fi
TIMESTAMP_BACKEND="${TIMESTAMP_BACKEND:-transformers}"
LANGUAGE="${LANGUAGE:-english}"

CHUNK_SEC="${CHUNK_SEC:-0.64}"
RIGHT_CONTEXT_SEC="${RIGHT_CONTEXT_SEC:-0.16}"
MIN_UTT_MATCH_RATIO="${MIN_UTT_MATCH_RATIO:-0.5}"
VAL_SPEAKER_RATIO="${VAL_SPEAKER_RATIO:-0.5}"
SPLIT_SEED="${SPLIT_SEED:-1337}"
TRAIN_DEVICE="${TRAIN_DEVICE:-}"
GOPT_DATA_MULTI_GPU="${GOPT_DATA_MULTI_GPU:-1}"
GOPT_DATA_GPU_IDS="${GOPT_DATA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
GOPT_DATA_FINALIZE_DEVICE="${GOPT_DATA_FINALIZE_DEVICE:-cpu}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-4}"
ASR_MIN_BATCH_SIZE="${ASR_MIN_BATCH_SIZE:-1}"
ASR_MAX_NEW_TOKENS="${ASR_MAX_NEW_TOKENS:-128}"
ASR_NO_REPEAT_NGRAM_SIZE="${ASR_NO_REPEAT_NGRAM_SIZE:-0}"
ASR_MAX_WORDS="${ASR_MAX_WORDS:-64}"
ASR_MAX_VISIBLE_PHONES="${ASR_MAX_VISIBLE_PHONES:-100}"
ASR_MAX_PHONE_RATIO="${ASR_MAX_PHONE_RATIO:-3.0}"
ASR_REPEAT_NGRAM_MIN_REPEATS="${ASR_REPEAT_NGRAM_MIN_REPEATS:-4}"
ASR_REPEAT_MAX_NGRAM_SIZE="${ASR_REPEAT_MAX_NGRAM_SIZE:-12}"
ASR_REPEAT_NGRAM_COVERAGE="${ASR_REPEAT_NGRAM_COVERAGE:-0.6}"
ASR_REPEAT_TOKEN_RATIO="${ASR_REPEAT_TOKEN_RATIO:-0.5}"
ASR_TORCH_DTYPE="${ASR_TORCH_DTYPE:-auto}"
ASR_USE_CACHE="${ASR_USE_CACHE:-0}"
ASR_EMPTY_CACHE="${ASR_EMPTY_CACHE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

WHISPER_PREFIX_DIR="${WHISPER_PREFIX_DIR:-${REPO_ROOT}/data/streaming_whisper_prefix}"
WHISPER_EXP_DIR="${WHISPER_EXP_DIR:-${REPO_ROOT}/exp/streaming-whisper-base}"
ASR_GOPT_DATA_DIR="${ASR_GOPT_DATA_DIR:-${REPO_ROOT}/data/streaming_asr_gopt}"
GOPT_EXP_DIR="${GOPT_EXP_DIR:-${REPO_ROOT}/exp/streaming-asr-gopt}"

WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-8}"
WHISPER_EVAL_BATCH_SIZE="${WHISPER_EVAL_BATCH_SIZE:-8}"
WHISPER_EPOCHS="${WHISPER_EPOCHS:-8}"
WHISPER_NUM_WORKERS="${WHISPER_NUM_WORKERS:-4}"
WHISPER_EVAL_GENERATE_MAX_SAMPLES="${WHISPER_EVAL_GENERATE_MAX_SAMPLES:-256}"
WHISPER_COMPILE="${WHISPER_COMPILE:-0}"
WHISPER_TF32="${WHISPER_TF32:-0}"

GOPT_BATCH_SIZE="${GOPT_BATCH_SIZE:-25}"
GOPT_EPOCHS="${GOPT_EPOCHS:-100}"
GOPT_DEPTH="${GOPT_DEPTH:-3}"
GOPT_HEADS="${GOPT_HEADS:-1}"
GOPT_EMBED_DIM="${GOPT_EMBED_DIM:-24}"
GOPT_MAIN_CONTEXT_TOKENS="${GOPT_MAIN_CONTEXT_TOKENS:-4,8,12,16}"
GOPT_RIGHT_CONTEXT_TOKENS="${GOPT_RIGHT_CONTEXT_TOKENS:-0,1,2,4}"
GOPT_NUM_WORKERS="${GOPT_NUM_WORKERS:-4}"
GOPT_COMPILE="${GOPT_COMPILE:-0}"
GOPT_TF32="${GOPT_TF32:-0}"

ASR_MODEL_PATH="${ASR_MODEL_PATH:-${WHISPER_EXP_DIR}/best_model}"
if [[ ! -d "${ASR_MODEL_PATH}" && -d "${WHISPER_EXP_DIR}/last_model" ]]; then
  ASR_MODEL_PATH="${WHISPER_EXP_DIR}/last_model"
fi

if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
elif [[ "${TRAIN_DEVICE:-}" == cuda* || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE

whisper_resume_args=()
gopt_resume_args=()
gopt_data_resume_args=()
if [[ "${AUTO_RESUME}" == "1" && -f "${WHISPER_EXP_DIR}/last_checkpoint.pt" ]]; then
  whisper_resume_args+=(--resume)
fi
if [[ "${AUTO_RESUME}" == "1" && -f "${GOPT_EXP_DIR}/last_checkpoint.pt" ]]; then
  gopt_resume_args+=(--resume)
fi
if [[ "${AUTO_RESUME}" == "1" && -d "${ASR_GOPT_DATA_DIR}/progress" ]]; then
  gopt_data_resume_args+=(--resume)
else
  gopt_data_resume_args+=(--overwrite)
fi

split_csv() {
  local raw="${1:-}"
  local -n out_ref=$2
  out_ref=()
  if [[ -z "${raw}" ]]; then
    return 0
  fi
  IFS=',' read -r -a out_ref <<< "${raw}"
}

run_prefix_data() {
  device_args=()
  if [[ -n "${TRAIN_DEVICE}" ]]; then
    device_args+=(--device "${TRAIN_DEVICE}")
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_whisper_prefix_data.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --output-dir "${WHISPER_PREFIX_DIR}" \
    --aligner-model "${ALIGNER_MODEL}" \
    --val-speaker-ratio "${VAL_SPEAKER_RATIO}" \
    --split-seed "${SPLIT_SEED}" \
    --chunk-sec "${CHUNK_SEC}" \
    --right-context-sec "${RIGHT_CONTEXT_SEC}" \
    "${device_args[@]}" \
    --overwrite
}

run_whisper_train() {
  mkdir -p "${WHISPER_EXP_DIR}"
  device_args=()
  extra_args=()
  if [[ -n "${TRAIN_DEVICE}" ]]; then
    device_args+=(--device "${TRAIN_DEVICE}")
  fi
  if [[ "${WHISPER_COMPILE}" == "1" ]]; then
    extra_args+=(--compile)
  fi
  if [[ "${WHISPER_TF32}" == "1" ]]; then
    extra_args+=(--tf32)
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/src/train_streaming_whisper.py" \
    --data-dir "${WHISPER_PREFIX_DIR}" \
    --exp-dir "${WHISPER_EXP_DIR}" \
    --model-name-or-path "${WHISPER_BASE_MODEL}" \
    --language "${LANGUAGE}" \
    --batch-size "${WHISPER_BATCH_SIZE}" \
    --eval-batch-size "${WHISPER_EVAL_BATCH_SIZE}" \
    --num-workers "${WHISPER_NUM_WORKERS}" \
    --eval-generate-max-samples "${WHISPER_EVAL_GENERATE_MAX_SAMPLES}" \
    --n-epochs "${WHISPER_EPOCHS}" \
    "${device_args[@]}" \
    "${extra_args[@]}" \
    "${whisper_resume_args[@]}"
}

run_asr_gopt_data() {
  device_args=()
  extra_args=()
  common_args=()
  if [[ -n "${TRAIN_DEVICE}" ]]; then
    device_args+=(--device "${TRAIN_DEVICE}")
  fi
  if [[ "${ASR_USE_CACHE}" == "1" ]]; then
    extra_args+=(--asr-use-cache)
  fi
  if [[ "${ASR_EMPTY_CACHE}" == "1" ]]; then
    extra_args+=(--asr-empty-cache)
  fi
  common_args=(
    --dataset-root "${DATASET_ROOT}"
    --scores-json "${SCORES_JSON}"
    --output-dir "${ASR_GOPT_DATA_DIR}"
    --aligner-model "${ALIGNER_MODEL}"
    --asr-model "${ASR_MODEL_PATH}"
    --val-speaker-ratio "${VAL_SPEAKER_RATIO}"
    --split-seed "${SPLIT_SEED}"
    --timestamp-backend "${TIMESTAMP_BACKEND}"
    --language "${LANGUAGE}"
    --chunk-sec "${CHUNK_SEC}"
    --right-context-sec "${RIGHT_CONTEXT_SEC}"
    --min-utt-match-ratio "${MIN_UTT_MATCH_RATIO}"
    --asr-batch-size "${ASR_BATCH_SIZE}"
    --asr-min-batch-size "${ASR_MIN_BATCH_SIZE}"
    --asr-max-new-tokens "${ASR_MAX_NEW_TOKENS}"
    --asr-no-repeat-ngram-size "${ASR_NO_REPEAT_NGRAM_SIZE}"
    --asr-max-words "${ASR_MAX_WORDS}"
    --asr-max-visible-phones "${ASR_MAX_VISIBLE_PHONES}"
    --asr-max-phone-ratio "${ASR_MAX_PHONE_RATIO}"
    --asr-repeat-ngram-min-repeats "${ASR_REPEAT_NGRAM_MIN_REPEATS}"
    --asr-repeat-max-ngram-size "${ASR_REPEAT_MAX_NGRAM_SIZE}"
    --asr-repeat-ngram-coverage "${ASR_REPEAT_NGRAM_COVERAGE}"
    --asr-repeat-token-ratio "${ASR_REPEAT_TOKEN_RATIO}"
    --asr-torch-dtype "${ASR_TORCH_DTYPE}"
  )

  gpu_ids=()
  split_csv "${GOPT_DATA_GPU_IDS}" gpu_ids
  if [[ "${GOPT_DATA_MULTI_GPU}" == "1" && "${#gpu_ids[@]}" -gt 1 ]]; then
    shard_root="${ASR_GOPT_DATA_DIR}/shards"
    if [[ "${AUTO_RESUME}" != "1" || ! -d "${shard_root}" ]]; then
      rm -rf "${ASR_GOPT_DATA_DIR}"
      mkdir -p "${ASR_GOPT_DATA_DIR}"
    fi
    log_dir="${ASR_GOPT_DATA_DIR}/worker_logs"
    mkdir -p "${log_dir}"

    pids=()
    for idx in "${!gpu_ids[@]}"; do
      gpu_id="${gpu_ids[$idx]}"
      worker_log="${log_dir}/shard_${idx}.log"
      worker_output_dir="${shard_root}/shard_${idx}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_streaming_asr_gopt_data.py" \
        --dataset-root "${DATASET_ROOT}" \
        --scores-json "${SCORES_JSON}" \
        --output-dir "${worker_output_dir}" \
        --aligner-model "${ALIGNER_MODEL}" \
        --asr-model "${ASR_MODEL_PATH}" \
        --val-speaker-ratio "${VAL_SPEAKER_RATIO}" \
        --split-seed "${SPLIT_SEED}" \
        --timestamp-backend "${TIMESTAMP_BACKEND}" \
        --language "${LANGUAGE}" \
        --chunk-sec "${CHUNK_SEC}" \
        --right-context-sec "${RIGHT_CONTEXT_SEC}" \
        --min-utt-match-ratio "${MIN_UTT_MATCH_RATIO}" \
        --asr-batch-size "${ASR_BATCH_SIZE}" \
        --asr-min-batch-size "${ASR_MIN_BATCH_SIZE}" \
        --asr-max-new-tokens "${ASR_MAX_NEW_TOKENS}" \
        --asr-no-repeat-ngram-size "${ASR_NO_REPEAT_NGRAM_SIZE}" \
        --asr-max-words "${ASR_MAX_WORDS}" \
        --asr-max-visible-phones "${ASR_MAX_VISIBLE_PHONES}" \
        --asr-max-phone-ratio "${ASR_MAX_PHONE_RATIO}" \
        --asr-repeat-ngram-min-repeats "${ASR_REPEAT_NGRAM_MIN_REPEATS}" \
        --asr-repeat-max-ngram-size "${ASR_REPEAT_MAX_NGRAM_SIZE}" \
        --asr-repeat-ngram-coverage "${ASR_REPEAT_NGRAM_COVERAGE}" \
        --asr-repeat-token-ratio "${ASR_REPEAT_TOKEN_RATIO}" \
        --asr-torch-dtype "${ASR_TORCH_DTYPE}" \
        --device cuda \
        --num-shards "${#gpu_ids[@]}" \
        --shard-index "${idx}" \
        --skip-finalize \
        --resume \
        "${extra_args[@]}" > "${worker_log}" 2>&1 &
      pids+=($!)
    done

    worker_failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        worker_failed=1
      fi
    done
    if [[ "${worker_failed}" == "1" ]]; then
      echo "One or more gopt_data shard workers failed. Check ${log_dir}/shard_*.log" >&2
      return 1
    fi

    rm -rf "${ASR_GOPT_DATA_DIR}/progress"
    mkdir -p "${ASR_GOPT_DATA_DIR}/progress/train" "${ASR_GOPT_DATA_DIR}/progress/val" "${ASR_GOPT_DATA_DIR}/progress/test"
    for idx in "${!gpu_ids[@]}"; do
      worker_output_dir="${shard_root}/shard_${idx}"
      for split_name in train val test; do
        if [[ -d "${worker_output_dir}/progress/${split_name}" ]]; then
          cp -f "${worker_output_dir}/progress/${split_name}/"*.pkl "${ASR_GOPT_DATA_DIR}/progress/${split_name}/" 2>/dev/null || true
        fi
      done
    done

    finalize_device_args=()
    if [[ -n "${GOPT_DATA_FINALIZE_DEVICE}" ]]; then
      finalize_device_args+=(--device "${GOPT_DATA_FINALIZE_DEVICE}")
    fi
    "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_streaming_asr_gopt_data.py" \
      "${common_args[@]}" \
      "${finalize_device_args[@]}" \
      --finalize-only \
      --resume
    return
  fi

  "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_streaming_asr_gopt_data.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --output-dir "${ASR_GOPT_DATA_DIR}" \
    --aligner-model "${ALIGNER_MODEL}" \
    --asr-model "${ASR_MODEL_PATH}" \
    --val-speaker-ratio "${VAL_SPEAKER_RATIO}" \
    --split-seed "${SPLIT_SEED}" \
    --timestamp-backend "${TIMESTAMP_BACKEND}" \
    --language "${LANGUAGE}" \
    --chunk-sec "${CHUNK_SEC}" \
    --right-context-sec "${RIGHT_CONTEXT_SEC}" \
    --min-utt-match-ratio "${MIN_UTT_MATCH_RATIO}" \
    --asr-batch-size "${ASR_BATCH_SIZE}" \
    --asr-min-batch-size "${ASR_MIN_BATCH_SIZE}" \
    --asr-max-new-tokens "${ASR_MAX_NEW_TOKENS}" \
    --asr-no-repeat-ngram-size "${ASR_NO_REPEAT_NGRAM_SIZE}" \
    --asr-max-words "${ASR_MAX_WORDS}" \
    --asr-max-visible-phones "${ASR_MAX_VISIBLE_PHONES}" \
    --asr-max-phone-ratio "${ASR_MAX_PHONE_RATIO}" \
    --asr-repeat-ngram-min-repeats "${ASR_REPEAT_NGRAM_MIN_REPEATS}" \
    --asr-repeat-max-ngram-size "${ASR_REPEAT_MAX_NGRAM_SIZE}" \
    --asr-repeat-ngram-coverage "${ASR_REPEAT_NGRAM_COVERAGE}" \
    --asr-repeat-token-ratio "${ASR_REPEAT_TOKEN_RATIO}" \
    --asr-torch-dtype "${ASR_TORCH_DTYPE}" \
    "${device_args[@]}" \
    "${extra_args[@]}" \
    "${gopt_data_resume_args[@]}"
}

run_gopt_train() {
  mkdir -p "${GOPT_EXP_DIR}"
  device_args=()
  extra_args=()
  if [[ -n "${TRAIN_DEVICE}" ]]; then
    device_args+=(--device "${TRAIN_DEVICE}")
  fi
  if [[ "${GOPT_COMPILE}" == "1" ]]; then
    extra_args+=(--compile)
  fi
  if [[ "${GOPT_TF32}" == "1" ]]; then
    extra_args+=(--tf32)
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/src/train_streaming_charsiu.py" \
    --data-dir "${ASR_GOPT_DATA_DIR}" \
    --exp-dir "${GOPT_EXP_DIR}" \
    --depth "${GOPT_DEPTH}" \
    --heads "${GOPT_HEADS}" \
    --batch-size "${GOPT_BATCH_SIZE}" \
    --num-workers "${GOPT_NUM_WORKERS}" \
    --embed-dim "${GOPT_EMBED_DIM}" \
    --model streaming_gopt \
    --n-epochs "${GOPT_EPOCHS}" \
    --main-context-tokens "${GOPT_MAIN_CONTEXT_TOKENS}" \
    --right-context-tokens "${GOPT_RIGHT_CONTEXT_TOKENS}" \
    "${device_args[@]}" \
    "${extra_args[@]}" \
    "${gopt_resume_args[@]}"
}

case "${STAGE}" in
  prefix)
    run_prefix_data
    ;;
  whisper)
    run_whisper_train
    ;;
  gopt_data)
    run_asr_gopt_data
    ;;
  gopt)
    run_gopt_train
    ;;
  all)
    run_prefix_data
    run_whisper_train
    run_asr_gopt_data
    run_gopt_train
    ;;
  *)
    echo "Usage: $0 [prefix|whisper|gopt_data|gopt|all]" >&2
    exit 1
    ;;
esac
