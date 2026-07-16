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

WHISPER_PREFIX_DIR="${WHISPER_PREFIX_DIR:-${REPO_ROOT}/data/streaming_whisper_prefix}"
WHISPER_EXP_DIR="${WHISPER_EXP_DIR:-${REPO_ROOT}/exp/streaming-whisper-base}"
ASR_GOPT_DATA_DIR="${ASR_GOPT_DATA_DIR:-${REPO_ROOT}/data/streaming_asr_gopt}"
GOPT_EXP_DIR="${GOPT_EXP_DIR:-${REPO_ROOT}/exp/streaming-asr-gopt}"

WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-8}"
WHISPER_EVAL_BATCH_SIZE="${WHISPER_EVAL_BATCH_SIZE:-8}"
WHISPER_EPOCHS="${WHISPER_EPOCHS:-8}"

GOPT_BATCH_SIZE="${GOPT_BATCH_SIZE:-25}"
GOPT_EPOCHS="${GOPT_EPOCHS:-100}"
GOPT_DEPTH="${GOPT_DEPTH:-3}"
GOPT_HEADS="${GOPT_HEADS:-1}"
GOPT_EMBED_DIM="${GOPT_EMBED_DIM:-24}"
GOPT_MAIN_CONTEXT_TOKENS="${GOPT_MAIN_CONTEXT_TOKENS:-4,8,12,16}"
GOPT_RIGHT_CONTEXT_TOKENS="${GOPT_RIGHT_CONTEXT_TOKENS:-0,1,2,4}"

ASR_MODEL_PATH="${ASR_MODEL_PATH:-${WHISPER_EXP_DIR}/best_model}"
if [[ ! -d "${ASR_MODEL_PATH}" && -d "${WHISPER_EXP_DIR}/last_model" ]]; then
  ASR_MODEL_PATH="${WHISPER_EXP_DIR}/last_model"
fi

whisper_resume_args=()
gopt_resume_args=()
if [[ "${AUTO_RESUME}" == "1" && -f "${WHISPER_EXP_DIR}/last_checkpoint.pt" ]]; then
  whisper_resume_args+=(--resume)
fi
if [[ "${AUTO_RESUME}" == "1" && -f "${GOPT_EXP_DIR}/last_checkpoint.pt" ]]; then
  gopt_resume_args+=(--resume)
fi

run_prefix_data() {
  "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_whisper_prefix_data.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --output-dir "${WHISPER_PREFIX_DIR}" \
    --aligner-model "${ALIGNER_MODEL}" \
    --chunk-sec "${CHUNK_SEC}" \
    --right-context-sec "${RIGHT_CONTEXT_SEC}" \
    --overwrite
}

run_whisper_train() {
  mkdir -p "${WHISPER_EXP_DIR}"
  "${PYTHON_BIN}" "${REPO_ROOT}/src/train_streaming_whisper.py" \
    --data-dir "${WHISPER_PREFIX_DIR}" \
    --exp-dir "${WHISPER_EXP_DIR}" \
    --model-name-or-path "${WHISPER_BASE_MODEL}" \
    --language "${LANGUAGE}" \
    --batch-size "${WHISPER_BATCH_SIZE}" \
    --eval-batch-size "${WHISPER_EVAL_BATCH_SIZE}" \
    --n-epochs "${WHISPER_EPOCHS}" \
    "${whisper_resume_args[@]}"
}

run_asr_gopt_data() {
  "${PYTHON_BIN}" "${REPO_ROOT}/src/prep_data/build_streaming_asr_gopt_data.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --output-dir "${ASR_GOPT_DATA_DIR}" \
    --aligner-model "${ALIGNER_MODEL}" \
    --asr-model "${ASR_MODEL_PATH}" \
    --timestamp-backend "${TIMESTAMP_BACKEND}" \
    --language "${LANGUAGE}" \
    --chunk-sec "${CHUNK_SEC}" \
    --right-context-sec "${RIGHT_CONTEXT_SEC}" \
    --min-utt-match-ratio "${MIN_UTT_MATCH_RATIO}" \
    --overwrite
}

run_gopt_train() {
  mkdir -p "${GOPT_EXP_DIR}"
  "${PYTHON_BIN}" "${REPO_ROOT}/src/train_streaming_charsiu.py" \
    --data-dir "${ASR_GOPT_DATA_DIR}" \
    --exp-dir "${GOPT_EXP_DIR}" \
    --depth "${GOPT_DEPTH}" \
    --heads "${GOPT_HEADS}" \
    --batch-size "${GOPT_BATCH_SIZE}" \
    --embed-dim "${GOPT_EMBED_DIM}" \
    --model streaming_gopt \
    --n-epochs "${GOPT_EPOCHS}" \
    --main-context-tokens "${GOPT_MAIN_CONTEXT_TOKENS}" \
    --right-context-tokens "${GOPT_RIGHT_CONTEXT_TOKENS}" \
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
