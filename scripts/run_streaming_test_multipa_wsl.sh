#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

MODE="${1:-all}"

STREAMING_DATA_ROOT="${STREAMING_DATA_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/data/streaming_asr_gopt_v6_asrconf}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}/speechocean762/speechocean762}"
SCORES_JSON="${SCORES_JSON:-${REPO_ROOT}/src/prep_data/scores.json}"
MULTIPA_REPO_ROOT="${MULTIPA_REPO_ROOT:-${WORKSPACE_ROOT}/MultiPA}"
MULTIPA_ENV="${MULTIPA_ENV:-multipa}"
MULTIPA_CKPTDIR="${MULTIPA_CKPTDIR:-${MULTIPA_REPO_ROOT}/model_assessment}"
FAIRSEQ_BASE_MODEL="${FAIRSEQ_BASE_MODEL:-${MULTIPA_REPO_ROOT}/fairseq_hubert/hubert_base_ls960.pt}"
FAIRSEQ_ROBERTA="${FAIRSEQ_ROBERTA:-${MULTIPA_REPO_ROOT}/fairseq_roberta}"
RUN_NAME="${RUN_NAME:-multipa_open_streaming_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/eval/${RUN_NAME}}"

SUBSET_DIR="${OUTPUT_ROOT}/subset"
PREP_DIR="${OUTPUT_ROOT}/prepared_multipa_subset"
GT_ALIGN_DIR="${OUTPUT_ROOT}/gt_alignment"
PREDICTION_DIR="${OUTPUT_ROOT}/predictions"
MULTIPA_RESULTS_DIR="${MULTIPA_REPO_ROOT}/Results"
DATALIST_NAME="test_streaming_subset.txt"
MULTIPA_OUTFILE="$(basename "${MULTIPA_CKPTDIR}")_${DATALIST_NAME%.txt}_mb.txt"
MULTIPA_OUTPUT_PATH="${MULTIPA_RESULTS_DIR}/${MULTIPA_OUTFILE}"
PREDICTION_PATH="${PREDICTION_DIR}/${MULTIPA_OUTFILE}"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"

mkdir -p "${OUTPUT_ROOT}" "${SUBSET_DIR}" "${PREP_DIR}" "${GT_ALIGN_DIR}" "${PREDICTION_DIR}"

export_subset() {
  python3 "${REPO_ROOT}/scripts/export_streaming_test_subset.py" \
    --streaming-data-root "${STREAMING_DATA_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --dataset-root "${DATASET_ROOT}" \
    --split test \
    --output-dir "${SUBSET_DIR}"
}

prepare_subset() {
  python3 "${REPO_ROOT}/scripts/prepare_multipa_streaming_subset.py" \
    --manifest-jsonl "${SUBSET_DIR}/test_streaming_subset.jsonl" \
    --output-dir "${PREP_DIR}"
}

run_multipa() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  cd "${MULTIPA_REPO_ROOT}"
  python test_open.py \
    --fairseq_base_model "${FAIRSEQ_BASE_MODEL}" \
    --fairseq_roberta "${FAIRSEQ_ROBERTA}" \
    --datadir "${PREP_DIR}/wav_flat" \
    --datalist "${PREP_DIR}/${DATALIST_NAME}" \
    --ckptdir "${MULTIPA_CKPTDIR}"
  cp "${MULTIPA_OUTPUT_PATH}" "${PREDICTION_PATH}"
}

eval_multipa() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  cd "${REPO_ROOT}"
  python "${REPO_ROOT}/scripts/eval_gopt_open_metrics.py" \
    --prediction-path "${PREDICTION_PATH}" \
    --dataset-root "${DATASET_ROOT}" \
    --gt-alignment-dir "${GT_ALIGN_DIR}" \
    --multipa-repo-root "${MULTIPA_REPO_ROOT}" \
    --ensure-gt-alignments \
    --compare-reference multipa \
    --output-json "${SUMMARY_JSON}"
}

case "${MODE}" in
  subset)
    export_subset
    ;;
  prepare)
    export_subset
    prepare_subset
    ;;
  predict)
    export_subset
    prepare_subset
    run_multipa
    ;;
  predict_only)
    run_multipa
    ;;
  eval)
    export_subset
    prepare_subset
    run_multipa
    eval_multipa
    ;;
  eval_only)
    eval_multipa
    ;;
  all)
    export_subset
    prepare_subset
    run_multipa
    eval_multipa
    ;;
  *)
    echo "Usage: $0 [subset|prepare|predict|predict_only|eval|eval_only|all]" >&2
    exit 1
    ;;
esac
