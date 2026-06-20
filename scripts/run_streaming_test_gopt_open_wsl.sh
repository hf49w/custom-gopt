#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

MODE="${1:-all}"

STREAMING_DATA_ROOT="${STREAMING_DATA_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/data/streaming_asr_gopt_v6_asrconf}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}/speechocean762/speechocean762}"
SCORES_JSON="${SCORES_JSON:-${REPO_ROOT}/src/prep_data/scores.json}"
LEXICON_TXT="${LEXICON_TXT:-${DATASET_ROOT}/resource/lexicon.txt}"
TRANSCRIPT_MODEL="${TRANSCRIPT_MODEL:-openai/whisper-medium.en}"
TRANSCRIPT_DEVICE="${TRANSCRIPT_DEVICE:-cuda:0}"
TRANSCRIPT_TSV="${TRANSCRIPT_TSV:-}"
GOPT_ENV="${GOPT_ENV:-gopt-py38}"
MULTIPA_ENV="${MULTIPA_ENV:-multipa}"
MULTIPA_REPO_ROOT="${MULTIPA_REPO_ROOT:-${WORKSPACE_ROOT}/MultiPA_prompt_whisper}"
KALDI_GOP_ROOT="${KALDI_GOP_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/gop_speechocean762/s5}"
LIBRISPEECH_EG_ROOT="${LIBRISPEECH_EG_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/librispeech/s5}"
GOPT_CHECKPOINT="${GOPT_CHECKPOINT:-${REPO_ROOT}/pretrained_models/gopt_librispeech/best_audio_model.pth}"
KALDI_CMD="${KALDI_CMD:-run.pl}"
NJ="${NJ:-8}"
PREDICT_DEVICE="${PREDICT_DEVICE:-cpu}"
ALIGN_DEVICE="${ALIGN_DEVICE:-}"

MODEL_TAG="$(echo "${TRANSCRIPT_MODEL}" | tr '/:.' '_')"
RUN_NAME="${RUN_NAME:-gopt_open_${MODEL_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/eval/${RUN_NAME}}"

SUBSET_DIR="${OUTPUT_ROOT}/subset"
PREP_DIR="${OUTPUT_ROOT}/prepared_dataset"
SEQ_DIR="${OUTPUT_ROOT}/seq"
PRED_DIR="${OUTPUT_ROOT}/predictions"
GT_ALIGN_DIR="${OUTPUT_ROOT}/gt_alignment"

TRANSCRIPT_OUT="${OUTPUT_ROOT}/transcripts.tsv"
PREDICTION_PATH="${PRED_DIR}/${RUN_NAME}_test_mb.txt"
PREDICT_SUMMARY_JSON="${PRED_DIR}/${RUN_NAME}_predict_summary.json"
EVAL_JSON="${OUTPUT_ROOT}/summary.json"

mkdir -p "${OUTPUT_ROOT}" "${SUBSET_DIR}" "${SEQ_DIR}" "${PRED_DIR}" "${GT_ALIGN_DIR}"

export_subset() {
  python3 "${REPO_ROOT}/scripts/export_streaming_test_subset.py" \
    --streaming-data-root "${STREAMING_DATA_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --dataset-root "${DATASET_ROOT}" \
    --split test \
    --output-dir "${SUBSET_DIR}"
}

export_transcripts() {
  if [[ -n "${TRANSCRIPT_TSV}" ]]; then
    cp "${TRANSCRIPT_TSV}" "${TRANSCRIPT_OUT}"
    return
  fi
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  export USE_TF=0
  export TRANSFORMERS_NO_TF=1
  python "${REPO_ROOT}/scripts/export_whisper_open_transcripts.py" \
    --manifest-jsonl "${SUBSET_DIR}/test_streaming_subset.jsonl" \
    --dataset-root "${DATASET_ROOT}" \
    --model "${TRANSCRIPT_MODEL}" \
    --device "${TRANSCRIPT_DEVICE}" \
    --output-tsv "${TRANSCRIPT_OUT}" \
    --normalize-mode multipa_open
}

prepare_dataset() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  export USE_TF=0
  export TRANSFORMERS_NO_TF=1
  python "${REPO_ROOT}/scripts/prepare_gopt_open_dataset.py" \
    --manifest-jsonl "${SUBSET_DIR}/test_streaming_subset.jsonl" \
    --transcript-tsv "${TRANSCRIPT_OUT}" \
    --lexicon-txt "${LEXICON_TXT}" \
    --reference-text-json "${SCORES_JSON}" \
    --output-root "${PREP_DIR}"
}

run_kaldi() {
  PREPARED_DATASET_ROOT="${PREP_DIR}" \
  KALDI_GOP_ROOT="${KALDI_GOP_ROOT}" \
  LIBRISPEECH_EG_ROOT="${LIBRISPEECH_EG_ROOT}" \
  RUN_TAG="${RUN_NAME}" \
  NJ="${NJ}" \
  KALDI_CMD="${KALDI_CMD}" \
  bash "${REPO_ROOT}/scripts/run_kaldi_gopt_open_subset_wsl.sh"
}

build_seq() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${GOPT_ENV}"
  python "${REPO_ROOT}/scripts/build_gopt_open_seq_from_kaldi.py" \
    --feature-scp "${KALDI_GOP_ROOT}/exp/${RUN_NAME}_gop_${RUN_NAME}_test/feat.scp" \
    --pseudo-scores-json "${PREP_DIR}/resource/scores.json" \
    --reference-train-labels-phn-csv "${REPO_ROOT}/data/raw_kaldi_gop/librispeech/tr_labels_phn.csv" \
    --output-dir "${SEQ_DIR}" \
    --prefix te
}

predict_scores() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  export USE_TF=0
  export TRANSFORMERS_NO_TF=1
  python "${REPO_ROOT}/scripts/predict_gopt_open.py" \
    --seq-data-dir "${SEQ_DIR}" \
    --manifest-jsonl "${SUBSET_DIR}/test_streaming_subset.jsonl" \
    --transcript-tsv "${TRANSCRIPT_OUT}" \
    --checkpoint "${GOPT_CHECKPOINT}" \
    --device "${PREDICT_DEVICE}" \
    --multipa-repo-root "${MULTIPA_REPO_ROOT}" \
    --align-device "${ALIGN_DEVICE}" \
    --invalid-utt-json "${PREP_DIR}/skipped_manifest.json" \
    --output-path "${PREDICTION_PATH}" \
    --summary-json "${PREDICT_SUMMARY_JSON}"
}

eval_scores() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  export USE_TF=0
  export TRANSFORMERS_NO_TF=1
  python "${REPO_ROOT}/scripts/eval_gopt_open_metrics.py" \
    --prediction-path "${PREDICTION_PATH}" \
    --dataset-root "${DATASET_ROOT}" \
    --gt-alignment-dir "${GT_ALIGN_DIR}" \
    --multipa-repo-root "${MULTIPA_REPO_ROOT}" \
    --align-device "${ALIGN_DEVICE}" \
    --ensure-gt-alignments \
    --compare-reference multipa_gopt_open \
    --output-json "${EVAL_JSON}"
}

case "${MODE}" in
  subset)
    export_subset
    ;;
  transcripts)
    export_subset
    export_transcripts
    ;;
  prepare)
    export_subset
    export_transcripts
    prepare_dataset
    ;;
  kaldi)
    export_subset
    export_transcripts
    prepare_dataset
    run_kaldi
    ;;
  seq)
    export_subset
    export_transcripts
    prepare_dataset
    run_kaldi
    build_seq
    ;;
  seq_only)
    build_seq
    ;;
  predict)
    export_subset
    export_transcripts
    prepare_dataset
    run_kaldi
    build_seq
    predict_scores
    ;;
  predict_only)
    predict_scores
    ;;
  eval)
    export_subset
    export_transcripts
    prepare_dataset
    run_kaldi
    build_seq
    predict_scores
    eval_scores
    ;;
  eval_only)
    eval_scores
    ;;
  from_seq)
    predict_scores
    eval_scores
    ;;
  all)
    export_subset
    export_transcripts
    prepare_dataset
    run_kaldi
    build_seq
    predict_scores
    eval_scores
    ;;
  *)
    echo "Usage: $0 [subset|transcripts|prepare|kaldi|seq|seq_only|predict|predict_only|eval|eval_only|from_seq|all]" >&2
    exit 1
    ;;
esac
