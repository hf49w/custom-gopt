#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

MODE="${1:-all}"

STREAMING_DATA_ROOT="${STREAMING_DATA_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/data/streaming_asr_gopt_v6_asrconf}"
SCORES_JSON="${SCORES_JSON:-${REPO_ROOT}/src/prep_data/scores.json}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}/speechocean762/speechocean762}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/eval/gopt_streaming_test_baselines}"

SEQ_DATA_DIR="${SEQ_DATA_DIR:-${REPO_ROOT}/data/seq_data_librispeech}"
KEYS_PHN_CSV="${KEYS_PHN_CSV:-${REPO_ROOT}/data/raw_kaldi_gop/librispeech/te_keys_phn.csv}"
GOPT_CHECKPOINT="${GOPT_CHECKPOINT:-${REPO_ROOT}/pretrained_models/gopt_librispeech/best_audio_model.pth}"
GOPT_AM="${GOPT_AM:-librispeech}"

GOPT_ENV="${GOPT_ENV:-gopt-py38}"
WHISPER_ENV="${WHISPER_ENV:-multipa}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-medium.en}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda:0}"

SUBSET_DIR="${OUTPUT_ROOT}/subset"
CLOSED_DIR="${OUTPUT_ROOT}/gopt_closed"
OPEN_PREP_DIR="${OUTPUT_ROOT}/gopt_open_prep"

if [[ ! -f "${SCORES_JSON}" ]]; then
  echo "Missing SCORES_JSON: ${SCORES_JSON}" >&2
  exit 1
fi

if [[ ! -d "${STREAMING_DATA_ROOT}" ]]; then
  echo "Missing STREAMING_DATA_ROOT: ${STREAMING_DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${SUBSET_DIR}" "${CLOSED_DIR}" "${OPEN_PREP_DIR}"

export_subset() {
  python3 "${REPO_ROOT}/scripts/export_streaming_test_subset.py" \
    --streaming-data-root "${STREAMING_DATA_ROOT}" \
    --scores-json "${SCORES_JSON}" \
    --dataset-root "${DATASET_ROOT}" \
    --split test \
    --output-dir "${SUBSET_DIR}"
}

run_closed_gopt() {
  python3 "${REPO_ROOT}/scripts/build_gopt_seq_subset.py" \
    --utt-id-list "${SUBSET_DIR}/test_utt_ids.txt" \
    --seq-data-dir "${SEQ_DATA_DIR}" \
    --keys-phn-csv "${KEYS_PHN_CSV}" \
    --prefix te \
    --output-dir "${CLOSED_DIR}/seq_subset"

  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${GOPT_ENV}"
  python "${REPO_ROOT}/scripts/eval_gopt_paper_metrics.py" \
    --seq-data-dir "${CLOSED_DIR}/seq_subset" \
    --checkpoint "${GOPT_CHECKPOINT}" \
    --am "${GOPT_AM}" \
    --device cpu \
    --prefix te \
    --output-json "${CLOSED_DIR}/gopt_closed_metrics.json" \
    --compare-reference gopt_readme_librispeech
}

prepare_open_baseline() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${WHISPER_ENV}"
  python "${REPO_ROOT}/scripts/export_whisper_open_transcripts.py" \
    --manifest-jsonl "${SUBSET_DIR}/test_streaming_subset.jsonl" \
    --model "${WHISPER_MODEL}" \
    --device "${WHISPER_DEVICE}" \
    --output-tsv "${OPEN_PREP_DIR}/whisper_medium_en_test.tsv"

  cat <<EOF

[NEXT STEP FOR GOPT-open]
The exact MultiPA-paper GOPT-open baseline still needs GOP feature extraction
from the Whisper transcripts above. This repo does not include a one-shot
end-to-end GOPT-open reproduction bridge.

Prepared files:
  subset manifest : ${SUBSET_DIR}/test_streaming_subset.jsonl
  utterance ids   : ${SUBSET_DIR}/test_utt_ids.txt
  Whisper TSV     : ${OPEN_PREP_DIR}/whisper_medium_en_test.tsv

Use these with your local Kaldi GOP pipeline and the workflow documented in:
  ${REPO_ROOT}/steps_of_inference.md

Remaining work for an exact GOPT-open reproduction:
  1. Build open-set GOP features from the Whisper transcripts.
  2. Run the original GOPT checkpoint (${GOPT_CHECKPOINT}) on those features.
  3. Convert the predictions into an open-eval record format with ASR words.
  4. Score them with MultiPA's open-response evaluation protocol.
EOF
}

case "${MODE}" in
  subset)
    export_subset
    ;;
  closed)
    export_subset
    run_closed_gopt
    ;;
  open_prepare)
    export_subset
    prepare_open_baseline
    ;;
  all)
    export_subset
    run_closed_gopt
    prepare_open_baseline
    ;;
  *)
    echo "Usage: $0 [subset|closed|open_prepare|all]" >&2
    exit 1
    ;;
esac
