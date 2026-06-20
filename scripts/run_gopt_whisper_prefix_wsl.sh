#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

MODE="${1:-all}"
TRANSCRIPT_MODEL="${TRANSCRIPT_MODEL:-openai/whisper-base.en}"
TRANSCRIPT_BACKEND="${TRANSCRIPT_BACKEND:-transformers}"
TRANSCRIPT_DEVICE="${TRANSCRIPT_DEVICE:-cuda:0}"
if [[ -z "${TRANSCRIPT_BATCH_SIZE:-}" ]]; then
  if [[ "${TRANSCRIPT_MODEL}" == *"medium"* ]]; then
    TRANSCRIPT_BATCH_SIZE=2
  else
    TRANSCRIPT_BATCH_SIZE=16
  fi
fi
OPENAI_WHISPER_CACHE="${OPENAI_WHISPER_CACHE:-${HOME}/.cache/whisper}"
TRANSCRIPT_MAX_NEW_TOKENS="${TRANSCRIPT_MAX_NEW_TOKENS:-32}"
TRANSCRIPT_NO_REPEAT_NGRAM_SIZE="${TRANSCRIPT_NO_REPEAT_NGRAM_SIZE:-3}"
PREDICT_DEVICE="${PREDICT_DEVICE:-cpu}"
MULTIPA_ENV="${MULTIPA_ENV:-multipa}"
GOPT_ENV="${GOPT_ENV:-gopt-py38}"
NJ="${NJ:-8}"
KALDI_CMD="${KALDI_CMD:-run.pl}"

PREFIX_MANIFEST="${PREFIX_MANIFEST:-${REPO_ROOT}/downloads/custom-gopt-252/eval/prefix_streaming/shared_test_prefixes.jsonl}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}/speechocean762/speechocean762}"
LEXICON_TXT="${LEXICON_TXT:-${DATASET_ROOT}/resource/lexicon.txt}"
KALDI_GOP_ROOT="${KALDI_GOP_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/gop_speechocean762/s5}"
LIBRISPEECH_EG_ROOT="${LIBRISPEECH_EG_ROOT:-${WORKSPACE_ROOT}/kaldi/egs/librispeech/s5}"
GOPT_CHECKPOINT="${GOPT_CHECKPOINT:-${REPO_ROOT}/pretrained_models/gopt_librispeech/best_audio_model.pth}"

MODEL_TAG="$(echo "${TRANSCRIPT_MODEL}" | tr '/:.' '_')"
RUN_TAG="${RUN_TAG:-gopt_prefix_${MODEL_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/eval/prefix_streaming/${RUN_TAG}}"
PREFIX_AUDIO_ROOT="${PREFIX_AUDIO_ROOT:-${REPO_ROOT}/downloads/custom-gopt-252/eval/prefix_streaming/prefix_audio}"

PREFIX_AUDIO_MANIFEST="${PREFIX_AUDIO_ROOT}/prefix_audio_manifest.jsonl"
TRANSCRIPT_TSV="${OUTPUT_ROOT}/transcripts.tsv"
PREPARED_ROOT="${OUTPUT_ROOT}/prepared_dataset"
SEQ_DIR="${OUTPUT_ROOT}/seq"
OUTPUT_JSONL="${OUTPUT_ROOT}/predictions.jsonl"
FEATURE_SCP="${KALDI_GOP_ROOT}/exp/${RUN_TAG}_gop_${RUN_TAG}_test/feat.scp"

mkdir -p "${OUTPUT_ROOT}" "${SEQ_DIR}"

prepare_audio() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  python "${REPO_ROOT}/scripts/prepare_prefix_audio_manifest.py" \
    --prefix-manifest "${PREFIX_MANIFEST}" \
    --output-root "${PREFIX_AUDIO_ROOT}"
}

transcribe() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  export USE_TF=0
  export TRANSFORMERS_NO_TF=1
  python "${REPO_ROOT}/scripts/export_whisper_open_transcripts.py" \
    --manifest-jsonl "${PREFIX_AUDIO_MANIFEST}" \
    --model "${TRANSCRIPT_MODEL}" \
    --backend "${TRANSCRIPT_BACKEND}" \
    --device "${TRANSCRIPT_DEVICE}" \
    --openai-whisper-cache "${OPENAI_WHISPER_CACHE}" \
    --batch-size "${TRANSCRIPT_BATCH_SIZE}" \
    --max-new-tokens "${TRANSCRIPT_MAX_NEW_TOKENS}" \
    --no-repeat-ngram-size "${TRANSCRIPT_NO_REPEAT_NGRAM_SIZE}" \
    --output-tsv "${TRANSCRIPT_TSV}" \
    --normalize-mode multipa_open \
    --resume
}

prepare_dataset() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${MULTIPA_ENV}"
  python "${REPO_ROOT}/scripts/prepare_gopt_open_dataset.py" \
    --manifest-jsonl "${PREFIX_AUDIO_MANIFEST}" \
    --transcript-tsv "${TRANSCRIPT_TSV}" \
    --lexicon-txt "${LEXICON_TXT}" \
    --max-phones 50 \
    --output-root "${PREPARED_ROOT}"
}

run_kaldi() {
  PREPARED_DATASET_ROOT="${PREPARED_ROOT}" \
  KALDI_GOP_ROOT="${KALDI_GOP_ROOT}" \
  LIBRISPEECH_EG_ROOT="${LIBRISPEECH_EG_ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  NJ="${NJ}" \
  KALDI_CMD="${KALDI_CMD}" \
  bash "${REPO_ROOT}/scripts/run_kaldi_gopt_open_subset_wsl.sh"
}

build_seq() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${GOPT_ENV}"
  python "${REPO_ROOT}/scripts/build_gopt_open_seq_from_kaldi.py" \
    --feature-scp "${FEATURE_SCP}" \
    --pseudo-scores-json "${PREPARED_ROOT}/resource/scores.json" \
    --reference-train-labels-phn-csv "${REPO_ROOT}/data/raw_kaldi_gop/librispeech/tr_labels_phn.csv" \
    --output-dir "${SEQ_DIR}" \
    --prefix te
}

evaluate() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate "${GOPT_ENV}"
  python "${REPO_ROOT}/scripts/eval_gopt_whisper_prefix.py" \
    --prefix-audio-manifest "${PREFIX_AUDIO_MANIFEST}" \
    --transcript-tsv "${TRANSCRIPT_TSV}" \
    --seq-data-dir "${SEQ_DIR}" \
    --checkpoint "${GOPT_CHECKPOINT}" \
    --repo-src "${REPO_ROOT}/src" \
    --asr-model-name "${TRANSCRIPT_MODEL}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --device "${PREDICT_DEVICE}" \
    --batch-size 128
}

case "${MODE}" in
  audio)
    prepare_audio
    ;;
  transcripts)
    prepare_audio
    transcribe
    ;;
  prepare)
    prepare_audio
    transcribe
    prepare_dataset
    ;;
  kaldi)
    prepare_audio
    transcribe
    prepare_dataset
    run_kaldi
    ;;
  seq)
    build_seq
    ;;
  eval)
    evaluate
    ;;
  from_kaldi)
    build_seq
    evaluate
    ;;
  all)
    prepare_audio
    transcribe
    prepare_dataset
    run_kaldi
    build_seq
    evaluate
    ;;
  *)
    echo "Usage: $0 [audio|transcripts|prepare|kaldi|seq|eval|from_kaldi|all]" >&2
    exit 1
    ;;
esac
