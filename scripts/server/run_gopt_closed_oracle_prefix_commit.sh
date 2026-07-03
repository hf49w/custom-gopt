#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/DATA_2/guest/custom-gopt}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/gopt_closed_oracle_prefix_pcn_commit}"

mkdir -p "${OUT_DIR}"
cd "${REPO}"

"${REPO}/.conda_env/bin/python" scripts/eval_gopt_closed_oracle_prefix.py \
  --prefix-manifest "${REPO}/data/streaming_pcn_gopt_v2_stateful/test_manifest.jsonl" \
  --scores-json "${REPO}/src/prep_data/scores.json" \
  --seq-data-dir "${REPO}/data/seq_data_librispeech" \
  --keys-phn-csv "${REPO}/data/raw_kaldi_gop/librispeech/te_keys_phn.csv" \
  --checkpoint "${REPO}/pretrained_models/gopt_librispeech/best_audio_model.pth" \
  --repo-src "${REPO}/src" \
  --output-jsonl "${OUT_DIR}/predictions.jsonl" \
  --device cpu \
  --batch-size 256 \
  --word-count-source manifest_field \
  --word-count-field cumulative_committed_word_count

"${REPO}/.conda_env/bin/python" scripts/summarize_streaming_jsonl_pcc.py \
  --model gopt_closed_oracle_prefix_pcn_commit="${OUT_DIR}/predictions.jsonl" \
  --output-json "${OUT_DIR}/streaming_pcc_summary.json"

echo "[gopt-closed] complete: ${OUT_DIR}"
