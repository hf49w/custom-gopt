#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/DATA_2/guest/custom-gopt}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/gopt_closed_oracle_prefix_gt_time}"
GPU="${GPU:-0}"

mkdir -p "${OUT_DIR}"
cd "${REPO}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export XDG_CACHE_HOME=/DATA_2/MultiPA/.cache
export CHARSIU_TOKENIZER_EN_CMU=/DATA_2/guest/custom-gopt/server_assets/src/charsiu_repo/local

"${REPO}/.multipa_env/bin/python" scripts/eval_gopt_closed_oracle_prefix.py \
  --prefix-manifest "${REPO}/data/streaming_pcn_gopt_v2_stateful/test_manifest.jsonl" \
  --scores-json "${REPO}/src/prep_data/scores.json" \
  --seq-data-dir "${REPO}/data/seq_data_librispeech" \
  --keys-phn-csv "${REPO}/data/raw_kaldi_gop/librispeech/te_keys_phn.csv" \
  --checkpoint "${REPO}/pretrained_models/gopt_librispeech/best_audio_model.pth" \
  --repo-src "${REPO}/src" \
  --output-jsonl "${OUT_DIR}/predictions.jsonl" \
  --device cpu \
  --batch-size 128 \
  --word-count-source gt_time \
  --time-field audio_end \
  --multipa-repo-root /DATA_2/MultiPA \
  --aligner "${REPO}/server_assets/models/charsiu-en_w2v2_fc_10ms" \
  --align-device cuda:0 \
  --word-time-cache "${OUT_DIR}/gt_word_time_cache"

"${REPO}/.conda_env/bin/python" scripts/summarize_streaming_jsonl_pcc.py \
  --model gopt_closed_oracle_prefix_gt_time="${OUT_DIR}/predictions.jsonl" \
  --output-json "${OUT_DIR}/streaming_pcc_summary.json"

echo "[gopt-closed-gt-time] complete: ${OUT_DIR}"
