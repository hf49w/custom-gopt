#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/DATA_2/guest/custom-gopt}"
MULTIPA_ROOT="${MULTIPA_ROOT:-/DATA_2/MultiPA}"
EVAL_ROOT="${EVAL_ROOT:-${REPO}/downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming}"
PREFIX_AUDIO_MANIFEST="${PREFIX_AUDIO_MANIFEST:-${EVAL_ROOT}/prefix_audio/prefix_audio_manifest.jsonl}"
PREFIX_MANIFEST="${PREFIX_MANIFEST:-${EVAL_ROOT}/shared_test_prefixes.jsonl}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/multipa/all_chunks_full_4gpu_20260701}"
SCORES_JSON="${SCORES_JSON:-${REPO}/src/prep_data/scores.json}"
ALIGNER="${ALIGNER:-${REPO}/server_assets/models/charsiu-en_w2v2_fc_10ms}"
PY="${PY:-${REPO}/.multipa_env/bin/python}"
GPUS_TEXT="${GPUS_TEXT:-0 1 2 7}"

read -r -a GPUS <<< "${GPUS_TEXT}"

mkdir -p "${OUT_DIR}/shards" "${OUT_DIR}/logs"
cd "${REPO}"
for i in "${!GPUS[@]}"; do
  "${REPO}/.conda_env/bin/python" scripts/shard_jsonl.py \
    --input-jsonl "${PREFIX_AUDIO_MANIFEST}" \
    --output-jsonl "${OUT_DIR}/shards/prefix_audio_manifest.shard_${i}.jsonl" \
    --num-shards "${#GPUS[@]}" \
    --shard-index "${i}"
done

cd "${MULTIPA_ROOT}"
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XDG_CACHE_HOME=/DATA_2/MultiPA/.cache
    export CHARSU_TOKENIZER_EN_CMU=/DATA_2/guest/custom-gopt/server_assets/src/charsiu_repo/local
    export CHARSIU_TOKENIZER_EN_CMU=/DATA_2/guest/custom-gopt/server_assets/src/charsiu_repo/local
    export HF_HOME=/DATA_2/guest/custom-gopt/server_assets/hf_home
    export TRANSFORMERS_CACHE=/DATA_2/guest/custom-gopt/server_assets/hf_home/transformers
    "${PY}" eval_multipa_prefix.py \
      --prefix-manifest "${OUT_DIR}/shards/prefix_audio_manifest.shard_${i}.jsonl" \
      --output-jsonl "${OUT_DIR}/multipa.shard_${i}.jsonl" \
      --resume \
      --fairseq-base-model "${MULTIPA_ROOT}/fairseq_hubert/hubert_base_ls960.pt" \
      --fairseq-roberta "${MULTIPA_ROOT}/fairseq_roberta" \
      --ckptdir "${MULTIPA_ROOT}/model_assessment" \
      --aligner-model "${ALIGNER}" \
      --whisper-sentence-model medium.en \
      --whisper-word-model base.en \
      --local-model-cache /DATA_2/guest/custom-gopt/server_assets/multipa_model_cache
  ) > "${OUT_DIR}/logs/shard_${i}.gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != 0 ]]; then
  echo "[multipa] one or more shards failed" >&2
  exit 1
fi

cat "${OUT_DIR}"/multipa.shard_*.jsonl > "${OUT_DIR}/multipa.raw.jsonl"
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=7
export XDG_CACHE_HOME=/DATA_2/MultiPA/.cache
"${PY}" scripts/attach_prefix_word_targets.py \
  --prediction-jsonl "${OUT_DIR}/multipa.raw.jsonl" \
  --output-jsonl "${OUT_DIR}/multipa.word_targets.jsonl" \
  --scores-json "${SCORES_JSON}" \
  --prefix-manifest "${PREFIX_MANIFEST}" \
  --prefix-audio-manifest "${PREFIX_AUDIO_MANIFEST}" \
  --multipa-repo-root "${MULTIPA_ROOT}" \
  --aligner "${ALIGNER}" \
  --align-device cuda:0 \
  --resume \
  --cache-dir "${OUT_DIR}/charsiu_word_time_cache"

"${REPO}/.conda_env/bin/python" scripts/summarize_streaming_jsonl_pcc.py \
  --model multipa="${OUT_DIR}/multipa.word_targets.jsonl" \
  --output-json "${OUT_DIR}/multipa_streaming_pcc_summary.json"

echo "[multipa] complete: ${OUT_DIR}"
