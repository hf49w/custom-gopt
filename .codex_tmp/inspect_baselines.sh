#!/usr/bin/env bash
set -euo pipefail
cd /DATA_2/guest/custom-gopt
files=(
  "downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_open_whisper_base/predictions.phone_word_targets.jsonl"
  "downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_open_whisper_medium/predictions.phone_word_targets.jsonl"
  "downloads/custom-gopt-252/eval/same_manifest_streaming/multipa/multipa.shard_0.jsonl"
  "downloads/custom-gopt-252/eval/same_manifest_streaming/multipa/multipa.shard_1.jsonl"
  "downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_closed_oracle_prefix_gt_time/predictions.jsonl"
  "downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_closed_oracle_prefix_pcn_commit/predictions.jsonl"
  "exp/streaming-asr-gopt-v6-asrconf/result.csv"
  "exp/streaming-asr-gopt/result.csv"
)
for f in "${files[@]}"; do
  echo "==== $f"
  if [[ -f "$f" ]]; then
    wc -l "$f"
    head -1 "$f" | cut -c1-2500
    echo
  else
    echo "missing"
  fi
done
