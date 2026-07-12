#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/DATA_2/guest/custom-gopt}"
MULTIPA_ROOT="${MULTIPA_ROOT:-/DATA_2/MultiPA}"
DATA_DIR="${DATA_DIR:-$ROOT/data/streaming_pcn_gopt_v2_stateful}"
EXP_DIR="${EXP_DIR:-$ROOT/exp/streaming-pcn-gopt-v2-stateful-teacher}"
CUSTOM_PY="${CUSTOM_PY:-$ROOT/.conda_env/bin/python}"
MULTIPA_PY="${MULTIPA_PY:-$ROOT/.multipa_env/bin/python}"
TEACHER_GPUS="${TEACHER_GPUS:-2,4,5}"
TRAIN_GPU="${TRAIN_GPU:-6}"
OLD_SUPERVISOR_PID="${OLD_SUPERVISOR_PID:-}"
POLL_SEC="${POLL_SEC:-60}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-2500}"
EXPECTED_VAL="${EXPECTED_VAL:-1260}"
EXPECTED_TEST="${EXPECTED_TEST:-1240}"

ASSET_ROOT="$ROOT/server_assets"
TEACHER_DIR="$DATA_DIR/teacher_multipa"
TEACHER_MANIFEST="$TEACHER_DIR/train_val_manifest.jsonl"
TEACHER_JSONL="$TEACHER_DIR/multipa_train_val.jsonl"

export TMPDIR="$ASSET_ROOT/tmp"
export PIP_CACHE_DIR="$ASSET_ROOT/pip_cache"
export CONDA_PKGS_DIRS="$ASSET_ROOT/conda_pkgs"
export HF_HOME="$ASSET_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$ASSET_ROOT/cache"
export TORCH_HOME="$ASSET_ROOT/torch_cache"
export NLTK_DATA="$ASSET_ROOT/nltk_data"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CHARSIU_TOKENIZER="$MULTIPA_ROOT/charsiu_tokenizer_en_cmu"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$NLTK_DATA" "$TEACHER_DIR"

if [ "$(readlink -f "$MULTIPA_ROOT")" = "/DATA_2/guest/MultiPA_pic" ]; then
  echo "/DATA_2/guest/MultiPA_pic is not the original MultiPA repository; use /DATA_2/MultiPA." >&2
  exit 2
fi
for required_path in \
  "$MULTIPA_ROOT/eval_multipa_prefix.py" \
  "$MULTIPA_ROOT/fairseq_hubert/hubert_base_ls960.pt" \
  "$MULTIPA_ROOT/fairseq_roberta" \
  "$MULTIPA_ROOT/model_assessment"; do
  if [ ! -e "$required_path" ]; then
    echo "Missing required MultiPA asset: $required_path" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPU_LIST <<< "$TEACHER_GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "TEACHER_GPUS must contain at least one GPU index" >&2
  exit 2
fi
for gpu in "${GPU_LIST[@]}" "$TRAIN_GPU"; do
  if [ "$gpu" = "3" ]; then
    echo "GPU3 is disabled for this pipeline" >&2
    exit 2
  fi
done

progress_count() {
  local split="$1"
  if [ ! -d "$DATA_DIR/progress/$split" ]; then
    echo 0
    return
  fi
  find "$DATA_DIR/progress/$split" -maxdepth 1 -type f -name '*.pkl' 2>/dev/null | wc -l
}

echo "[$(date '+%F %T')] waiting for existing PCN generation; no generator is started or stopped"
while true; do
  train_count="$(progress_count train)"
  val_count="$(progress_count val)"
  test_count="$(progress_count test)"
  generator_count="$(pgrep -fc '[b]uild_streaming_pcn_gopt_data.py' || true)"
  echo "[$(date '+%F %T')] progress train=$train_count/$EXPECTED_TRAIN val=$val_count/$EXPECTED_VAL test=$test_count/$EXPECTED_TEST generators=$generator_count"
  if [ "$train_count" -ge "$EXPECTED_TRAIN" ] && \
     [ "$val_count" -ge "$EXPECTED_VAL" ] && \
     [ "$test_count" -ge "$EXPECTED_TEST" ] && \
     [ "$generator_count" -eq 0 ]; then
    break
  fi
  sleep "$POLL_SEC"
done

if [ -n "$OLD_SUPERVISOR_PID" ] && kill -0 "$OLD_SUPERVISOR_PID" 2>/dev/null; then
  echo "[$(date '+%F %T')] terminating stopped legacy supervisor pid=$OLD_SUPERVISOR_PID"
  kill -KILL "$OLD_SUPERVISOR_PID" || true
fi

cd "$ROOT"
set -a
source scripts/server/server_paths.env
set +a

echo "[$(date '+%F %T')] finalizing PCN arrays"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONPATH=src/prep_data "$CUSTOM_PY" \
  src/prep_data/build_streaming_pcn_gopt_data.py \
  --dataset-root "$DATASET_ROOT" \
  --scores-json src/prep_data/scores.json \
  --output-dir "$DATA_DIR" \
  --target-splits train,val,test \
  --aligner-model "$ALIGNER_MODEL_DIR" \
  --charsiu-src-dir "$CHARSIU_SRC_DIR" \
  --asr-model exp/streaming-whisper-base/best_model \
  --language english \
  --nbest 5 \
  --beam-size 5 \
  --asr-max-new-tokens 64 \
  --asr-no-repeat-ngram-size 3 \
  --chunk-sec "$CHUNK_SEC" \
  --right-context-sec "$RIGHT_CONTEXT_SEC" \
  --device cuda \
  --resume \
  --finalize-only

cat "$DATA_DIR/train_manifest.jsonl" "$DATA_DIR/val_manifest.jsonl" > "$TEACHER_MANIFEST"
num_shards="${#GPU_LIST[@]}"
teacher_pids=()
echo "[$(date '+%F %T')] MultiPA teacher export start shards=$num_shards gpus=$TEACHER_GPUS"
for shard_index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard_index]}"
  shard_jsonl="$TEACHER_DIR/multipa_train_val.shard_${shard_index}.jsonl"
  shard_log="$TEACHER_DIR/multipa_train_val.shard_${shard_index}.log"
  (
    cd "$MULTIPA_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$MULTIPA_PY" eval_multipa_prefix.py \
      --prefix-manifest "$TEACHER_MANIFEST" \
      --output-jsonl "$shard_jsonl" \
      --resume \
      --num-shards "$num_shards" \
      --shard-index "$shard_index" \
      --fairseq-base-model "$MULTIPA_ROOT/fairseq_hubert/hubert_base_ls960.pt" \
      --fairseq-roberta "$MULTIPA_ROOT/fairseq_roberta" \
      --ckptdir "$MULTIPA_ROOT/model_assessment" \
      --aligner-model "$ASSET_ROOT/models/charsiu-en_w2v2_fc_10ms" \
      --whisper-sentence-model /DATA_2/guest/custom-whisper/data/models/whisper/medium.en.pt \
      --whisper-word-model /DATA_2/guest/custom-whisper/data/models/whisper/base.en.pt \
      --local-model-cache "$ASSET_ROOT/multipa_model_cache" \
      > "$shard_log" 2>&1
  ) &
  teacher_pids+=("$!")
  echo "[$(date '+%F %T')] teacher shard=$shard_index gpu=$gpu pid=${teacher_pids[-1]} log=$shard_log"
done

teacher_failed=0
for pid in "${teacher_pids[@]}"; do
  if ! wait "$pid"; then
    teacher_failed=1
  fi
done
if [ "$teacher_failed" -ne 0 ]; then
  echo "One or more MultiPA teacher shards failed; rerun this script to resume JSONL export" >&2
  exit 3
fi

: > "$TEACHER_JSONL"
for shard_index in "${!GPU_LIST[@]}"; do
  cat "$TEACHER_DIR/multipa_train_val.shard_${shard_index}.jsonl" >> "$TEACHER_JSONL"
done

"$MULTIPA_PY" - "$TEACHER_JSONL" <<'PY'
import json
import sys
from collections import Counter

counts = Counter()
with open(sys.argv[1], encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            counts[json.loads(line).get('status', 'unknown')] += 1
print('[teacher-summary]', dict(counts), flush=True)
if counts['ok'] == 0:
    raise SystemExit('teacher export produced no valid rows')
PY

echo "[$(date '+%F %T')] injecting teacher targets into train/val NPZ"
"$CUSTOM_PY" scripts/local/inject_multipa_teacher_pcn.py \
  --data-dir "$DATA_DIR" \
  --teacher-jsonl "$TEACHER_JSONL" \
  --splits train,val

echo "[$(date '+%F %T')] teacher-guided PCN training start exp=$EXP_DIR gpu=$TRAIN_GPU"
resume_args=()
if [ -e "$EXP_DIR/checkpoint_last.pth" ]; then
  resume_args+=(--resume)
fi
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONPATH=src "$CUSTOM_PY" src/train_streaming_pcn.py \
  --data-dir "$DATA_DIR" \
  --exp-dir "$EXP_DIR" \
  --embed-dim 40 \
  --depth 2 \
  --heads 2 \
  --gru-dim 32 \
  --batch-size 16 \
  --num-workers 4 \
  --n-epochs 80 \
  --tbptt-steps 0 \
  --loss-w-teacher-score 0.5 \
  --loss-w-prefix-kd 0.5 \
  --loss-w-rank 0.1 \
  --loss-w-state-projection 0 \
  --device cuda \
  --tf32 \
  "${resume_args[@]}"
echo "[$(date '+%F %T')] teacher-guided PCN training and test complete"
