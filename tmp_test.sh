cd /DATA_2/guest/custom-gopt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /DATA_2/guest/custom-gopt/.conda_env
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/DATA_2/guest/custom-gopt/server_assets/hf_cache
export TRANSFORMERS_CACHE=/DATA_2/guest/custom-gopt/server_assets/hf_cache/transformers
export HF_HUB_CACHE=/DATA_2/guest/custom-gopt/server_assets/hf_cache/hub
export CHARSIU_SRC_DIR=/DATA_2/guest/custom-gopt/server_assets/src/charsiu_repo
export CUDA_VISIBLE_DEVICES=0
rm -rf /DATA_2/guest/custom-gopt/data/test_worker
mkdir -p /DATA_2/guest/custom-gopt/data/test_worker/worker_logs
python src/prep_data/build_streaming_asr_gopt_data.py \
  --dataset-root /DATA_2/guest/custom-gopt/server_assets/speechocean762/speechocean762 \
  --scores-json /DATA_2/guest/custom-gopt/src/prep_data/scores.json \
  --output-dir /DATA_2/guest/custom-gopt/data/test_worker \
  --aligner-model /DATA_2/guest/custom-gopt/server_assets/models/charsiu_en_w2v2_tiny_fc_10ms \
  --asr-model /DATA_2/guest/custom-gopt/exp/streaming-whisper-base/best_model \
  --val-speaker-ratio 0.5 \
  --split-seed 1337 \
  --timestamp-backend transformers \
  --language english \
  --chunk-sec 0.64 \
  --right-context-sec 0.16 \
  --min-utt-match-ratio 0.5 \
  --asr-batch-size 4 \
  --asr-min-batch-size 1 \
  --asr-max-new-tokens 128 \
  --asr-torch-dtype auto \
  --device cuda \
  --num-shards 4 \
  --shard-index 0 \
  --skip-finalize \
  --resume
