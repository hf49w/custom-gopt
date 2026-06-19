# Server Deploy

## 1. Push Code To GitHub

Target repo:

- `https://github.com/hf49w/custom-gopt.git`

The server should clone code from GitHub. Do not copy the repo directory manually.

## 2. Files Added For Server Deploy

- `scripts/server/server_paths.env.example`
- `scripts/server/setup_server_env.sh`
- `scripts/server/export_whisper_model.py`
- `scripts/server/upload_assets_to_server.sh`
- `scripts/server/train_on_server.sh`
- `src/run_streaming_whisper_gopt_wsl.sh`

## 3. Local Machine: Prepare Whisper Base Model For Upload

Run from the repo root:

```bash
python scripts/server/export_whisper_model.py \
  --model-name-or-path openai/whisper-base \
  --output-dir /path/to/local_assets/whisper-base
```

## 4. Local Machine: Create Server Env File

```bash
cp scripts/server/server_paths.env.example scripts/server/server_paths.env
```

Fill in at least:

- `SERVER_USER`
- `SERVER_HOST`
- `SERVER_PORT`
- `LOCAL_DATASET_ROOT`
- `LOCAL_WHISPER_BASE_MODEL_DIR`
- `LOCAL_CHARSIU_SRC_DIR`
- `LOCAL_ALIGNER_MODEL_DIR`

## 5. Local Machine: Upload Dataset And Whisper Model

```bash
chmod +x scripts/server/upload_assets_to_server.sh
./scripts/server/upload_assets_to_server.sh scripts/server/server_paths.env
```

This uploads:

- SpeechOcean762 dataset directory to `${SERVER_DATA_ROOT}/speechocean762/speechocean762`
- Whisper base model directory to `${WHISPER_BASE_MODEL_DIR}`
- Official Charsiu repo directory to `${CHARSIU_SRC_DIR}`
- Charsiu frame-classification checkpoint directory to `${ALIGNER_MODEL_DIR}`

## 6. Server: Clone Repo And Build Environment

On the server:

```bash
git clone https://github.com/hf49w/custom-gopt.git ~/work/custom-gopt
cd ~/work/custom-gopt
cp scripts/server/server_paths.env.example scripts/server/server_paths.env
```

Edit `scripts/server/server_paths.env`, then run:

```bash
chmod +x scripts/server/setup_server_env.sh
./scripts/server/setup_server_env.sh scripts/server/server_paths.env
```

If the system disk is small, keep all install-time caches on the project disk:

- set `ENV_MANAGER="conda"`
- set `CONDA_ENV_PREFIX` to a directory under the project disk
- set `TMPDIR`, `PIP_CACHE_DIR`, and `CONDA_PKGS_DIRS` under the project disk

This avoids `No space left on device` during large `torch` downloads.

For official Charsiu mode, also set:

- `CHARSIU_SRC_DIR` to the uploaded Charsiu repo root or its `src/` directory
- `ALIGNER_MODEL_DIR` to the uploaded `charsiu_en_w2v2_tiny_fc_10ms` directory

For lower-memory ASR-driven chunk generation, also set:

- `ASR_BATCH_SIZE="4"`
- `ASR_MIN_BATCH_SIZE="1"`
- `ASR_MAX_NEW_TOKENS="128"`
- `ASR_TORCH_DTYPE="auto"`
- `ASR_USE_CACHE="0"`
- `ASR_EMPTY_CACHE="1"`
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

For multi-GPU server runs, also set:

- `CUDA_VISIBLE_DEVICES="0,1,2,3"` to expose multiple GPUs
- `TRAIN_DEVICE="cuda"` so training uses the visible GPUs
- `GOPT_DATA_MULTI_GPU="1"` to shard `gopt_data` preprocessing across GPUs
- `GOPT_DATA_GPU_IDS="0,1,2,3"` to pick the physical GPUs used by `gopt_data`
- `GOPT_DATA_FINALIZE_DEVICE="cpu"` to aggregate shard results without occupying a GPU

With these settings:

- `train_streaming_whisper.py` uses `nn.DataParallel` across the visible GPUs
- `train_streaming_charsiu.py` already uses `nn.DataParallel`
- `build_streaming_asr_gopt_data.py` is launched as one shard worker per `GOPT_DATA_GPU_IDS` entry, then finalized once after all workers finish

The preprocessing scripts import `Charsiu.charsiu_forced_aligner` from `CHARSIU_SRC_DIR` and no longer rely on a hand-built `AutoProcessor` fallback.

## 7. Server: Train

Run the full pipeline:

```bash
chmod +x scripts/server/train_on_server.sh
./scripts/server/train_on_server.sh scripts/server/server_paths.env all
```

Run a single stage:

```bash
./scripts/server/train_on_server.sh scripts/server/server_paths.env prefix
./scripts/server/train_on_server.sh scripts/server/server_paths.env whisper
./scripts/server/train_on_server.sh scripts/server/server_paths.env gopt_data
./scripts/server/train_on_server.sh scripts/server/server_paths.env gopt
```

The train script automatically enables resume mode.

## 8. Stage Inputs / Outputs

### Prefix data build

Input:

- `${DATASET_ROOT}/train/wav.scp`
- `${DATASET_ROOT}/test/wav.scp`
- `${REPO_DIR}/src/prep_data/scores.json`
- `${ALIGNER_MODEL}` or `${ALIGNER_MODEL_DIR}`

Output:

- `${REPO_DIR}/data/streaming_whisper_prefix/train_prefix.jsonl`
- `${REPO_DIR}/data/streaming_whisper_prefix/test_prefix.jsonl`
- `${REPO_DIR}/data/streaming_whisper_prefix/metadata.json`

### Whisper training

Input:

- `${REPO_DIR}/data/streaming_whisper_prefix/train_prefix.jsonl`
- `${REPO_DIR}/data/streaming_whisper_prefix/test_prefix.jsonl`
- `${WHISPER_BASE_MODEL_DIR}`

Output:

- `${REPO_DIR}/exp/streaming-whisper-base/best_model/`
- `${REPO_DIR}/exp/streaming-whisper-base/last_model/`
- `${REPO_DIR}/exp/streaming-whisper-base/last_checkpoint.pt`

Default epochs:

- `8`

### ASR-driven GOPT data build

Input:

- `${DATASET_ROOT}/train/wav.scp`
- `${DATASET_ROOT}/test/wav.scp`
- `${REPO_DIR}/src/prep_data/scores.json`
- `${REPO_DIR}/exp/streaming-whisper-base/best_model/` or `last_model/`

Output:

- `${REPO_DIR}/data/streaming_asr_gopt/train_chunks.npz`
- `${REPO_DIR}/data/streaming_asr_gopt/test_chunks.npz`
- `${REPO_DIR}/data/streaming_asr_gopt/metadata.json`

### Streaming GOPT training

Input:

- `${REPO_DIR}/data/streaming_asr_gopt/train_chunks.npz`
- `${REPO_DIR}/data/streaming_asr_gopt/test_chunks.npz`

Output:

- `${REPO_DIR}/exp/streaming-asr-gopt/models/best_audio_model.pth`
- `${REPO_DIR}/exp/streaming-asr-gopt/last_checkpoint.pt`
- `${REPO_DIR}/exp/streaming-asr-gopt/result.csv`

Default epochs:

- `100`
