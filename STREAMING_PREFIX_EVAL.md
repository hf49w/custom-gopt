# Prefix Streaming Evaluation

This protocol reuses the existing v6 `test_manifest.jsonl`. It does not rerun
the completed full-utterance evaluation.

The shared manifest contains 4,669 retained chunks from 1,212 test
utterances. The original test split has 1,240 utterances; 28 utterances have no
retained v6 chunk and are outside this conditional comparison.

## 1. Build The Shared Prefix Manifest

Run once in either WSL environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multipa

python /mnt/d/研究生/智能体/gopt/scripts/build_prefix_eval_manifest.py \
  --streaming-data-root /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/data/streaming_asr_gopt_v6_asrconf \
  --scores-json /mnt/d/研究生/智能体/gopt_charsiu/src/prep_data/scores.json \
  --dataset-root /mnt/d/研究生/智能体/speechocean762/speechocean762 \
  --split test \
  --output-jsonl /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/shared_test_prefixes.jsonl
```

## 2. StreamingGOPT v6

Environment: `multipa`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multipa

python /mnt/d/研究生/智能体/gopt_charsiu/scripts/local/eval_streaming_gopt_prefix.py \
  --prefix-manifest /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/shared_test_prefixes.jsonl \
  --data-root /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/data/streaming_asr_gopt_v6_asrconf \
  --model-dir /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/model_v6 \
  --repo-src /mnt/d/研究生/智能体/gopt_charsiu/src \
  --output-jsonl /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/streaming_gopt_v6.jsonl \
  --device cuda \
  --batch-size 128 \
  --main-context-tokens 8 \
  --right-context-tokens 2
```

This is the native streaming-aware model and directly consumes the saved v6
chunk features.

## 3. Original GOPT

Environment: `gopt-py38`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gopt-py38

python /mnt/d/研究生/智能体/gopt/scripts/eval_original_gopt_prefix.py \
  --prefix-manifest /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/shared_test_prefixes.jsonl \
  --seq-data-dir /mnt/d/研究生/智能体/gopt/data/seq_data_librispeech \
  --keys-phn-csv /mnt/d/研究生/智能体/gopt/data/raw_kaldi_gop/librispeech/te_keys_phn.csv \
  --checkpoint /mnt/d/研究生/智能体/gopt/pretrained_models/gopt_librispeech/best_audio_model.pth \
  --repo-src /mnt/d/研究生/智能体/gopt/src \
  --output-jsonl /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/original_gopt.jsonl \
  --device cpu \
  --batch-size 128
```

The original checkpoint requires 84-dimensional Kaldi GOP features. The
program directly runs that checkpoint after truncating each completed
test-utterance feature sequence to the current v6 `visible_phone_count`, which
matches the number of phone tokens consumed by StreamingGOPT at that prefix.
It is marked `oracle_reference_phone_prefix`: it measures prefix sensitivity
with the reference phone order and full-test Kaldi features, not robustness to
ASR phone substitutions.

## 4. MultiPA

Environment: `multipa`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multipa
cd /mnt/d/研究生/智能体/MultiPA

python eval_multipa_prefix.py \
  --prefix-manifest /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/shared_test_prefixes.jsonl \
  --output-jsonl /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/multipa.jsonl \
  --fairseq-base-model /mnt/d/研究生/智能体/MultiPA/fairseq_hubert/hubert_base_ls960.pt \
  --fairseq-roberta /mnt/d/研究生/智能体/MultiPA/fairseq_roberta \
  --ckptdir /mnt/d/研究生/智能体/MultiPA/model_assessment \
  --resume
```

MultiPA is recomputed from the raw waveform cut at each `audio_end`. It is much
slower than the two GOPT programs. The JSONL is flushed after every chunk and
`--resume` skips completed `(utt_id, chunk_id)` pairs. Add `--device cpu` if
the GPU does not have enough memory for both Whisper models and MultiPA.

## 5. Summarize

```bash
conda activate multipa

python /mnt/d/研究生/智能体/gopt/scripts/summarize_prefix_streaming_eval.py \
  --input-jsonl \
    /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/streaming_gopt_v6.jsonl \
    /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/original_gopt.jsonl \
    /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/multipa.jsonl \
  --output-json /mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/summary.json
```

The summary reports PCC/MSE/MAE at 25%, 50%, 75%, and 100% progress, coverage
at every progress point, adjacent-prefix score changes, convergence progress,
and inference-time percentiles. For a progress point, it selects the latest
available retained chunk not exceeding that point.

The example GOPT commands use batching for fast score evaluation, so their
recorded timing is marked `batch_amortized`. Set `--batch-size 1` for a separate
online-latency run before comparing latency against MultiPA.

## 6. GOPT With Whisper Text

This variant cuts the same 4,669 audio prefixes, transcribes every prefix with
Whisper, treats that ASR transcript as the canonical text used to build Kaldi
GOP features, and then runs the original GOPT checkpoint.

Whisper base:

```bash
cd /mnt/d/研究生/智能体/gopt
TRANSCRIPT_MODEL=openai/whisper-base.en \
TRANSCRIPT_BACKEND=transformers \
TRANSCRIPT_BATCH_SIZE=16 \
TRANSCRIPT_DEVICE=cuda:0 \
RUN_TAG=gopt_prefix_whisper_base_en \
OUTPUT_ROOT=/mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/gopt_whisper_base \
bash scripts/run_gopt_whisper_prefix_wsl.sh all
```

Whisper medium:

```bash
cd /mnt/d/研究生/智能体/gopt
TRANSCRIPT_MODEL=openai/whisper-medium.en \
TRANSCRIPT_BACKEND=transformers \
TRANSCRIPT_BATCH_SIZE=2 \
TRANSCRIPT_DEVICE=cuda:0 \
RUN_TAG=gopt_prefix_whisper_medium_en \
OUTPUT_ROOT=/mnt/d/研究生/智能体/gopt/downloads/custom-gopt-252/eval/prefix_streaming/gopt_whisper_medium \
bash scripts/run_gopt_whisper_prefix_wsl.sh all
```

Both transcript stages support resume. Rerun the same command after interruption;
existing transcript rows and prefix WAV files are reused. The default
`transformers` backend is forced offline and reads the existing Hugging Face
cache. Alternatively, `TRANSCRIPT_BACKEND=openai-whisper` reads
`~/.cache/whisper/base.en.pt` or `~/.cache/whisper/medium.en.pt`. Neither mode
contacts Hugging Face. The outputs are:

```text
.../gopt_whisper_base/predictions.jsonl
.../gopt_whisper_medium/predictions.jsonl
```
