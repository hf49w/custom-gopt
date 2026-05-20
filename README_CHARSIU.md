# Charsiu Pipeline

This copy replaces the Kaldi GOP front-end with phone-aligned frame features from `charsiu/en_w2v2_tiny_fc_10ms`.

## What changed

- `src/prep_data/build_charsiu_seq_data.py`
  - reads `scores.json` and SpeechOcean762 `wav.scp`
  - loads the Charsiu frame-classification checkpoint from Hugging Face
  - aligns reference phones to frame posteriors with a monotonic phone-level aligner
  - pools frame probabilities into one feature vector per phone
  - writes `tr_feat.npy`, `tr_label_phn.npy`, `tr_label_word.npy`, `tr_label_utt.npy`, and the test equivalents

- `src/train_charsiu.py`
  - trains GOPT directly from the Charsiu-built sequence data
  - infers `feat_dim`, `seq_len`, and `phn_num` from `metadata.json`

- `src/models/gopt.py` and `src/models/baseline.py`
  - now accept variable `input_dim`, `seq_len`, and `phn_num`

- `src/prep_data/build_streaming_charsiu_data.py`
  - aligns each utterance offline with Charsiu
  - expands one utterance into multiple streaming prefix chunks
  - keeps full phone features only up to `commit_time + right_context`
  - creates `phone_loss_mask`, `word_loss_mask`, and `utt_loss_mask` so training only uses committed tokens and final-chunk utterance labels

- `src/models/streaming_gopt.py`
  - puts utterance CLS tokens after the current phone prefix
  - uses block-wise limited-future attention so phone tokens only see past blocks, the current block, and a small right-context budget

- `src/train_streaming_charsiu.py`
  - trains the streaming scorer from chunked prefix data
  - samples different block sizes per batch to simulate dynamic latency settings

- `src/prep_data/build_whisper_prefix_data.py`
  - converts offline gold alignments into prefix-level ASR supervision manifests
  - writes `train_prefix.jsonl` / `test_prefix.jsonl` for Whisper prefix fine-tuning

- `src/train_streaming_whisper.py`
  - fine-tunes Whisper on prefix chunks so the ASR sees the same streaming commit schedule as GOPT

- `src/prep_data/build_streaming_asr_gopt_data.py`
  - decodes each prefix with Whisper
  - converts ASR word hypotheses back to pseudo canonical phone sequences
  - rebuilds streaming GOPT chunk tensors with stricter `phone_loss_mask`, `word_loss_mask`, and `utt_loss_mask`

- `STREAMING_WHISPER_GOPT_PLAN.md`
  - documents the ASR data format, prefix labeling rules, commit strategy, and ASR-conditioned loss masking

- `SERVER_DEPLOY.md`
  - server-side clone, environment setup, asset upload, and training workflow

## Usage

From the repo root:

```powershell
src\run_charsiu.ps1
```

Or run the steps manually:

```powershell
python src\prep_data\build_charsiu_seq_data.py `
  --dataset-root ..\speechocean762\speechocean762 `
  --scores-json src\prep_data\scores.json `
  --output-dir data\seq_data_charsiu_tiny `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --overwrite

python src\train_charsiu.py `
  --data-dir data\seq_data_charsiu_tiny `
  --exp-dir exp\charsiu-tiny-gopt `
  --goptdepth 3 `
  --goptheads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model gopt
```

## Streaming Usage

From the repo root:

```powershell
src\run_streaming_charsiu.ps1
```

Or run the steps manually:

```powershell
python src\prep_data\build_streaming_charsiu_data.py `
  --dataset-root ..\speechocean762\speechocean762 `
  --scores-json src\prep_data\scores.json `
  --output-dir data\streaming_charsiu_tiny `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --overwrite

python src\train_streaming_charsiu.py `
  --data-dir data\streaming_charsiu_tiny `
  --exp-dir exp\streaming-charsiu-gopt `
  --depth 3 `
  --heads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model streaming_gopt `
  --main-context-tokens 4,8,12,16 `
  --right-context-tokens 0,1,2,4
```

## Streaming Whisper + GOPT Usage

This is the recommended path for weak-reference streaming scoring:

```powershell
src\run_streaming_whisper_gopt.ps1
```

WSL / Linux:

```bash
chmod +x src/run_streaming_whisper_gopt_wsl.sh
./src/run_streaming_whisper_gopt_wsl.sh all
```

Resume after interruption:

```bash
AUTO_RESUME=1 ./src/run_streaming_whisper_gopt_wsl.sh whisper
AUTO_RESUME=1 ./src/run_streaming_whisper_gopt_wsl.sh gopt
```

Manual steps:

```powershell
python src\prep_data\build_whisper_prefix_data.py `
  --dataset-root ..\speechocean762\speechocean762 `
  --scores-json src\prep_data\scores.json `
  --output-dir data\streaming_whisper_prefix `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --overwrite

python src\train_streaming_whisper.py `
  --data-dir data\streaming_whisper_prefix `
  --exp-dir exp\streaming-whisper-base `
  --model-name-or-path openai/whisper-base `
  --language english `
  --batch-size 8 `
  --eval-batch-size 8 `
  --n-epochs 8

python src\prep_data\build_streaming_asr_gopt_data.py `
  --dataset-root ..\speechocean762\speechocean762 `
  --scores-json src\prep_data\scores.json `
  --output-dir data\streaming_asr_gopt `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --asr-model exp\streaming-whisper-base\best_model `
  --timestamp-backend transformers `
  --language english `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --min-utt-match-ratio 0.5 `
  --overwrite

python src\train_streaming_charsiu.py `
  --data-dir data\streaming_asr_gopt `
  --exp-dir exp\streaming-asr-gopt `
  --depth 3 `
  --heads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model streaming_gopt `
  --main-context-tokens 4,8,12,16 `
  --right-context-tokens 0,1,2,4
```

Bash commands are the same in WSL, except use `/` path separators and add `--resume` when continuing a stopped run.

## Notes

- This is no longer Kaldi GOP. The per-phone features are pooled Charsiu frame probabilities plus a few confidence/duration statistics.
- The pretrained Kaldi-based checkpoints in `pretrained_models/` are not compatible with this new front-end. Train from scratch.
- The script assumes SpeechOcean762-style labels from `scores.json`. If you switch datasets, adapt the label loader first.
- The streaming pipeline is still trained from offline Charsiu alignments, but it simulates streaming inference by exposing only committed prefixes plus limited right context and by masking losses for unfinished phone/word units.
- The ASR-conditioned streaming pipeline is weak-reference by design. Phone and word losses are only kept on committed ASR words that can be aligned back to matching gold words.
- `build_streaming_asr_gopt_data.py` supports both `--timestamp-backend transformers` and `--timestamp-backend whisper_timestamped`. The default script uses `transformers` for locally fine-tuned Whisper checkpoints; use `whisper_timestamped` when you want timestamp extraction directly from a base or compatible Whisper checkpoint.
- `train_streaming_whisper.py` writes both `best_model/` and `last_model/`, plus `last_checkpoint.pt` for resume.
- `train_streaming_charsiu.py` writes `models/best_audio_model.pth` and `last_checkpoint.pt` for resume.
