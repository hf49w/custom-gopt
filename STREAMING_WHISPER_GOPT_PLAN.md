# Streaming Whisper-GOPT Plan

## 1. ASR Prefix Data Format

Build `train_prefix.jsonl` and `test_prefix.jsonl` with one row per prefix chunk.

Required fields:

- `utt_id`: utterance id
- `chunk_id`: prefix index inside the utterance
- `audio_path`: full wav path
- `audio_start`: always `0.0` for prefix training
- `audio_end`: visible audio boundary for this chunk
- `commit_time`: time boundary for committed labels
- `right_context_sec`: `audio_end - commit_time`
- `target_text`: committed transcript prefix used as the ASR label
- `visible_text`: text that is visible in the current chunk, including right context
- `prompt_text`: previous committed text prefix
- `full_text`: full reference transcript
- `committed_word_ids`: gold word ids whose end time is `<= commit_time`
- `visible_word_ids`: gold word ids whose end time is `<= audio_end`
- `is_final`: whether this is the final chunk of the utterance

## 2. Prefix Labeling Rules

- Word boundaries come from offline Charsiu alignment on the gold transcript.
- `target_text` contains only committed words.
- `visible_text` contains committed words plus words fully exposed by right context.
- Prefix text is lowercased for Whisper supervision.
- GOPT still uses phone-level features, but phone supervision only applies after ASR hypotheses are mapped back to gold words.

## 3. Commit Strategy

ASR training:

- Use oracle commit from offline gold word end times.
- For chunk `k`, expose audio up to `commit_time + right_context`.
- Train Whisper to predict only the committed prefix text.

ASR-conditioned GOPT data building:

- Decode every visible prefix with Whisper.
- Gate words by timestamp: only words with `word_end <= commit_time` are eligible for supervision.
- Add a stability gate: for non-final chunks, committed word count is capped by the longest common prefix shared with the previous chunk hypothesis.
- Final chunk can commit all timestamp-qualified words.

## 4. GOPT Loss Mask Changes

Visible tokens:

- All visible ASR words with a lexicon phone sequence are converted to phone tokens and fed to Streaming GOPT as context.

`phone_loss_mask`:

- `1` only for phone tokens that belong to committed ASR words
- the ASR word must match a gold word under monotonic LCS alignment
- the canonical phone sequence must match the gold phone sequence
- the aligned phone segment must end before `commit_time`

`word_loss_mask`:

- `1` for all phone tokens inside a committed matched word
- `0` for unmatched, unstable, or OOV ASR words

`utt_loss_mask`:

- `1` only on final chunks
- final chunk must also satisfy a minimum matched committed-word ratio threshold

## 5. Code Paths

- `src/prep_data/build_whisper_prefix_data.py`
  - builds streaming ASR prefix manifests from gold alignments
- `src/train_streaming_whisper.py`
  - fine-tunes Whisper on prefix chunks
- `src/prep_data/build_streaming_asr_gopt_data.py`
  - decodes prefixes with Whisper and rebuilds chunked GOPT data from ASR hypotheses
- `src/train_streaming_charsiu.py`
  - reuses the existing streaming GOPT trainer on the new ASR-driven chunk data

## 6. Training Stages

### Stage 1: Prefix Data Build

Input:

- `speechocean762/speechocean762/train/wav.scp`
- `speechocean762/speechocean762/test/wav.scp`
- `src/prep_data/scores.json`
- `charsiu/en_w2v2_tiny_fc_10ms`

Output:

- `data/streaming_whisper_prefix/train_prefix.jsonl`
- `data/streaming_whisper_prefix/test_prefix.jsonl`
- `data/streaming_whisper_prefix/metadata.json`

Train/Test split:

- Uses the original SpeechOcean762 `train` split as ASR training data
- Uses the original SpeechOcean762 `test` split as ASR evaluation data

Epochs:

- no training in this stage

### Stage 2: Whisper Prefix Fine-Tuning

Input:

- `data/streaming_whisper_prefix/train_prefix.jsonl`
- `data/streaming_whisper_prefix/test_prefix.jsonl`
- base model `openai/whisper-base`

Output:

- `exp/streaming-whisper-base/history.json`
- `exp/streaming-whisper-base/last_checkpoint.pt`
- `exp/streaming-whisper-base/last_model/`
- `exp/streaming-whisper-base/best_model/`

Train/Test split:

- train on `train_prefix.jsonl`
- evaluate WER on `test_prefix.jsonl`

Epochs:

- default `8`

### Stage 3: ASR-Driven GOPT Data Build

Input:

- `speechocean762/speechocean762/train/wav.scp`
- `speechocean762/speechocean762/test/wav.scp`
- `src/prep_data/scores.json`
- `exp/streaming-whisper-base/best_model/` or `last_model/`
- `charsiu/en_w2v2_tiny_fc_10ms`

Output:

- `data/streaming_asr_gopt/train_chunks.npz`
- `data/streaming_asr_gopt/test_chunks.npz`
- `data/streaming_asr_gopt/train_manifest.jsonl`
- `data/streaming_asr_gopt/test_manifest.jsonl`
- `data/streaming_asr_gopt/metadata.json`

Train/Test split:

- Uses the original SpeechOcean762 `train` split as GOPT training chunks
- Uses the original SpeechOcean762 `test` split as GOPT evaluation chunks

Epochs:

- no training in this stage

### Stage 4: Streaming GOPT Training

Input:

- `data/streaming_asr_gopt/train_chunks.npz`
- `data/streaming_asr_gopt/test_chunks.npz`
- `data/streaming_asr_gopt/metadata.json`

Output:

- `exp/streaming-asr-gopt/result.csv`
- `exp/streaming-asr-gopt/last_checkpoint.pt`
- `exp/streaming-asr-gopt/models/best_audio_model.pth`
- `exp/streaming-asr-gopt/preds/*.npy`

Train/Test split:

- train on `train_chunks.npz`
- evaluate on final chunks from `test_chunks.npz`

Epochs:

- default `100`

## 7. Resume Behavior

- `src/train_streaming_whisper.py --resume`
  - reloads `exp-dir/last_checkpoint.pt`
  - restores model, optimizer, scheduler, `history`, `global_step`, and `best_wer`
- `src/train_streaming_charsiu.py --resume`
  - reloads `exp-dir/last_checkpoint.pt`
  - restores model, optimizer, scheduler, `result`, `global_step`, `best_epoch`, and `best_mse`
- `src/run_streaming_whisper_gopt_wsl.sh`
  - auto-adds `--resume` when `AUTO_RESUME=1` and a stage checkpoint exists
