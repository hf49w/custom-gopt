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

## PCN/N-best Streaming Data

`src/prep_data/build_streaming_pcn_gopt_data.py` builds the stateful PCN dataset without changing the existing v6 ASR-driven pipeline. The schema is `streaming_pcn_gopt_v2_stateful`.

This is an N-best-derived phoneme confusion network, not a strict ASR lattice. It is built from Whisper N-best text hypotheses, local token/word transition scores, G2P phones, and Charsiu acoustic evidence. SpeechOcean762 GT text/phones/scores are used only to create training labels and loss masks; inference does not read GT text, phones, `scores.json`, or SpeechOcean labels.

Example:

```powershell
python src\prep_data\build_streaming_pcn_gopt_data.py `
  --dataset-root ..\speechocean762\speechocean762 `
  --scores-json src\prep_data\scores.json `
  --output-dir data\streaming_pcn_gopt_v2_stateful `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --asr-model exp\streaming-whisper-base\best_model `
  --language english `
  --nbest 5 `
  --beam-size 8 `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --overwrite
```

Main arrays in each `<split>_chunks.npz`:

- `cn_post`: phone confusion-network posterior per PCN slot, with the last dimension reserved for epsilon.
- `cn_stats`: epsilon probability, entropy, top-1 probability, top1-top2 margin, prefix stability.
- `acoustic_post`: Charsiu frame-evidence posterior aligned to each PCN slot.
- `acoustic_stats`: acoustic entropy, acoustic margin, duration, PCN-Charsiu JS divergence.
- `prosody`: F0, voiced probability, log-energy, pause, rate, and articulation-rate features computed from the visible prefix.
- `phone_target`, `word_target`, `utt_target`: SpeechOcean762 supervision labels.
- `asr_correct_target`: whether the slot/word is supported as ASR-correct.
- `uncertainty_target`: high when GT is absent from N-best or the CN posterior is uncertain.
- `soft_label_weight`: 1 for exact matches, PCN posterior for approximate phone matches, 0 for mismatches or GT absent from N-best.
- `commit_mask`: compatibility alias for `cumulative_commit_mask`.
- `cumulative_commit_mask`: all PCN slots committed by this chunk.
- `new_commit_mask`: only newly committed complete words after monotonic top-phone mapping from the previous chunk.
- `mapped_old_slot`: current-slot to previous-slot mapping for stability training.
- `previous_chunk_id`, `utterance_index`, `state_reset`, `new_committed_word_count`, `cumulative_committed_word_count`: stateful streaming metadata.
- `confidence_target`, `confidence_loss_mask`: soft reliability targets from soft label weight, PCN entropy, and Charsiu acoustic support.
- `abstention_target`, `abstention_loss_mask`: whether a slot should abstain from reliable diagnosis.
- `visible_len`, `is_final`: chunk visibility and final-chunk marker.

This builder calls the real Transformers `generate(..., num_return_sequences=K, output_scores=True, return_dict_in_generate=True)` path and stores `token_ids`, `token_logprobs`, `token_confidences`, `word_token_ranges`, `word_logprobs`, `word_confidences`, `sequence_score`, and `length_normalized_sequence_score`. Word timestamps are primarily obtained by aligning each N-best G2P phone sequence to Charsiu frame posteriors and aggregating phone spans to words (`timestamp_source="charsiu_hypothesis_alignment"`). If that fails, the script falls back to duration-proportional estimates and records `timestamp_source="duration_proportional_fallback"` plus metadata counts.

### PCN Student Training

The PCN student model is `PCNStreamingScorer` in `src/models/streaming_pcn_gopt.py`. Its structure is:

- Whisper N-best PCN posterior -> 16D projection.
- Charsiu acoustic posterior -> 16D projection.
- DSP prosody statistics -> 8D projection.
- Reliability gate from CN entropy, acoustic entropy, PCN-Charsiu JS divergence, and prefix stability.
- Local causal Transformer over PCN slots.
- Online word pooling over `new_commit_mask` only.
- Causal GRU sentence state without tail CLS tokens.
- Heads for phone score, word scores, utterance scores, ASR correctness, uncertainty, confidence, and abstention.

The GRU is trained in true streaming order. `PCNUtteranceDataset` groups all chunks for one utterance, `DataLoader` may shuffle utterances but never shuffles chunks inside an utterance, and training unrolls:

```python
state = None
for chunk in utterance_chunks:
    if state_reset:
        state = None
    output = model(..., new_commit_mask=chunk["new_commit_mask"], prev_state=state)
    state = output["next_state"]
```

`forward(..., detach_next_state=False)` keeps gradients across chunks by default. Use `--tbptt-steps N` to detach every N chunks. `stream_step(...)` detaches state for inference.

Train without teacher distillation:

```powershell
python src\train_streaming_pcn.py `
  --data-dir data\streaming_pcn_gopt_v2_stateful `
  --exp-dir exp\streaming-pcn-gopt `
  --embed-dim 40 `
  --depth 2 `
  --heads 2 `
  --gru-dim 32 `
  --batch-size 32 `
  --tbptt-steps 0 `
  --n-epochs 80
```

The trainer adds confidence, abstention, calibration, and adjacent-chunk stability losses:

- `--loss-w-confidence` default `0.2`
- `--loss-w-abstention` default `0.2`
- `--loss-w-calibration` default `0.1`
- `--loss-w-phone-stability` default `0.02`
- `--loss-w-word-stability` default `0.02`
- `--loss-w-utt-stability` default `0.02`

`state_projection` is disabled by default. It is created only when `teacher_state_embedding` exists in the data and `--loss-w-state-projection > 0`; otherwise no hidden-state distillation is claimed or trained.

### Optional MultiPA Prefix Distillation

Run MultiPA on every prefix using the existing MultiPA script from `D:\研究生\智能体\MultiPA`:

```bash
cd /mnt/d/研究生/智能体/MultiPA
python eval_multipa_prefix.py \
  --prefix-manifest /mnt/d/研究生/智能体/gopt_charsiu/data/streaming_pcn_gopt/train_manifest.jsonl \
  --output-jsonl /mnt/d/研究生/智能体/gopt_charsiu/data/streaming_pcn_gopt/multipa_train_prefix.jsonl \
  --resume
```

Then inject the teacher scores into the PCN NPZ files:

```powershell
python scripts\local\inject_multipa_teacher_pcn.py `
  --data-dir data\streaming_pcn_gopt_v2_stateful `
  --teacher-jsonl data\streaming_pcn_gopt_v2_stateful\multipa_train_prefix.jsonl `
  --splits train
```

Run the same command for `val` and `test` JSONL files when available. `inject_multipa_teacher_pcn.py` writes:

- `teacher_prefix_utt_score`
- `teacher_final_utt_score`
- `teacher_utt_mask`
- `teacher_utt_dim_mask`
- `teacher_word_score`
- `teacher_word_mask`

MultiPA does not output completeness, so `teacher_utt_dim_mask[:, 1] = 0`; human labels still supervise completeness.

### No-GT Streaming Inference

`scripts/local/infer_streaming_pcn.py` runs the deployed path. Inputs are only WAV, Whisper, Charsiu, and the PCN checkpoint/config:

```powershell
python scripts\local\infer_streaming_pcn.py `
  --wav sample.wav `
  --whisper-model openai/whisper-base `
  --aligner-model charsiu/en_w2v2_tiny_fc_10ms `
  --checkpoint exp\streaming-pcn-gopt\models\best_audio_model.pth `
  --config exp\streaming-pcn-gopt\config.json `
  --output-jsonl exp\streaming-pcn-gopt\sample_stream.jsonl `
  --device cuda
```

For each chunk it writes `chunk_id`, `audio_end`, committed hypotheses, newly committed words, PCN summary, phone/word/utterance scores, confidence, abstention probability, whether the sentence state updated, commit diagnostics, and `process_time_sec`.

## Notes

- This is no longer Kaldi GOP. The per-phone features are pooled Charsiu frame probabilities plus a few confidence/duration statistics.
- The pretrained Kaldi-based checkpoints in `pretrained_models/` are not compatible with this new front-end. Train from scratch.
- The script assumes SpeechOcean762-style labels from `scores.json`. If you switch datasets, adapt the label loader first.
- The streaming pipeline is still trained from offline Charsiu alignments, but it simulates streaming inference by exposing only committed prefixes plus limited right context and by masking losses for unfinished phone/word units.
- The ASR-conditioned streaming pipeline is weak-reference by design. Phone and word losses are only kept on committed ASR words that can be aligned back to matching gold words.
- `build_streaming_asr_gopt_data.py` supports both `--timestamp-backend transformers` and `--timestamp-backend whisper_timestamped`. The default script uses `transformers` for locally fine-tuned Whisper checkpoints; use `whisper_timestamped` when you want timestamp extraction directly from a base or compatible Whisper checkpoint.
- `train_streaming_whisper.py` writes both `best_model/` and `last_model/`, plus `last_checkpoint.pt` for resume.
- `train_streaming_charsiu.py` writes `models/best_audio_model.pth` and `last_checkpoint.pt` for resume.
