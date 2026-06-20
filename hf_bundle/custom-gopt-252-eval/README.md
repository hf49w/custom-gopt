---
tags:
- speech
- whisper
- forced-alignment
- pronunciation-assessment
- gopt
license: other
---

# custom-gopt-252-eval

This repository bundles the models needed for the local evaluation pipeline:

`Whisper ASR -> Charsiu phone alignment -> Streaming GOPT pronunciation scoring`

The current Streaming GOPT checkpoint is the v6 ASR-confidence version. It uses 47-dimensional phone-segment features: the previous Charsiu-derived acoustic features plus one ASR confidence feature.

## Files

- `streaming_gopt_best/best_audio_model.pth`: best validation Streaming GOPT checkpoint.
- `streaming_gopt_best/config.json`: model architecture and training arguments.
- `streaming_gopt_best/inference_assets.json`: normalization statistics and phone-id mapping used by the inference example.
- `streaming_gopt_best/result.csv`: per-epoch training and validation metrics.
- `streaming_gopt_best/test_metrics.json`: held-out test metrics for the selected checkpoint.
- `whisper_best_model/`: Whisper ASR model used by the pipeline.
- `charsiu_en_w2v2_tiny_fc_10ms/`: Charsiu frame-level phone alignment model.
- `examples/infer_one_audio.py`: one-audio inference example.

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="faeea/custom-gopt-252-eval",
    repo_type="model",
    local_dir="./hf_models/custom-gopt-252-eval",
)
```

Then set:

```bash
export BUNDLE_DIR=$PWD/hf_models/custom-gopt-252-eval
```

## Code Dependencies

This model bundle is not a standalone Transformers model. The inference script needs the model definitions from `custom-gopt` and the official Charsiu code.

```bash
git clone https://github.com/hf49w/custom-gopt.git
git clone https://github.com/lingjzhu/charsiu third_party/charsiu_repo
git -C third_party/charsiu_repo checkout 13a69f2a22ca0c0962b75cc693399b0ae23a12c9
```

Install the project dependencies in the `custom-gopt` repository, then install the small extra NLTK assets used by Charsiu/G2P:

```bash
pip install -r requirements.txt
python -m pip install nltk
python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## One-Audio Inference

```bash
python "$BUNDLE_DIR/examples/infer_one_audio.py" \
  --audio /path/to/demo.wav \
  --bundle-dir "$BUNDLE_DIR" \
  --repo-root /path/to/custom-gopt \
  --charsiu-src-dir /path/to/third_party/charsiu_repo \
  --device cuda \
  --output-json ./one_audio_score.json
```

Use `--device cpu` when CUDA is unavailable.

The example accepts an English short utterance audio file. A mono 16 kHz WAV is preferred; other sample rates are resampled inside the script.

## Output Meaning

The Streaming GOPT model forward pass returns:

- `u1`: utterance-level accuracy.
- `u2`: utterance-level completeness.
- `u3`: utterance-level fluency.
- `u4`: utterance-level prosodic score.
- `u5`: utterance-level total score.
- `p`: phone-level pronunciation score for each visible phone token.
- `w1`: word-level accuracy.
- `w2`: word-level stress.
- `w3`: word-level total score.
- `w4`: word-level ASR accuracy.

The example script reports the user-facing utterance scores as:

```json
{
  "utterance_scores": {
    "accuracy": 8.4,
    "completeness": 10.0,
    "fluency": 8.3,
    "prosodic": 7.9,
    "total": 8.0
  },
  "overall_score": 8.0
}
```

For `accuracy`, `completeness`, `fluency`, `prosodic`, and `total`, higher is better. These scores follow the SpeechOcean-style pronunciation scoring scale used during training.

The word-level outputs have this meaning:

- `word_accuracy`: pronunciation accuracy for the word.
- `word_stress`: stress score for the word.
- `word_total`: overall word score.
- `word_asr_accuracy`: whether the ASR-driven word matched the expected word.

`word_asr_accuracy` is a 0/1-style score. In practice, it can be used as a word-read-correctly indicator: a value near `1` means the word was recognized/matched as read correctly, while a value near `0` means the word was not matched, not recognized correctly, or was not committed yet in the streaming prefix.

中文提示：词级别的 `asr_accuracy` 评分可以当作“这个词是否读对”的辅助指标。它不替代发音分数本身，但很适合用来标记某个词在 ASR 视角下是否正确读出。

## Test Metrics

Best checkpoint epoch: `11`

Held-out test metrics:

- `phone_test_mse`: `0.049197`
- `phone_test_pcc`: `0.397571`
- `utt_test_pcc`: `[0.651909, 0.012690, 0.724998, 0.733279, 0.681940]`
- `word_test_pcc`: `[0.404714, -0.003912, 0.412258, 0.417467]`

The word PCC order is:

`accuracy`, `stress`, `total`, `asr_accuracy`

## Notes

- The model was trained on SpeechOcean762-style English learner speech.
- The pipeline does not require a reference transcript at inference time; it first obtains a transcript from Whisper, aligns phones with Charsiu, then scores with Streaming GOPT.
- ASR errors can affect downstream word and phone alignment. Use `word_asr_accuracy` to identify words that the ASR-driven pipeline likely did or did not match correctly.
