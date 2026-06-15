$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$datasetRoot = Join-Path $workspaceRoot "speechocean762\speechocean762"

$prefixDataDir = Join-Path $repoRoot "data\streaming_whisper_prefix"
$whisperExpDir = Join-Path $repoRoot "exp\streaming-whisper-base"
$asrGoptDataDir = Join-Path $repoRoot "data\streaming_asr_gopt"
$goptExpDir = Join-Path $repoRoot "exp\streaming-asr-gopt"

$alignerModel = "charsiu/en_w2v2_tiny_fc_10ms"
$asrBaseModel = "openai/whisper-base"
$fineTunedAsrModel = Join-Path $whisperExpDir "best_model"

python (Join-Path $repoRoot "src\prep_data\build_whisper_prefix_data.py") `
  --dataset-root $datasetRoot `
  --scores-json (Join-Path $repoRoot "src\prep_data\scores.json") `
  --output-dir $prefixDataDir `
  --aligner-model $alignerModel `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --overwrite

python (Join-Path $repoRoot "src\train_streaming_whisper.py") `
  --data-dir $prefixDataDir `
  --exp-dir $whisperExpDir `
  --model-name-or-path $asrBaseModel `
  --language english `
  --batch-size 8 `
  --eval-batch-size 8 `
  --n-epochs 8

python (Join-Path $repoRoot "src\prep_data\build_streaming_asr_gopt_data.py") `
  --dataset-root $datasetRoot `
  --scores-json (Join-Path $repoRoot "src\prep_data\scores.json") `
  --output-dir $asrGoptDataDir `
  --aligner-model $alignerModel `
  --asr-model $fineTunedAsrModel `
  --timestamp-backend transformers `
  --language english `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --min-utt-match-ratio 0.5 `
  --overwrite

python (Join-Path $repoRoot "src\train_streaming_charsiu.py") `
  --data-dir $asrGoptDataDir `
  --exp-dir $goptExpDir `
  --depth 3 `
  --heads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model streaming_gopt `
  --main-context-tokens 4,8,12,16 `
  --right-context-tokens 0,1,2,4
