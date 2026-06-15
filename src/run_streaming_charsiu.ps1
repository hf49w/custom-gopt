$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$datasetRoot = Join-Path $workspaceRoot "speechocean762\speechocean762"
$streamDataDir = Join-Path $repoRoot "data\streaming_charsiu_tiny"
$expDir = Join-Path $repoRoot "exp\streaming-charsiu-gopt"
$alignerModel = "charsiu/en_w2v2_tiny_fc_10ms"

python (Join-Path $repoRoot "src\prep_data\build_streaming_charsiu_data.py") `
  --dataset-root $datasetRoot `
  --scores-json (Join-Path $repoRoot "src\prep_data\scores.json") `
  --output-dir $streamDataDir `
  --aligner-model $alignerModel `
  --chunk-sec 0.64 `
  --right-context-sec 0.16 `
  --overwrite

python (Join-Path $repoRoot "src\train_streaming_charsiu.py") `
  --data-dir $streamDataDir `
  --exp-dir $expDir `
  --depth 3 `
  --heads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model streaming_gopt `
  --main-context-tokens 4,8,12,16 `
  --right-context-tokens 0,1,2,4
