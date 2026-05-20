$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$datasetRoot = Join-Path $workspaceRoot "speechocean762\speechocean762"
$seqDataDir = Join-Path $repoRoot "data\seq_data_charsiu_tiny"
$expDir = Join-Path $repoRoot "exp\charsiu-tiny-gopt"
$alignerModel = "charsiu/en_w2v2_tiny_fc_10ms"

python (Join-Path $repoRoot "src\prep_data\build_charsiu_seq_data.py") `
  --dataset-root $datasetRoot `
  --scores-json (Join-Path $repoRoot "src\prep_data\scores.json") `
  --output-dir $seqDataDir `
  --aligner-model $alignerModel `
  --overwrite

python (Join-Path $repoRoot "src\train_charsiu.py") `
  --data-dir $seqDataDir `
  --exp-dir $expDir `
  --goptdepth 3 `
  --goptheads 1 `
  --batch-size 25 `
  --embed-dim 24 `
  --model gopt
