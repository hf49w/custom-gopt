#!/usr/bin/env bash
set -euo pipefail
cd /DATA_2/guest/custom-gopt
ROOT=exp/pcn_extra_20260704_2130
LOG=server_run_logs/pcn_extra_20260704_2130
echo '=== now ==='
date
hostname
echo '=== processes ==='
ps -ef | grep -E 'pcn_extra_20260704_2130|train_streaming_pcn|run_pcn_extra' | grep -v grep || true
echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -8 || true
echo '=== logs ==='
ls -lh "$LOG" 2>/dev/null || true
for f in "$LOG"/*.pid; do [ -f "$f" ] && echo "$f=$(cat "$f")"; done
echo '=== runner ==='
cat "$LOG/runner.log" 2>/dev/null || true
for exp in A_loss_dimmask B_relaxed_softlabel; do
  echo "=== $exp log tail ==="
  tail -40 "$LOG/$exp.log" 2>/dev/null || true
  echo "=== $exp files ==="
  find "$ROOT/$exp" -maxdepth 3 -type f -printf '%p %s\n' 2>/dev/null | sort || true
  echo "=== $exp metrics ==="
  /DATA_2/guest/custom-gopt/.conda_env/bin/python - <<PY
import json, pathlib, math
exp = '$exp'
root = pathlib.Path('$ROOT') / exp
for name in ['config.json','history.json','test_metrics.json']:
    p = root / name
    print('FILE', p, 'exists=', p.exists())
    if not p.exists():
        continue
    obj = json.load(open(p, encoding='utf-8'))
    if name == 'config.json':
        args = obj.get('args', {})
        print({k: args.get(k) for k in ['utt_dim_weights','soft_label_policy','utt_pooling_head','fusion_mode','n_epochs','batch_size','loss_w_teacher_score','loss_w_prefix_kd','loss_w_rank']})
    elif name == 'history.json':
        print('epochs', len(obj))
        if obj:
            last = obj[-1]
            print('last_epoch', last.get('epoch'))
            print('last_train_loss', last.get('train',{}).get('loss'))
            print('last_val_loss', last.get('val',{}).get('loss'))
            vals=[r.get('val',{}).get('loss') for r in obj if isinstance(r.get('val',{}).get('loss'), (int,float))]
            print('best_val_loss', min(vals) if vals else None)
    elif name == 'test_metrics.json':
        keys=['loss','coverage_100_mae','coverage_100_pcc','coverage_90_mae','coverage_90_pcc','coverage_80_mae','coverage_80_pcc','coverage_70_mae','coverage_70_pcc','mean_adjacent_utt_delta','phone_revision_rate','word_revision_rate','mean_effective_supervision_weight','supervised_slot_count']
        print({k: obj.get(k) for k in keys if k in obj})
PY
done
