#!/usr/bin/env bash
set -euo pipefail
cd /DATA_2/guest/custom-gopt
printf 'PROCS\n'
ps -ef | grep -E 'pcn_extra_20260704_2130|train_streaming_pcn' | grep -v grep || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -8 || true
printf 'LOGS\n'
ls -l server_run_logs/pcn_extra_20260704_2130 2>/dev/null || true
printf 'RUNNER\n'
cat server_run_logs/pcn_extra_20260704_2130/runner.log 2>/dev/null || true
printf 'A_TAIL\n'
tail -80 server_run_logs/pcn_extra_20260704_2130/A_loss_dimmask.log 2>/dev/null || true
printf 'B_TAIL\n'
tail -80 server_run_logs/pcn_extra_20260704_2130/B_relaxed_softlabel.log 2>/dev/null || true
printf 'FILES\n'
find exp/pcn_extra_20260704_2130 -maxdepth 3 -type f -printf '%p %s\n' 2>/dev/null | sort || true
