#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-print}"
TARGET="${2:-}"

PCN_DATA_DIR="${PCN_DATA_DIR:-data/streaming_pcn_gopt_v2_stateful}"
PCN_DATA_WITH_ORACLE_DIR="${PCN_DATA_WITH_ORACLE_DIR:-data/streaming_pcn_gopt_v2_stateful_oracle}"
PCN_SLOT_PROSODY_DATA_DIR="${PCN_SLOT_PROSODY_DATA_DIR:-$PCN_DATA_WITH_ORACLE_DIR}"
ORACLE_TEACHER_JSONL="${ORACLE_TEACHER_JSONL:-}"
GPU="${GPU:-0}"
EXP_ROOT="${EXP_ROOT:-exp/pcn_extra}"

BASE_TRAIN_ARGS=(
  --n-epochs 80
  --batch-size 16
  --num-workers 4
)

A_ARGS=(
  --utt-dim-weights 1,0,1,1,1
  --loss-w-teacher-score 1.0
  --loss-w-prefix-kd 1.0
  --loss-w-rank 0.2
  --loss-w-phone-stability 0.01
  --loss-w-word-stability 0.01
  --loss-w-utt-stability 0.01
)
B_ARGS=("${A_ARGS[@]}" --soft-label-policy relaxed)
C_ARGS=("${B_ARGS[@]}" --loss-w-oracle-word 1.0 --loss-w-oracle-utt-prefix 0.5 --loss-w-oracle-utt-final 0.7 --loss-w-oracle-phone 0.3)
D_ARGS=("${C_ARGS[@]}" --utt-pooling-head gru_visible)
E_ARGS=("${D_ARGS[@]}" --fusion-mode concat_vector_gate)
F_ARGS=("${E_ARGS[@]}" --embed-dim 64 --depth 3 --heads 4 --gru-dim 64)
G_ARGS=("${E_ARGS[@]}")

usage() {
  cat <<'EOF'
Usage:
  scripts/server/run_pcn_extra_experiments.sh print
  scripts/server/run_pcn_extra_experiments.sh run A_loss_dimmask
  scripts/server/run_pcn_extra_experiments.sh run_all

Environment:
  PCN_DATA_DIR
  PCN_DATA_WITH_ORACLE_DIR
  ORACLE_TEACHER_JSONL
  GPU
  EXP_ROOT
  PCN_SLOT_PROSODY_DATA_DIR optional, defaults to PCN_DATA_WITH_ORACLE_DIR
EOF
}

join_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

data_dir_for_exp() {
  case "$1" in
    C_oracle_teacher|D_visible_pooling|E_vector_gate|F_capacity_64) printf '%s\n' "$PCN_DATA_WITH_ORACLE_DIR" ;;
    G_slot_prosody) printf '%s\n' "$PCN_SLOT_PROSODY_DATA_DIR" ;;
    *) printf '%s\n' "$PCN_DATA_DIR" ;;
  esac
}

args_for_exp() {
  case "$1" in
    A_loss_dimmask) printf '%s\n' "${A_ARGS[@]}" ;;
    B_relaxed_softlabel) printf '%s\n' "${B_ARGS[@]}" ;;
    C_oracle_teacher) printf '%s\n' "${C_ARGS[@]}" ;;
    D_visible_pooling) printf '%s\n' "${D_ARGS[@]}" ;;
    E_vector_gate) printf '%s\n' "${E_ARGS[@]}" ;;
    F_capacity_64) printf '%s\n' "${F_ARGS[@]}" ;;
    G_slot_prosody) printf '%s\n' "${G_ARGS[@]}" ;;
    *) return 1 ;;
  esac
}

all_exps() {
  printf '%s\n' \
    A_loss_dimmask \
    B_relaxed_softlabel \
    C_oracle_teacher \
    D_visible_pooling \
    E_vector_gate \
    F_capacity_64 \
    G_slot_prosody
}

print_oracle_injection_hint() {
  if [[ -n "$ORACLE_TEACHER_JSONL" ]]; then
    join_cmd python scripts/local/inject_oracle_gopt_teacher_pcn.py \
      --data-dir "$PCN_DATA_DIR" \
      --oracle-jsonl "$ORACLE_TEACHER_JSONL" \
      --output-dir "$PCN_DATA_WITH_ORACLE_DIR" \
      --splits train,val,test
  fi
}

command_for_exp() {
  local exp_name="$1"
  local data_di
  data_dir="$(data_dir_for_exp "$exp_name")"
  mapfile -t extra_args < <(args_for_exp "$exp_name")
  join_cmd env CUDA_VISIBLE_DEVICES="$GPU" python src/train_streaming_pcn.py \
    --data-dir "$data_dir" \
    --exp-dir "$EXP_ROOT/$exp_name" \
    "${BASE_TRAIN_ARGS[@]}" \
    "${extra_args[@]}"
}

run_exp() {
  local exp_name="$1"
  local exp_dir="$EXP_ROOT/$exp_name"
  if [[ -e "$exp_dir" ]]; then
    echo "Refusing to overwrite existing exp: $exp_dir" >&2
    exit 2
  fi
  mkdir -p "$EXP_ROOT"
  mapfile -t extra_args < <(args_for_exp "$exp_name")
  env CUDA_VISIBLE_DEVICES="$GPU" python src/train_streaming_pcn.py \
    --data-dir "$(data_dir_for_exp "$exp_name")" \
    --exp-dir "$exp_dir" \
    "${BASE_TRAIN_ARGS[@]}" \
    "${extra_args[@]}"
}

case "$ACTION" in
  print)
    usage
    print_oracle_injection_hint
    while IFS= read -r exp_name; do
      command_for_exp "$exp_name"
    done < <(all_exps)
    ;;
  run)
    if [[ -z "$TARGET" ]]; then
      echo "Missing experiment name." >&2
      usage
      exit 1
    fi
    args_for_exp "$TARGET" >/dev/null
    run_exp "$TARGET"
    ;;
  run_all)
    while IFS= read -r exp_name; do
      run_exp "$exp_name"
    done < <(all_exps)
    ;;
  *)
    usage
    exit 1
    ;;
esac
