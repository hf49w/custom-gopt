# PCN Extra Experiments

These experiments are additive. Existing PCN v2 stateful data, checkpoints, and training commands keep their default behavior unless one of the new flags below is passed explicitly. The PCN source remains an N-best-derived PCN, not a strict lattice PCN.

## Purpose

- `A_loss_dimmask`: masks the utterance completeness dimension with `--utt-dim-weights 1,0,1,1,1` while keeping the original model structure.
- `B_relaxed_softlabel`: adds `--soft-label-policy relaxed` so difficult ASR slots with PCN or acoustic support can keep low-weight phone/word supervision.
- `C_oracle_teacher`: trains with closed-oracle GOPT teacher fields injected as `oracle_*`. This is privileged training supervision only; inference never uses GT text.
- `D_visible_pooling`: adds `--utt-pooling-head gru_visible` so final/all-visible slot summaries can inform utterance scores.
- `E_vector_gate`: adds `--fusion-mode concat_vector_gate` to retain PCN/acoustic differences instead of only scalar interpolation.
- `F_capacity_64`: raises the lightweight student capacity with `--embed-dim 64 --depth 3 --heads 4 --gru-dim 64`.
- `G_slot_prosody`: uses data rebuilt with `--include-slot-prosody` so slot-local duration, energy, F0, and stress fallback features can support stress/prosody.

## New Training Flags

```bash
--utt-dim-weights 1,1,1,1,1
--word-dim-weights 1,1,1
--teacher-word-dim-weights 1,1,1
--oracle-word-dim-weights 1,1,1
--soft-label-policy original|relaxed
--relaxed-min-gt-posterior 0.05
--relaxed-acoustic-scale 0.3
--relaxed-min-weight 0.05
--relaxed-max-weight 0.5
--loss-w-oracle-phone 0.0
--loss-w-oracle-word 0.0
--loss-w-oracle-utt-prefix 0.0
--loss-w-oracle-utt-final 0.0
--utt-pooling-head gru|gru_visible
--fusion-mode scalar_gate|concat_vector_gate
--loss-w-stress-pearson 0.0
--loss-w-oracle-stress-pearson 0.0
--loss-w-teacher-stress-pearson 0.0
--loss-w-stress-rank 0.0
--loss-w-oracle-stress-rank 0.0
--stress-loss-mask all|vowel|voiced_or_vowel
--stress-branch none|detached|gradscale
--stress-grad-scale 0.2
```

## Data And Teacher Injection

Build the normal PCN v2 stateful data as before. To include slot prosody in a new dataset directory:

```bash
python src/prep_data/build_streaming_pcn_gopt_data.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir data/streaming_pcn_gopt_v2_stateful_slotprosody \
  --include-slot-prosody
```

Inject a closed-oracle teacher into a new data directory without overwriting MultiPA fields:

```bash
python scripts/local/inject_oracle_gopt_teacher_pcn.py \
  --data-dir data/streaming_pcn_gopt_v2_stateful \
  --oracle-jsonl oracle_gopt_prefix.jsonl \
  --output-dir data/streaming_pcn_gopt_v2_stateful_oracle \
  --splits train,val,test
```

CSV input is also supported:

```bash
python scripts/local/inject_oracle_gopt_teacher_pcn.py \
  --data-dir data/streaming_pcn_gopt_v2_stateful \
  --oracle-csv oracle_gopt_scores.csv \
  --output-dir data/streaming_pcn_gopt_v2_stateful_oracle \
  --splits train,val,test
```

## Server Commands

The server launcher defaults to printing commands:

```bash
PCN_DATA_DIR=data/streaming_pcn_gopt_v2_stateful \
PCN_DATA_WITH_ORACLE_DIR=data/streaming_pcn_gopt_v2_stateful_oracle \
ORACLE_TEACHER_JSONL=oracle_gopt_prefix.jsonl \
GPU=0 \
EXP_ROOT=exp/pcn_extra \
scripts/server/run_pcn_extra_experiments.sh print
```

Run one experiment only when ready:

```bash
scripts/server/run_pcn_extra_experiments.sh run A_loss_dimmask
```

Run all staged experiments:

```bash
scripts/server/run_pcn_extra_experiments.sh run_all
```

The script refuses to run if the target experiment directory already exists.

## Summary

```bash
python scripts/local/summarize_pcn_extra_experiments.py \
  --exp-root exp/pcn_extra \
  --baseline-json exp/streaming-pcn-gopt/test_metrics.json \
  --output-csv exp/pcn_extra_summary.csv
```

Missing metrics are left blank.

## Stress Experiments On 252

The stress-specific branch is opt-in. Defaults keep previous experiments and checkpoint loading behavior unchanged. For the July 2026 oracle run, use:

```bash
cd /DATA_2/guest/custom-gopt
./scripts/server/run_pcn_stress_experiments_252.sh prepare
./scripts/server/run_pcn_stress_experiments_252.sh print
```

`prepare` builds `data_streaming_pcn_oracle_gopt_full_slotprosody` by augmenting the existing oracle data directory. It does not rerun Whisper or Charsiu; energy, F0, and lexical stress one-hot fields are zero fallback when frame-level features are unavailable. The runner always skips GPU3 and only selects GPUs with no compute process, memory below 1000 MiB, and utilization below 10%.

Experiments `H` through `M` start from `G_oracle_capacity64`, always keep `--utt-dim-weights 1,0,1,1,1`, and add word stress weighting, stress Pearson/rank losses, detached or grad-scaled stress branches, and optional slot-prosody stress masks.

## Fallbacks

- `slot_prosody` lexical stress one-hot values are zero unless a reliable stress source is added later.
- Inference fills normalized zero slot prosody when a checkpoint expects slot prosody but runtime extraction is unavailable.
- Oracle phone alignment uses slot-time overlap when `slot_times` exist in the manifest; older data without `slot_times` falls back to phone index.
