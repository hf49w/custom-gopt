import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


ABLATION_ROOT = REPO_ROOT / 'paper_experiments' / 'ablations'
MAIN_ROOT = REPO_ROOT / 'paper_experiments' / 'main_comparison'
FROZEN_PATH = MAIN_ROOT / 'frozen_main_model.json'
OFFICIAL_SEED_PATH = MAIN_ROOT / 'official_m_seed_set.yaml'
PRIMARY_RULE_PATH = MAIN_ROOT / 'primary_checkpoint_rule.yaml'


TRAIN_ARG_ORDER = [
    'data_dir',
    'lr',
    'n_epochs',
    'batch_size',
    'num_workers',
    'embed_dim',
    'depth',
    'heads',
    'gru_dim',
    'main_context_tokens',
    'tbptt_steps',
    'loss_w_phone',
    'loss_w_word',
    'loss_w_utt',
    'loss_w_asr',
    'loss_w_uncertainty',
    'loss_w_confidence',
    'loss_w_abstention',
    'loss_w_calibration',
    'loss_w_teacher_score',
    'loss_w_prefix_kd',
    'loss_w_rank',
    'loss_w_oracle_phone',
    'loss_w_oracle_word',
    'loss_w_oracle_utt_prefix',
    'loss_w_oracle_utt_final',
    'loss_w_stress_pearson',
    'loss_w_oracle_stress_pearson',
    'loss_w_teacher_stress_pearson',
    'loss_w_stress_rank',
    'loss_w_oracle_stress_rank',
    'loss_w_phone_stability',
    'loss_w_word_stability',
    'loss_w_utt_stability',
    'loss_w_commit_consistency',
    'loss_w_state_projection',
    'utt_dim_weights',
    'word_dim_weights',
    'teacher_word_dim_weights',
    'oracle_word_dim_weights',
    'soft_label_policy',
    'relaxed_min_gt_posterior',
    'relaxed_acoustic_scale',
    'relaxed_min_weight',
    'relaxed_max_weight',
    'utt_pooling_head',
    'fusion_mode',
    'pcn_input_mode',
    'rank_margin',
    'stress_rank_margin',
    'stress_rank_max_items',
    'stress_loss_mask',
    'stress_voiced_threshold',
    'stress_branch',
    'stress_grad_scale',
    'grad_clip_norm',
    'seed',
    'tf32',
    'compile',
    'disable_acoustic',
    'disable_prosody',
    'disable_uncertainty_stats',
]


ABLATIONS = {
    'M_top1_onehot': {
        'changed_factor': 'PCN posterior input replaced with true top-1 one-hot per slot',
        'overrides': {'pcn_input_mode': 'top1_onehot'},
        'expected_changed_fields': ['args.pcn_input_mode'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
        'notes': 'Only cn_post is replaced by argmax one-hot. Hypotheses are not duplicated.',
    },
    'M_no_acoustic': {
        'changed_factor': 'Remove Charsiu posterior/acoustic embedding/acoustic statistics branch',
        'overrides': {'disable_acoustic': True},
        'expected_changed_fields': ['args.disable_acoustic'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_prosody': {
        'changed_factor': 'Remove 14-D prefix prosody branch',
        'overrides': {'disable_prosody': True},
        'expected_changed_fields': ['args.disable_prosody'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
        'notes': 'Slot prosody remains enabled because this ablates only the utterance-level 14-D prefix prosody branch.',
    },
    'M_no_uncertainty_stats': {
        'changed_factor': 'Zero explicit entropy/margin/JS/prefix-stability uncertainty statistics',
        'overrides': {'disable_uncertainty_stats': True},
        'expected_changed_fields': ['args.disable_uncertainty_stats'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_vector_gate': {
        'changed_factor': 'Scalar reliability gate replaced with 16-D element-wise vector gate',
        'overrides': {'fusion_mode': 'concat_vector_gate'},
        'expected_changed_fields': ['args.fusion_mode'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
        'historical_reuse': 'rejected: historical F differs in embed_dim/depth/heads/gru_dim/batch_size and is not a single-variable M ablation.',
    },
    'M_fixed_average': {
        'changed_factor': 'Learned gate replaced with fixed 0.5*z_pcn + 0.5*z_acoustic',
        'overrides': {'fusion_mode': 'fixed_average'},
        'expected_changed_fields': ['args.fusion_mode'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_gru': {
        'changed_factor': 'Remove recurrent GRU state; use visible/committed pooling',
        'overrides': {'utt_pooling_head': 'visible_committed'},
        'expected_changed_fields': ['args.utt_pooling_head'],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_stability': {
        'changed_factor': 'Set all stability losses to zero',
        'overrides': {
            'loss_w_phone_stability': 0.0,
            'loss_w_word_stability': 0.0,
            'loss_w_utt_stability': 0.0,
        },
        'expected_changed_fields': [
            'args.loss_w_phone_stability',
            'args.loss_w_word_stability',
            'args.loss_w_utt_stability',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_replay_all_committed': {
        'changed_factor': 'Inference-only state update: replay all cumulative committed words from zero state',
        'overrides': {},
        'expected_changed_fields': ['evaluation.state_update_mode'],
        'requires_retraining': False,
        'reuse_checkpoint': 'M-Full primary checkpoint',
        'state_update_mode': 'replay_all_committed',
    },
    'M_no_multipa_teacher': {
        'changed_factor': 'Remove MultiPA teacher losses only',
        'overrides': {
            'loss_w_teacher_score': 0.0,
            'loss_w_prefix_kd': 0.0,
            'loss_w_rank': 0.0,
            'loss_w_teacher_stress_pearson': 0.0,
        },
        'expected_changed_fields': [
            'args.loss_w_teacher_score',
            'args.loss_w_prefix_kd',
            'args.loss_w_rank',
            'args.loss_w_teacher_stress_pearson',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_closed_gopt_teacher': {
        'changed_factor': 'Remove closed/oracle GOPT teacher losses only',
        'overrides': {
            'loss_w_oracle_phone': 0.0,
            'loss_w_oracle_word': 0.0,
            'loss_w_oracle_utt_prefix': 0.0,
            'loss_w_oracle_utt_final': 0.0,
            'loss_w_oracle_stress_pearson': 0.0,
            'loss_w_oracle_stress_rank': 0.0,
        },
        'expected_changed_fields': [
            'args.loss_w_oracle_phone',
            'args.loss_w_oracle_word',
            'args.loss_w_oracle_utt_prefix',
            'args.loss_w_oracle_utt_final',
            'args.loss_w_oracle_stress_pearson',
            'args.loss_w_oracle_stress_rank',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_teachers': {
        'changed_factor': 'Remove MultiPA and closed/oracle GOPT teacher losses while keeping human supervision',
        'overrides': {
            'loss_w_teacher_score': 0.0,
            'loss_w_prefix_kd': 0.0,
            'loss_w_rank': 0.0,
            'loss_w_teacher_stress_pearson': 0.0,
            'loss_w_oracle_phone': 0.0,
            'loss_w_oracle_word': 0.0,
            'loss_w_oracle_utt_prefix': 0.0,
            'loss_w_oracle_utt_final': 0.0,
            'loss_w_oracle_stress_pearson': 0.0,
            'loss_w_oracle_stress_rank': 0.0,
        },
        'expected_changed_fields': [
            'args.loss_w_teacher_score',
            'args.loss_w_prefix_kd',
            'args.loss_w_rank',
            'args.loss_w_teacher_stress_pearson',
            'args.loss_w_oracle_phone',
            'args.loss_w_oracle_word',
            'args.loss_w_oracle_utt_prefix',
            'args.loss_w_oracle_utt_final',
            'args.loss_w_oracle_stress_pearson',
            'args.loss_w_oracle_stress_rank',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_stress_weight': {
        'changed_factor': 'Set human/MultiPA/oracle word dimension weights to 1,1,1',
        'overrides': {
            'word_dim_weights': [1.0, 1.0, 1.0],
            'teacher_word_dim_weights': [1.0, 1.0, 1.0],
            'oracle_word_dim_weights': [1.0, 1.0, 1.0],
        },
        'expected_changed_fields': [
            'args.word_dim_weights',
            'args.teacher_word_dim_weights',
            'args.oracle_word_dim_weights',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
    },
    'M_no_auxiliary': {
        'changed_factor': 'Remove ASR correctness, uncertainty, confidence, abstention, and calibration auxiliary losses',
        'overrides': {
            'loss_w_asr': 0.0,
            'loss_w_uncertainty': 0.0,
            'loss_w_confidence': 0.0,
            'loss_w_abstention': 0.0,
            'loss_w_calibration': 0.0,
        },
        'expected_changed_fields': [
            'args.loss_w_asr',
            'args.loss_w_uncertainty',
            'args.loss_w_confidence',
            'args.loss_w_abstention',
            'args.loss_w_calibration',
        ],
        'requires_retraining': True,
        'reuse_checkpoint': None,
        'notes': 'Auxiliary heads are retained but their training losses are zero; main trunk capacity is unchanged.',
    },
}


def get_args():
    parser = argparse.ArgumentParser(description='Prepare strict single-variable Ours-M ablations.')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    args_p = sub.add_parser('args')
    args_p.add_argument('--experiment', required=True, choices=sorted(ABLATIONS))
    args_p.add_argument('--seed', type=int, required=True)
    args_p.add_argument('--exp-dir', required=True)
    sub.add_parser('list')
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs():
    for sub in ['configs', 'checkpoints', 'predictions', 'metrics', 'logs', 'pids']:
        (ABLATION_ROOT / sub).mkdir(parents=True, exist_ok=True)


def verify_frozen():
    frozen = read_json(FROZEN_PATH)
    if frozen.get('paper_name') != 'Ours-M' or frozen.get('experiment_id') != 'M_stress_scalar_gate_capacity64':
        raise SystemExit(
            f'frozen_main_model.json must point to Ours-M/M_stress_scalar_gate_capacity64; got '
            f'{frozen.get("paper_name")!r}/{frozen.get("experiment_id")!r}'
        )
    return frozen


def parse_official_seed_yaml():
    seeds = []
    for line in OFFICIAL_SEED_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('seed:'):
            seeds.append(int(line.split(':', 1)[1].strip()))
    return seeds


def load_base():
    frozen = verify_frozen()
    config = read_json(frozen['resolved_config_path'])
    args = dict(config.get('args', {}))
    defaults = {
        'pcn_input_mode': 'posterior',
        'disable_acoustic': False,
        'disable_prosody': False,
        'disable_uncertainty_stats': False,
        'compile': False,
        'resume': False,
    }
    for key, value in defaults.items():
        args.setdefault(key, value)
    return frozen, config, args


def normalize_value(value):
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    return value


def merge_args(base_args, experiment, seed, exp_dir):
    merged = {key: normalize_value(value) for key, value in base_args.items()}
    merged.update(ABLATIONS[experiment]['overrides'])
    merged['seed'] = int(seed)
    merged['exp_dir'] = str(exp_dir)
    merged['device'] = None
    merged['resume'] = False
    return merged


def scientific_diff(base_args, merged_args):
    ignored = {'seed', 'exp_dir', 'device', 'resume'}
    diff = {}
    for key in sorted(set(base_args) | set(merged_args)):
        if key in ignored:
            continue
        base = normalize_value(base_args.get(key))
        cur = normalize_value(merged_args.get(key))
        if base != cur:
            diff['args.' + key] = {'base': base, 'current': cur}
    return diff


def arg_to_cli_name(key):
    return '--' + key.replace('_', '-')


def value_to_cli(value):
    if isinstance(value, list):
        return ','.join(str(item) for item in value)
    return str(value)


def cli_tokens(args):
    tokens = []
    for key in TRAIN_ARG_ORDER:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, bool):
            if value:
                tokens.append(arg_to_cli_name(key))
            continue
        if value is None:
            continue
        tokens.extend([arg_to_cli_name(key), value_to_cli(value)])
    tokens.extend(['--exp-dir', str(args['exp_dir'])])
    return tokens


def write_yaml(path, rows):
    Path(path).write_text('\n'.join(rows) + '\n', encoding='utf-8')


def registry_rows(frozen, base_config, base_args):
    primary_seed = int(frozen.get('primary_seed', 1337))
    official_seeds = parse_official_seed_yaml() or list(frozen.get('official_seed_set', []))
    rows = [
        'full_reference:',
        '  paper_name: Ours-M',
        '  experiment_id: M_stress_scalar_gate_capacity64',
        '  reference_name: M-Full',
        '  selection_basis: author_fixed_architecture_after_exploratory_model_development',
        '  frozen_main_model: ' + str(FROZEN_PATH),
        '  primary_seed: ' + str(primary_seed),
        '  official_seeds: [' + ', '.join(str(seed) for seed in official_seeds) + ']',
        '  config_sha256: ' + sha256(frozen['resolved_config_path']),
        '  data_dir: ' + frozen['strict_prefix_train_val_test_data_dir'],
        '  data_metadata_sha256: ' + frozen['data_hash']['metadata_json_sha256'],
        '  charsiu_mode: ' + str(frozen.get('charsiu_mode')),
        '  parameter_count: computed_from_checkpoint_in_summarize',
        'seed_policy:',
        '  first_stage_seed: ' + str(primary_seed),
        '  multiseed_policy: use all official Ours-M seeds unless resource-limited before launch',
        '  no_test_based_seed_filtering: true',
        'historical_checkpoint_reuse:',
        '  reused:',
        '    - experiment_id: M_replay_all_committed',
        '      checkpoint: ' + frozen['primary_checkpoint_path'],
        '      reason: inference-only state update policy control',
        '  rejected:',
        '    - experiment_id: F_oracle_vector_gate',
        '      reason: differs from M in capacity/batch/config and is not a single-variable vector-gate ablation',
        '    - experiment_id: H_stress_weighted_G',
        '      reason: not M-Full reference and differs in data/slotprosody interpretation from valid ablation definition',
        '    - experiment_id: J_stress_detached_branch',
        '      reason: duplicate/invalid slotprosody control; not a single-variable M ablation',
        '    - experiment_id: K_slot_prosody_stress',
        '      reason: duplicate/invalid slotprosody control; not a single-variable M ablation',
        'experiments:',
    ]
    for name, spec in ABLATIONS.items():
        merged = merge_args(base_args, name, primary_seed, ABLATION_ROOT / 'checkpoints' / f'{name}_seed{primary_seed}')
        diff = scientific_diff(base_args, merged)
        unexpected = sorted(set(diff) - set(spec['expected_changed_fields']))
        rows.extend([
            '  - experiment_id: ' + name,
            '    changed_factor: ' + spec['changed_factor'],
            '    requires_retraining: ' + str(bool(spec['requires_retraining'])).lower(),
            '    reuse_checkpoint: ' + str(spec.get('reuse_checkpoint') or 'null'),
            '    first_stage_seed: ' + str(primary_seed),
            '    expected_changed_fields: [' + ', '.join(spec['expected_changed_fields']) + ']',
            '    actual_generated_changed_fields: [' + ', '.join(sorted(diff)) + ']',
            '    config_diff_status: ' + ('ok' if not unexpected else 'unexpected_diff'),
            '    parameter_count: computed_from_checkpoint_in_summarize',
        ])
        if spec.get('historical_reuse'):
            rows.append('    historical_reuse: ' + spec['historical_reuse'])
        if spec.get('notes'):
            rows.append('    notes: ' + spec['notes'])
    return rows


def prepare():
    ensure_dirs()
    frozen, base_config, base_args = load_base()
    registry = registry_rows(frozen, base_config, base_args)
    write_yaml(ABLATION_ROOT / 'ablation_registry.yaml', registry)
    write_retraining_requirements(frozen)
    write_multiseed_rule(frozen)
    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S CST'),
        'frozen_main_model': str(FROZEN_PATH),
        'official_seed_file_read': str(OFFICIAL_SEED_PATH),
        'primary_checkpoint_rule_read': str(PRIMARY_RULE_PATH),
        'experiments': sorted(ABLATIONS),
    }
    (ABLATION_ROOT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'registry': str(ABLATION_ROOT / 'ablation_registry.yaml')}, ensure_ascii=False))


def write_retraining_requirements(frozen):
    lines = [
        '# Ours-M Ablation Retraining Requirements',
        '',
        'Full reference: `M_stress_scalar_gate_capacity64` (`Ours-M` / `M-Full`).',
        '',
        'Pure inference controls:',
        '',
        '- `M_replay_all_committed`: reuses the Ours-M primary checkpoint and changes only the inference state update policy.',
        '',
        'Retraining required:',
        '',
    ]
    for name, spec in ABLATIONS.items():
        if not spec['requires_retraining']:
            continue
        lines.append(f'- `{name}`: {spec["changed_factor"]}.')
    lines.extend([
        '',
        'Historical A-M checkpoints are not reused as strict ablations unless the diff checker proves the current single-variable definition. Current rejected examples: F, H, J, K.',
        'Validation selects checkpoints. Test metrics must not change ablation definitions, checkpoint choice, or seed inclusion.',
    ])
    (ABLATION_ROOT / 'retraining_requirements.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_multiseed_rule(frozen):
    rows = [
        'created_before_new_ablation_test_metrics: true',
        'full_reference: Ours-M',
        'full_reference_experiment_id: M_stress_scalar_gate_capacity64',
        'selection_basis: validation_relative_impact_vs_M_Full',
        'candidate_pool: first_stage_canonical_seed_ablations',
        'test_metrics_allowed_for_selection: false',
        'official_seed_set: [' + ', '.join(str(seed) for seed in frozen.get('official_seed_set', [])) + ']',
        'seed_expansion_policy:',
        '  target_count: 8',
        '  use_all_official_m_seeds: true',
        '  no_result_based_seed_filtering: true',
        'validation_impact_score:',
        '  direction: larger_absolute_drop_or_change_has_higher_priority_for_multiseed_confirmation',
        '  terms:',
        '    - metric: word_total_pcc_100_delta_abs',
        '      weight: 0.25',
        '    - metric: sentence_total_pcc_100_delta_abs',
        '      weight: 0.20',
        '    - metric: phone_pcc_100_delta_abs',
        '      weight: 0.20',
        '    - metric: word_stress_pcc_100_delta_abs',
        '      weight: 0.15',
        '    - metric: prefix_sentence_total_pcc_50_75_delta_abs',
        '      weight: 0.10',
        '    - metric: prefix_word_total_pcc_50_75_delta_abs',
        '      weight: 0.10',
        'tie_breakers:',
        '  - prefer_core_architecture_ablation_over_loss_only_if_scores_equal',
        '  - lexicographic_experiment_id',
        'mandatory_reporting:',
        '  - single_seed_results_are_preliminary',
        '  - main_ablation_conclusions_require_same_seed_set_as_Ours-M',
    ]
    write_yaml(ABLATION_ROOT / 'multiseed_selection_rule.yaml', rows)


def emit_args(experiment, seed, exp_dir):
    ensure_dirs()
    frozen, base_config, base_args = load_base()
    if ABLATIONS[experiment]['requires_retraining'] is False:
        raise SystemExit(f'{experiment} is inference-only and should not be trained.')
    merged = merge_args(base_args, experiment, seed, exp_dir)
    diff = scientific_diff(base_args, merged)
    expected = set(ABLATIONS[experiment]['expected_changed_fields'])
    actual = set(diff)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    status = 'ok' if not unexpected and not missing else 'failed'
    out = {
        'experiment_id': experiment,
        'seed': int(seed),
        'exp_dir': str(exp_dir),
        'changed_factor': ABLATIONS[experiment]['changed_factor'],
        'expected_changed_fields': sorted(expected),
        'actual_generated_diff': diff,
        'unexpected_changed_fields': unexpected,
        'missing_expected_fields': missing,
        'config_diff_status': status,
        'parameter_count': 'computed_from_checkpoint_in_summarize',
        'base_parameter_count': 'computed_from_checkpoint_in_summarize',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S CST'),
    }
    prefix = ABLATION_ROOT / 'configs' / f'{experiment}_seed{seed}'
    (prefix.with_suffix('.expected_config_diff.json')).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (prefix.with_suffix('.train_args.json')).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if status != 'ok':
        raise SystemExit(json.dumps(out, ensure_ascii=False, indent=2))
    for token in cli_tokens(merged):
        print(token)


def list_experiments():
    for name, spec in ABLATIONS.items():
        print(json.dumps({'experiment_id': name, **spec}, ensure_ascii=False))


def main():
    args = get_args()
    if args.cmd == 'prepare':
        prepare()
    elif args.cmd == 'args':
        emit_args(args.experiment, args.seed, args.exp_dir)
    elif args.cmd == 'list':
        list_experiments()


if __name__ == '__main__':
    main()
