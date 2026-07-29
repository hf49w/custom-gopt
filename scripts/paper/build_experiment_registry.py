import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_ROOT = REPO_ROOT / 'exp' / 'pcn_extra_20260704_2130'
DEFAULT_CORRECTED_ROOT = REPO_ROOT / 'exp' / 'pcn_extra_correct_multipa_20260710'
DEFAULT_STREAMING_DATA_DIR = REPO_ROOT / 'data' / 'streaming_pcn_gopt_v2_stateful'
FORBIDDEN_MULTIPA_ROOT = Path('/DATA_2/guest/MultiPA_pic')
CORRECT_MULTIPA_ROOT = Path('/DATA_2/MultiPA')

SENTENCE_DIMS = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
WORD_DIMS = ['accuracy', 'stress', 'total']
EXPERIMENT_IDS = [
    'A_loss_dimmask',
    'B_relaxed_softlabel',
    'C_oracle_word_phone',
    'D_oracle_sentence_light',
    'E_oracle_sentence_balanced',
    'F_oracle_vector_gate',
    'G_oracle_capacity64',
    'H_stress_weighted_G',
    'I_stress_corr_G',
    'J_stress_detached_branch',
    'K_slot_prosody_stress',
    'L_stress_gradscale_voiced',
    'M_stress_scalar_gate_capacity64',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Build a reproducible manifest and audit report for PCN paper experiments.')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--legacy-exp-root', type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument('--corrected-exp-root', type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument('--streaming-data-dir', type=Path, default=DEFAULT_STREAMING_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'paper_experiments')
    parser.add_argument('--include-npz-summary', action='store_true', help='Also sum teacher/oracle masks in NPZ files; slower on compressed full data.')
    parser.add_argument('--include-data-summaries', action='store_true', help='Open each experiment data directory for metadata/NPZ summaries.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def safe_json_load(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception as exc:
        return {'_read_error': str(exc), '_path': str(path)}


def read_jsonl_head(path, limit=5):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def iter_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def command_output(args, cwd):
    try:
        proc = subprocess.run(args, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            return None, (proc.stderr or proc.stdout).strip()
        return proc.stdout.strip(), None
    except Exception as exc:
        return None, str(exc)


def repo_state(repo_root):
    commit, err = command_output(['git', 'rev-parse', 'HEAD'], repo_root)
    status, status_err = command_output(['git', 'status', '--short'], repo_root)
    branch, _ = command_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], repo_root)
    return {
        'git_commit': commit or 'unavailable',
        'git_branch': branch or 'unavailable',
        'git_status': status if status is not None else 'not_a_git_worktree',
        'git_error': err or status_err or '',
    }


def scalar_summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {'count': 0}
    values = sorted(values)
    return {
        'count': len(values),
        'min': values[0],
        'p25': values[int(0.25 * (len(values) - 1))],
        'mean': float(statistics.fmean(values)),
        'p50': values[int(0.50 * (len(values) - 1))],
        'p75': values[int(0.75 * (len(values) - 1))],
        'max': values[-1],
    }


def speaker_from_row(row):
    if row.get('speaker_id'):
        return str(row['speaker_id']).upper()
    wav_path = str(row.get('wav_path') or row.get('audio_path') or '')
    match = re.search(r'(SPEAKER\d+)', wav_path, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    parts = Path(wav_path).parts
    for part in reversed(parts):
        if part.upper().startswith('SPEAKER'):
            return part.upper()
    utt = str(row.get('utt_id', ''))
    return utt.split('_')[0].upper() if '_' in utt else 'UNKNOWN'


def read_manifest_rows(data_dir, split):
    path = Path(data_dir) / f'{split}_manifest.jsonl'
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def summarize_manifest(data_dir):
    out = {}
    for split in ['train', 'val', 'test']:
        path = Path(data_dir) / f'{split}_manifest.jsonl'
        utts = set()
        speakers = set()
        duration_by_utt = defaultdict(float)
        coverage = []
        rows = 0
        if path.exists():
            iterator = iter_jsonl(path)
        else:
            iterator = iter(())
        for row in iterator:
            rows += 1
            utt = str(row.get('utt_id', ''))
            if utt:
                utts.add(utt)
                duration_by_utt[utt] = max(duration_by_utt[utt], float(row.get('audio_end') or row.get('commit_time') or 0.0))
            speakers.add(speaker_from_row(row))
            if row.get('coverage_ratio') is not None:
                coverage.append(float(row['coverage_ratio']))
        utts_sorted = sorted(utts)
        speakers_sorted = sorted(speakers)
        out[split] = {
            'manifest': str(path),
            'rows': rows,
            'utterances': len(utts_sorted),
            'speakers': len(speakers_sorted),
            'speaker_ids': speakers_sorted,
            'duration_sec': scalar_summary(duration_by_utt.values()),
            'coverage_ratio': scalar_summary(coverage),
            'sample_utterance_ids': utts_sorted[:10],
        }
    return out


def target_distribution(scores, utt_ids=None):
    if not isinstance(scores, dict):
        return {}
    selected = set(str(x) for x in utt_ids) if utt_ids is not None else None
    out = {}
    for dim in SENTENCE_DIMS:
        vals = []
        for utt_id, row in scores.items():
            if selected is not None and str(utt_id) not in selected:
                continue
            if isinstance(row, dict) and row.get(dim) is not None:
                vals.append(float(row[dim]))
        out[f'utt_{dim}'] = scalar_summary(vals)
    word_vals = {dim: [] for dim in WORD_DIMS}
    phone_vals = []
    for utt_id, row in scores.items():
        if selected is not None and str(utt_id) not in selected:
            continue
        for word in row.get('words', []) if isinstance(row, dict) else []:
            if word.get('accuracy') is not None:
                word_vals['accuracy'].append(float(word['accuracy']))
            if word.get('stress') is not None:
                word_vals['stress'].append(float(word['stress']))
            if word.get('total') is not None:
                word_vals['total'].append(float(word['total']))
            for score in word.get('phones-accuracy', []) or []:
                phone_vals.append(float(score))
    for dim, vals in word_vals.items():
        out[f'word_{dim}'] = scalar_summary(vals)
    out['phone_accuracy'] = scalar_summary(phone_vals)
    return out


def csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def compact_metrics(metrics):
    if not isinstance(metrics, dict):
        return {}
    keep = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            keep[key] = float(value)
    return keep


def npz_field_summary(data_dir, include_sums=False):
    data_dir = Path(data_dir)
    out = {}
    for split in ['train', 'val', 'test']:
        path = data_dir / f'{split}_chunks.npz'
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                fields = list(data.files)
                cur = {'path': str(path), 'rows': int(data[fields[0]].shape[0]) if fields else 0, 'fields': fields}
                if include_sums:
                    for mask_name in ['teacher_utt_mask', 'oracle_utt_mask', 'teacher_word_mask', 'oracle_word_mask']:
                        if mask_name in data.files:
                            cur[f'{mask_name}_sum'] = float(np.asarray(data[mask_name]).sum())
                    for dim_mask in ['teacher_utt_dim_mask', 'oracle_utt_dim_mask']:
                        if dim_mask in data.files and np.asarray(data[dim_mask]).ndim >= 2:
                            cur[f'{dim_mask}_completeness_sum'] = float(np.asarray(data[dim_mask])[:, 1].sum())
                out[split] = cur
        except Exception as exc:
            out[split] = {'path': str(path), 'error': str(exc)}
    return out


def resolve_exp_dir(exp_root, experiment_id):
    candidates = []
    if experiment_id in {'A_loss_dimmask', 'B_relaxed_softlabel'}:
        candidates.extend([exp_root / 'runs' / experiment_id, exp_root / experiment_id])
    elif experiment_id[0] in {'C', 'D', 'E', 'F', 'G'}:
        candidates.extend([exp_root / 'oracle_runs' / experiment_id, exp_root / experiment_id])
    else:
        candidates.extend([exp_root / 'stress_runs' / experiment_id, exp_root / experiment_id])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def experiment_status(exp_dir):
    if (exp_dir / 'test_metrics.json').exists():
        return 'done'
    if (exp_dir / 'models' / 'best_audio_model.pth').exists() or (exp_dir / 'last_checkpoint.pt').exists():
        return 'checkpoint_without_test_metrics'
    if exp_dir.exists():
        return 'created'
    return 'missing'


def teacher_summary(exp_root, data_dir):
    data_dir = Path(data_dir) if data_dir else None
    summary = {
        'multipa_condition': 'embedded_teacher_fields_if_present',
        'multipa_source_root': None,
        'uses_forbidden_multipa_pic': False,
        'oracle_condition': 'not_oracle',
        'oracle_jsonl': None,
    }
    if data_dir:
        text = str(data_dir)
        if 'correct_multipa' in text or 'pcn_extra_correct_multipa' in str(exp_root):
            summary['multipa_source_root'] = str(CORRECT_MULTIPA_ROOT)
        elif 'pcn_extra_20260704_2130' in text:
            summary['multipa_source_root'] = 'legacy_unknown_or_pre_correction'
        if FORBIDDEN_MULTIPA_ROOT.as_posix() in text:
            summary['uses_forbidden_multipa_pic'] = True
    teacher_jsonl = Path(exp_root) / 'teacher_multipa_correct' / 'multipa_train_val.jsonl'
    if teacher_jsonl.exists():
        summary['multipa_jsonl'] = str(teacher_jsonl)
    legacy_teacher = REPO_ROOT / 'data' / 'streaming_pcn_gopt_v2_stateful' / 'teacher_multipa' / 'multipa_train_val.jsonl'
    if legacy_teacher.exists() and 'multipa_jsonl' not in summary:
        summary['multipa_jsonl'] = str(legacy_teacher)
    oracle_jsonl = Path(exp_root) / 'oracle_gopt_closed_prefix_gt_time_all_splits.jsonl'
    if oracle_jsonl.exists() and data_dir and ('oracle' in str(data_dir) or 'oracle' in str(exp_root)):
        summary['oracle_condition'] = 'GOPT_closed_oracle_GT_time_teacher_for_distillation_only'
        summary['oracle_jsonl'] = str(oracle_jsonl)
    return summary


def scan_experiment(exp_root, experiment_id, family, repo_info):
    exp_dir = resolve_exp_dir(exp_root, experiment_id)
    config_path = exp_dir / 'config.json'
    config = safe_json_load(config_path, {})
    args = config.get('args', {}) if isinstance(config, dict) else {}
    data_dir = Path(config.get('data_dir', '')) if isinstance(config, dict) and config.get('data_dir') else None
    metadata = safe_json_load(data_dir / 'metadata.json', {}) if data_dir else {}
    metrics = compact_metrics(safe_json_load(exp_dir / 'test_metrics.json', {}))
    checkpoint = exp_dir / 'models' / 'best_audio_model.pth'
    if not checkpoint.exists():
        checkpoint = exp_dir / 'last_checkpoint.pt'
    return {
        'experiment_id': experiment_id,
        'family': family,
        'model': 'PCNStreamingScorer',
        'seed': args.get('seed', config.get('seed')) if isinstance(config, dict) else None,
        'data_dir': str(data_dir) if data_dir else '',
        'split': 'test',
        'checkpoint': str(checkpoint) if checkpoint.exists() else '',
        'config': str(config_path) if config_path.exists() else '',
        'teacher': teacher_summary(exp_root, data_dir),
        'chunk_sec': metadata.get('chunk_sec') if isinstance(metadata, dict) else None,
        'right_context_sec': metadata.get('right_context_sec') if isinstance(metadata, dict) else None,
        'asr_model': metadata.get('asr_model') if isinstance(metadata, dict) else None,
        'charsiu_mode': metadata.get('charsiu_mode', 'full_wav_precomputed') if isinstance(metadata, dict) else 'unknown',
        'charsiu_aligner_model': metadata.get('aligner_model') if isinstance(metadata, dict) else None,
        'git_commit': repo_info['git_commit'],
        'status': experiment_status(exp_dir),
        'metrics': metrics,
    }


def scan_baselines(repo_root, repo_info):
    eval_root = repo_root / 'downloads' / 'custom-gopt-252' / 'eval'
    rows = []
    all_jsonl = []
    if eval_root.exists():
        try:
            proc = subprocess.run(
                ['find', str(eval_root), '-type', 'f', '-name', '*.jsonl'],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            all_jsonl = [Path(line) for line in proc.stdout.splitlines() if line.strip()]
        except Exception:
            all_jsonl = []
    patterns = {
        'MultiPA': ['multipa'],
        'GOPT-open-base': ['open', 'base'],
        'GOPT-open-medium': ['open', 'medium'],
        'GOPT-closed-oracle': ['closed', 'oracle'],
    }
    for name, tokens in patterns.items():
        found = []
        for path in all_jsonl:
            low = str(path).lower()
            if all(token in low for token in tokens):
                found.append(path)
        rows.append({
            'experiment_id': name,
            'family': 'baseline',
            'model': name,
            'seed': None,
            'data_dir': '',
            'split': 'test',
            'checkpoint': '',
            'config': '',
            'teacher': {
                'condition': 'GT-oracle' if 'closed' in name.lower() else 'GT-free',
                'source_files': [str(path) for path in found[:20]],
                'uses_forbidden_multipa_pic': any(FORBIDDEN_MULTIPA_ROOT.as_posix() in str(path) for path in found),
            },
            'chunk_sec': None,
            'right_context_sec': None,
            'asr_model': None,
            'charsiu_mode': None,
            'git_commit': repo_info['git_commit'],
            'status': 'predictions_found' if found else 'missing',
            'metrics': {},
        })
    return rows


def yaml_quote(value):
    text = str(value)
    if text == '' or any(ch in text for ch in ':#{}[],&*?|-<>=!%@\\\n') or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def to_yaml(value, indent=0):
    pad = ' ' * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f'{pad}{key}:')
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f'{pad}{key}: {to_yaml(item, 0).strip()}')
        return '\n'.join(lines)
    if isinstance(value, list):
        if not value:
            return f'{pad}[]'
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f'{pad}-')
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f'{pad}- {to_yaml(item, 0).strip()}')
        return '\n'.join(lines)
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return yaml_quote(value)


def write_text_no_overwrite(path, text, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists; pass --overwrite to replace it.')
    path.write_text(text, encoding='utf-8')


def audit_supported_metrics(repo_root):
    train_path = repo_root / 'src' / 'train_streaming_pcn.py'
    coverage_path = repo_root / 'scripts' / 'local' / 'eval_pcn_coverage_pcc.py'
    summarize_path = repo_root / 'scripts' / 'local' / 'summarize_pcn_extra_experiments.py'
    text = ''
    for path in [train_path, coverage_path, summarize_path]:
        if path.exists():
            text += '\n' + path.read_text(encoding='utf-8', errors='replace')
    checks = {
        'PCC': ['pcc(', 'coverage_pcc'],
        'MSE': ['mse', 'masked_mse'],
        'MAE': ['mae', 'coverage_100_mae'],
        'prefix_coverage': ['coverage_ratio', 'coverage_'],
        'revision': ['revision_phone', 'phone_revision_rate', 'word_revision_rate'],
        'confidence': ['confidence_ece', 'confidence_brier', 'confidence_target'],
        'AURC': ['aurc', 'risk_coverage'],
        'stability': ['prefix_stability', 'first_stable_chunk', 'utt_stability'],
    }
    return {
        name: {
            'supported': any(token in text for token in tokens),
            'evidence_tokens': [token for token in tokens if token in text],
        }
        for name, tokens in checks.items()
    }


def audit_original_vs_streaming(streaming_data_dir):
    metadata = safe_json_load(Path(streaming_data_dir) / 'metadata.json', {})
    scores_path = Path(metadata.get('scores_json', REPO_ROOT / 'src' / 'prep_data' / 'scores.json')) if isinstance(metadata, dict) else REPO_ROOT / 'src' / 'prep_data' / 'scores.json'
    if not scores_path.is_absolute():
        scores_path = REPO_ROOT / scores_path
    scores = safe_json_load(scores_path, {})
    split_summary = summarize_manifest(streaming_data_dir)
    split_utts = {split: set() for split in ['train', 'val', 'test']}
    for split in ['train', 'val', 'test']:
        path = Path(streaming_data_dir) / f'{split}_manifest.jsonl'
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            if row.get('utt_id'):
                split_utts[split].add(str(row['utt_id']))
    original_scored = set(str(key) for key in scores.keys()) if isinstance(scores, dict) else set()
    test_utts = split_utts.get('test', set())
    all_streaming = set().union(*split_utts.values())
    return {
        'scores_json': str(scores_path),
        'original_scored_utterances': len(original_scored),
        'streaming_split_summary': split_summary,
        'test_intersection_with_scores': len(test_utts & original_scored),
        'test_missing_from_scores': sorted(test_utts - original_scored)[:50],
        'scores_missing_from_streaming_any_split': sorted(original_scored - all_streaming)[:50],
        'test_target_distribution': target_distribution(scores, test_utts),
        'all_scored_target_distribution': target_distribution(scores, original_scored),
    }


def audit_charsiu_mode(repo_root, streaming_data_dir):
    metadata = safe_json_load(Path(streaming_data_dir) / 'metadata.json', {})
    mode = metadata.get('charsiu_mode', 'full_wav_precomputed') if isinstance(metadata, dict) else 'unknown'
    return {
        'data_dir': str(streaming_data_dir),
        'metadata_charsiu_mode': mode,
        'full_wav_precomputed_risk': mode in {'full_wav_precomputed', 'unknown'} or mode is None,
        'code_evidence': [
            'build_streaming_asr_gopt_data.align_gold_utterance computes audio_logits(audio_path) on the full WAV.',
            'build_streaming_pcn_gopt_data.build_examples_for_utterance historically calls select_visible_frames(item["probs"], ..., audio_end).',
            'If metadata has no charsiu_mode, treat old data as full_wav_precomputed and therefore potentially exposed to future acoustic context in Charsiu posteriors.',
        ],
        'strict_mode_available': '--charsiu-mode prefix_recompute' in (repo_root / 'src' / 'prep_data' / 'build_streaming_pcn_gopt_data.py').read_text(encoding='utf-8', errors='replace'),
    }


def render_report(payload):
    lines = [
        '# Paper Experiment Audit',
        '',
        f"Repository: `{payload['repo']['root']}`",
        f"Git commit: `{payload['repo']['git_commit']}`",
        f"Git status: `{payload['repo']['git_status'] or 'clean'}`",
        '',
        '## Experiment Status',
        '',
        '| experiment | family | status | data | checkpoint |',
        '|---|---|---|---|---|',
    ]
    for row in payload['experiments']:
        lines.append(
            f"| {row['experiment_id']} | {row['family']} | {row['status']} | "
            f"`{row.get('data_dir', '')}` | `{row.get('checkpoint', '')}` |"
        )
    lines.extend([
        '',
        '## Teacher Compliance',
        '',
    ])
    for row in payload['experiments']:
        teacher = row.get('teacher', {})
        if row['family'] == 'baseline':
            continue
        flag = 'FORBIDDEN' if teacher.get('uses_forbidden_multipa_pic') else 'ok'
        lines.append(f"- `{row['experiment_id']}`: MultiPA source `{teacher.get('multipa_source_root')}`, oracle `{teacher.get('oracle_condition')}`, compliance `{flag}`.")
    lines.extend([
        '',
        '## Original vs Streaming Test Set',
        '',
        f"- Original scored utterances: {payload['test_set_audit']['original_scored_utterances']}",
        f"- Streaming test utterances: {payload['test_set_audit']['streaming_split_summary'].get('test', {}).get('utterances', 0)}",
        f"- Test/scored intersection: {payload['test_set_audit']['test_intersection_with_scores']}",
        '',
        '## Charsiu Causality',
        '',
        f"- Existing metadata mode: `{payload['charsiu_audit']['metadata_charsiu_mode']}`",
        f"- Full-WAV future-context risk: `{payload['charsiu_audit']['full_wav_precomputed_risk']}`",
        f"- Strict prefix mode implemented in code: `{payload['charsiu_audit']['strict_mode_available']}`",
        '',
        '## Supported Metrics Found',
        '',
    ])
    for name, row in payload['supported_metrics'].items():
        lines.append(f"- {name}: {'yes' if row['supported'] else 'no'} ({', '.join(row['evidence_tokens'])})")
    lines.extend([
        '',
        '## Implementation Plan',
        '',
        '1. Use this manifest as the immutable paper registry; add new experiment IDs instead of reusing old output directories.',
        '2. Use `scripts/paper/evaluate_paper_models.py` for shared-test-list predictions and paper metrics.',
        '3. Treat old PCN data as `full_wav_precomputed`; use `--charsiu-mode prefix_recompute` for strict causal Charsiu data.',
        '4. Keep `/DATA_2/guest/MultiPA_pic` out of teacher generation and evaluation paths.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    repo_info = repo_state(args.repo_root)
    repo_info['root'] = str(args.repo_root)

    experiments = []
    for root, family in [(args.corrected_exp_root, 'corrected_multipa'), (args.legacy_exp_root, 'legacy')]:
        if not root.exists():
            continue
        for experiment_id in EXPERIMENT_IDS:
            experiments.append(scan_experiment(root, experiment_id, family, repo_info))
    experiments.extend(scan_baselines(args.repo_root, repo_info))

    data_dirs = sorted({row['data_dir'] for row in experiments if row.get('data_dir')})
    if args.include_data_summaries:
        data_summaries = {
            path: {
                'metadata': safe_json_load(Path(path) / 'metadata.json', {}),
                'npz': npz_field_summary(path, include_sums=args.include_npz_summary),
            }
            for path in data_dirs
        }
    else:
        data_summaries = {'skipped_by_default': True, 'data_dirs': data_dirs}

    payload = {
        'repo': repo_info,
        'experiments': experiments,
        'data_summaries': data_summaries,
        'test_set_audit': audit_original_vs_streaming(args.streaming_data_dir),
        'charsiu_audit': audit_charsiu_mode(args.repo_root, args.streaming_data_dir),
        'supported_metrics': audit_supported_metrics(args.repo_root),
    }

    registry_dir = args.output_dir / 'registry'
    registry_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'custom_gopt.paper_experiment_manifest.v1',
        'generated_by': 'scripts/paper/build_experiment_registry.py',
        'repo': repo_info,
        'experiments': experiments,
    }
    write_text_no_overwrite(registry_dir / 'experiment_manifest.yaml', to_yaml(manifest) + '\n', overwrite=args.overwrite)
    write_text_no_overwrite(args.output_dir / 'audit_report.json', json.dumps(payload, ensure_ascii=False, indent=2), overwrite=args.overwrite)
    write_text_no_overwrite(args.output_dir / 'audit_report.md', render_report(payload), overwrite=args.overwrite)
    print(json.dumps({
        'manifest': str(registry_dir / 'experiment_manifest.yaml'),
        'audit_report_json': str(args.output_dir / 'audit_report.json'),
        'audit_report_md': str(args.output_dir / 'audit_report.md'),
        'experiments': len(experiments),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
