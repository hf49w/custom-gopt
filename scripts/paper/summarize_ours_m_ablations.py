import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'paper'))

import evaluate_paper_models as epm
from prepare_ours_m_ablations import ABLATIONS, ABLATION_ROOT, FROZEN_PATH


SENTENCE_DIMS = ['accuracy', 'fluency', 'prosodic', 'total']
WORD_DIMS = ['accuracy', 'stress', 'total']
KEY_METRICS = [
    ('sentence', 'accuracy'),
    ('sentence', 'fluency'),
    ('sentence', 'prosodic'),
    ('sentence', 'total'),
    ('word', 'accuracy'),
    ('word', 'stress'),
    ('word', 'total'),
    ('phone', 'phone'),
]


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    path = Path(path)
    if not path.exists():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_parameter_count(path):
    path = Path(path)
    if not path.exists():
        return ''
    weights = epm.torch.load(path, map_location='cpu')
    if isinstance(weights, dict) and 'model_state' in weights:
        weights = weights['model_state']
    elif isinstance(weights, dict) and 'model' in weights:
        weights = weights['model']
    if not isinstance(weights, dict):
        return ''
    return int(sum(int(value.numel()) for value in weights.values() if hasattr(value, 'numel')))


def line_count(path):
    path = Path(path)
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


def read_records(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ['experiment_id'])
        writer.writeheader()
        writer.writerows(rows)


def pcc(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    if pred.size < 2 or np.std(pred) <= 1e-12 or np.std(target) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def scalar_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return {'n': 0, 'pcc': '', 'mse': '', 'mae': ''}
    return {
        'n': int(mask.sum()),
        'pcc': pcc(pred[mask], target[mask]),
        'mse': float(np.mean((pred[mask] - target[mask]) ** 2)),
        'mae': float(np.mean(np.abs(pred[mask] - target[mask]))),
    }


def sentence_value(row, source, dim):
    sent = row.get(source, {}).get('sentence', {})
    if isinstance(sent, dict):
        return sent.get(dim)
    return None


def iter_items(records, level, metric):
    for row in records:
        if not row.get('is_final'):
            continue
        speaker = row.get('speaker_id') or 'UNKNOWN'
        if level == 'sentence':
            pred = sentence_value(row, 'predictions', metric)
            target = sentence_value(row, 'targets', metric)
            if pred is not None and target is not None:
                yield {'key': (row.get('utt_id'),), 'speaker_id': speaker, 'pred': float(pred), 'target': float(target)}
        elif level == 'word':
            dim = WORD_DIMS.index(metric)
            preds = row.get('predictions', {}).get('word', [])
            targets = row.get('targets', {}).get('word', [])
            valid = row.get('valid_slot_mask') or [1] * len(preds)
            committed = row.get('committed_slot_mask') or valid
            for idx in range(min(len(preds), len(targets), len(valid), len(committed))):
                if not valid[idx] or not committed[idx]:
                    continue
                pred = preds[idx].get(metric) if isinstance(preds[idx], dict) else preds[idx][dim]
                target = targets[idx].get(metric) if isinstance(targets[idx], dict) else targets[idx][dim]
                if target is not None and float(target) >= 0:
                    yield {'key': (row.get('utt_id'), idx), 'speaker_id': speaker, 'pred': float(pred), 'target': float(target)}
        else:
            preds = row.get('predictions', {}).get('phone')
            targets = row.get('targets', {}).get('phone')
            if preds is None or targets is None:
                continue
            valid = row.get('valid_slot_mask') or [1] * len(preds)
            committed = row.get('committed_slot_mask') or valid
            for idx in range(min(len(preds), len(targets), len(valid), len(committed))):
                if valid[idx] and committed[idx] and float(targets[idx]) >= 0:
                    yield {'key': (row.get('utt_id'), idx), 'speaker_id': speaker, 'pred': float(preds[idx]), 'target': float(targets[idx])}


def paired_delta(ablation_records, full_records, level, metric, samples=500, seed=1337):
    ab = {item['key']: item for item in iter_items(ablation_records, level, metric)}
    full = {item['key']: item for item in iter_items(full_records, level, metric)}
    keys = sorted(set(ab) & set(full))
    if len(keys) < 2:
        return {'delta': '', 'ci_low': '', 'ci_high': '', 'paired_n': len(keys)}
    items = []
    for key in keys:
        items.append({
            'speaker_id': ab[key]['speaker_id'],
            'ab_pred': ab[key]['pred'],
            'full_pred': full[key]['pred'],
            'target': ab[key]['target'],
        })
    def delta(cur):
        return pcc([x['ab_pred'] for x in cur], [x['target'] for x in cur]) - pcc([x['full_pred'] for x in cur], [x['target'] for x in cur])
    by_spk = defaultdict(list)
    for item in items:
        by_spk[item['speaker_id']].append(item)
    speakers = sorted(by_spk)
    rng = random.Random(seed)
    vals = []
    for _ in range(samples):
        cur = []
        for _spk in speakers:
            cur.extend(by_spk[rng.choice(speakers)])
        vals.append(delta(cur))
    vals = sorted(v for v in vals if math.isfinite(v))
    d = delta(items)
    return {
        'delta': d,
        'ci_low': vals[int(0.025 * (len(vals) - 1))] if vals else d,
        'ci_high': vals[int(0.975 * (len(vals) - 1))] if vals else d,
        'paired_n': len(keys),
    }


def exp_dir_for(experiment, seed):
    return ABLATION_ROOT / 'checkpoints' / f'{experiment}_seed{seed}'


def evaluate_if_needed(experiment, seed, split, exp_dir, out_path, state_update_mode='incremental'):
    frozen = read_json(FROZEN_PATH)
    data_dir = Path(frozen['strict_prefix_train_val_test_data_dir'])
    shared_utts = epm.unique_test_list(data_dir, split)
    expected = len(epm.read_manifest(data_dir / f'{split}_manifest.jsonl'))
    if out_path.exists() and line_count(out_path) == expected:
        return read_records(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = epm.evaluate_pcn_experiment(
        experiment,
        exp_dir,
        split,
        shared_utts,
        epm.torch.device('cpu'),
        out_path,
        0,
        state_update_mode=state_update_mode,
    )
    return records


def gate_analysis_row(experiment, seed, records):
    gate, pcn_entropy, acoustic_entropy, js, phone_pred, phone_target = [], [], [], [], [], []
    for row in records:
        valid = row.get('valid_slot_mask') or []
        committed = row.get('committed_slot_mask') or valid
        gate_slots = row.get('gate', {}).get('slot', [])
        stats = row.get('input_stats', {})
        phone_p = row.get('predictions', {}).get('phone', [])
        phone_t = row.get('targets', {}).get('phone', [])
        n = min(len(gate_slots), len(valid), len(committed))
        for idx in range(n):
            if not valid[idx] or not committed[idx]:
                continue
            gate.append(float(gate_slots[idx]))
            if idx < len(stats.get('pcn_entropy', [])):
                pcn_entropy.append(float(stats['pcn_entropy'][idx]))
            if idx < len(stats.get('acoustic_entropy', [])):
                acoustic_entropy.append(float(stats['acoustic_entropy'][idx]))
            if idx < len(stats.get('pcn_acoustic_js', [])):
                js.append(float(stats['pcn_acoustic_js'][idx]))
            if idx < len(phone_p) and idx < len(phone_t) and float(phone_t[idx]) >= 0:
                phone_pred.append(float(phone_p[idx]))
                phone_target.append(float(phone_t[idx]))
    gate = np.asarray(gate, dtype=np.float64)
    if gate.size == 0:
        return {'experiment_id': experiment, 'seed': seed, 'gate_count': 0}
    def corr(values):
        values = np.asarray(values[:gate.size], dtype=np.float64)
        return pcc(gate[:values.size], values) if values.size > 1 else ''
    row = {
        'experiment_id': experiment,
        'seed': seed,
        'gate_count': int(gate.size),
        'gate_mean': float(np.mean(gate)),
        'gate_std': float(np.std(gate)),
        'gate_q05': float(np.quantile(gate, 0.05)),
        'gate_q25': float(np.quantile(gate, 0.25)),
        'gate_q50': float(np.quantile(gate, 0.50)),
        'gate_q75': float(np.quantile(gate, 0.75)),
        'gate_q95': float(np.quantile(gate, 0.95)),
        'gate_pcn_entropy_pcc': corr(pcn_entropy),
        'gate_acoustic_entropy_pcc': corr(acoustic_entropy),
        'gate_js_pcc': corr(js),
    }
    if pcn_entropy and phone_pred:
        ent = np.asarray(pcn_entropy[:len(phone_pred)], dtype=np.float64)
        pred = np.asarray(phone_pred, dtype=np.float64)
        target = np.asarray(phone_target, dtype=np.float64)
        qs = np.quantile(ent, [1 / 3, 2 / 3])
        for label, mask in [
            ('low_uncertainty', ent <= qs[0]),
            ('mid_uncertainty', (ent > qs[0]) & (ent <= qs[1])),
            ('high_uncertainty', ent > qs[1]),
        ]:
            row[label + '_phone_pcc'] = pcc(pred[mask], target[mask]) if mask.any() else ''
            row[label + '_n'] = int(mask.sum())
    return row


def main():
    frozen = read_json(FROZEN_PATH)
    seeds = [int(seed) for seed in frozen['official_seed_set']]
    primary_seed = int(frozen['primary_seed'])
    summary_rows = []
    by_seed_rows = []
    gate_rows = []
    metric_json = {}
    full_pred_root = REPO_ROOT / 'paper_experiments' / 'predictions' / 'main_comparison'

    for experiment, spec in ABLATIONS.items():
        target_seeds = [primary_seed] if spec['requires_retraining'] else [primary_seed]
        for seed in target_seeds:
            if spec['requires_retraining']:
                exp_dir = exp_dir_for(experiment, seed)
                complete = (exp_dir / 'models' / 'best_audio_model.pth').exists() and (exp_dir / 'test_metrics.json').exists()
            else:
                exp_dir = Path(frozen['resolved_config_path']).parent
                complete = True
            if not complete:
                summary_rows.append({
                    'experiment_id': experiment,
                    'seed_count': 0,
                    'changed_factor': spec['changed_factor'],
                    'completion_status': 'missing',
                    'charsiu_mode': frozen.get('charsiu_mode'),
                    'config_diff_status': 'pending',
                })
                continue
            state_mode = spec.get('state_update_mode', 'incremental')
            test_pred = ABLATION_ROOT / 'predictions' / f'{experiment}_seed{seed}_test.jsonl'
            val_pred = ABLATION_ROOT / 'predictions' / f'{experiment}_seed{seed}_val.jsonl'
            records = evaluate_if_needed(experiment, seed, 'test', exp_dir, test_pred, state_update_mode=state_mode)
            evaluate_if_needed(experiment, seed, 'val', exp_dir, val_pred, state_update_mode=state_mode)
            full_records = read_records(full_pred_root / f'Ours-M_seed{seed}.jsonl')
            exp_metrics = {}
            flat = {
                'experiment_id': experiment,
                'seed': seed,
                'seed_count': 1,
                'changed_factor': spec['changed_factor'],
                'completion_status': 'done',
                'charsiu_mode': frozen.get('charsiu_mode'),
                'checkpoint_hash': sha256(exp_dir / 'models' / 'best_audio_model.pth') if spec['requires_retraining'] else frozen['primary_checkpoint_sha256'],
                'config_diff_status': read_json(ABLATION_ROOT / 'configs' / f'{experiment}_seed{seed}.expected_config_diff.json', {}).get('config_diff_status', 'inference_only'),
            }
            config = read_json(exp_dir / 'config.json')
            if config:
                ckpt_path = exp_dir / 'models' / 'best_audio_model.pth'
                flat['parameter_count'] = checkpoint_parameter_count(ckpt_path)
            for level, metric in KEY_METRICS:
                items = list(iter_items(records, level, metric))
                sm = scalar_metrics([x['pred'] for x in items], [x['target'] for x in items])
                delta = paired_delta(records, full_records, level, metric)
                prefix = f'{level}_{metric}'
                for key, value in sm.items():
                    flat[f'{prefix}_{key}'] = value
                flat[f'{prefix}_delta_vs_m_full'] = delta['delta']
                flat[f'{prefix}_delta_ci_low'] = delta['ci_low']
                flat[f'{prefix}_delta_ci_high'] = delta['ci_high']
                flat[f'{prefix}_paired_n'] = delta['paired_n']
                exp_metrics[prefix] = {**sm, 'delta_vs_m_full': delta}
            coverage_rows = epm.coverage_metrics(records)
            for row in coverage_rows:
                if row.get('level') == 'streaming':
                    flat['adjacent_score_delta_mean'] = row.get('adjacent_score_delta', '')
                    flat['phone_revision_rate'] = row.get('phone_revision_rate', '')
                    flat['word_revision_rate'] = row.get('word_revision_rate', '')
                    flat['first_stable_chunk'] = row.get('first_stable_chunk', '')
            by_seed_rows.append(flat)
            summary_rows.append(flat)
            gate_rows.append(gate_analysis_row(experiment, seed, records))
            metric_json[f'{experiment}_seed{seed}'] = {
                'test_predictions': str(test_pred),
                'val_predictions': str(val_pred),
                'metrics': exp_metrics,
                'coverage_metrics': coverage_rows,
            }
    write_csv(REPO_ROOT / 'paper_experiments' / 'tables' / 'ablation_results_by_seed.csv', by_seed_rows)
    write_csv(REPO_ROOT / 'paper_experiments' / 'tables' / 'ablation_results.csv', summary_rows)
    write_csv(ABLATION_ROOT / 'gate_analysis.csv', gate_rows)
    (ABLATION_ROOT / 'metrics' / 'ablation_metrics.json').write_text(json.dumps(metric_json, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = [
        '# Ours-M Ablation Report',
        '',
        'Full reference is Ours-M / M-Full (`M_stress_scalar_gate_capacity64`). Ours-H is not used as the reference.',
        '',
        'Single-seed rows are preliminary. Multi-seed conclusions must follow `multiseed_selection_rule.yaml`.',
        '',
        f'Generated at {time.strftime("%Y-%m-%d %H:%M:%S CST")}.',
    ]
    (ABLATION_ROOT / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    manifest = read_json(ABLATION_ROOT / 'run_manifest.json', {})
    manifest.update({
        'summarized_at': time.strftime('%Y-%m-%d %H:%M:%S CST'),
        'summary_outputs': [
            str(REPO_ROOT / 'paper_experiments' / 'tables' / 'ablation_results.csv'),
            str(REPO_ROOT / 'paper_experiments' / 'tables' / 'ablation_results_by_seed.csv'),
            str(ABLATION_ROOT / 'gate_analysis.csv'),
        ],
    })
    (ABLATION_ROOT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'summarized', 'rows': len(summary_rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
