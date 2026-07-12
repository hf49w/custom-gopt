import argparse
import csv
import json
from pathlib import Path


CORE_KEYS = [
    'coverage_100_mae',
    'coverage_100_pcc',
    'coverage_90_mae',
    'coverage_90_pcc',
    'coverage_80_mae',
    'coverage_80_pcc',
    'coverage_70_mae',
    'coverage_70_pcc',
    'mean_adjacent_utt_delta',
    'phone_revision_rate',
    'word_revision_rate',
    'mean_effective_supervision_weight',
    'supervised_slot_count',
]
ARG_KEYS = [
    'utt_dim_weights',
    'soft_label_policy',
    'utt_pooling_head',
    'fusion_mode',
    'embed_dim',
    'depth',
    'heads',
    'gru_dim',
    'loss_w_teacher_score',
    'loss_w_prefix_kd',
    'loss_w_oracle_phone',
    'loss_w_oracle_word',
    'loss_w_oracle_utt_prefix',
    'loss_w_oracle_utt_final',
    'word_dim_weights',
    'teacher_word_dim_weights',
    'oracle_word_dim_weights',
    'loss_w_stress_pearson',
    'loss_w_oracle_stress_pearson',
    'loss_w_teacher_stress_pearson',
    'loss_w_stress_rank',
    'loss_w_oracle_stress_rank',
    'stress_loss_mask',
    'stress_branch',
    'stress_grad_scale',
]


def get_args():
    parser = argparse.ArgumentParser(description='Summarize PCN extra experiment metrics.')
    parser.add_argument('--exp-root', type=Path, required=True)
    parser.add_argument('--baseline-json', type=Path, default=None)
    parser.add_argument('--output-csv', type=Path, required=True)
    return parser.parse_args()


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return default


def best_val_loss(history):
    values = []
    for row in history if isinstance(history, list) else []:
        val = row.get('val', {}) if isinstance(row, dict) else {}
        loss = val.get('loss')
        if isinstance(loss, (int, float)):
            values.append(float(loss))
    return min(values) if values else ''


def args_summary(config):
    args = config.get('args', {}) if isinstance(config, dict) else {}
    parts = []
    for key in ARG_KEYS:
        if key in args:
            value = args[key]
        elif key in config:
            value = config[key]
        else:
            continue
        if isinstance(value, list):
            value = ','.join(str(item) for item in value)
        parts.append(f'{key}={value}')
    return ';'.join(parts)


def metric_columns(rows):
    keys = set(CORE_KEYS)
    for row in rows:
        metrics = row.get('_metrics', {})
        for key in metrics:
            low = key.lower()
            if 'pcc' in low or 'mae' in low or low.startswith(('final', 'full')):
                keys.add(key)
    return [key for key in CORE_KEYS if key in keys] + sorted(keys - set(CORE_KEYS))


def make_row(name, path, metrics, history, config):
    return {
        'exp': name,
        'path': str(path),
        'best_val_loss': best_val_loss(history),
        'args_summary': args_summary(config),
        '_metrics': metrics,
    }


def main():
    args = get_args()
    rows = []
    if args.baseline_json:
        rows.append(make_row('baseline', args.baseline_json.parent, read_json(args.baseline_json, {}), [], {}))
    for exp_dir in sorted(path for path in args.exp_root.iterdir() if path.is_dir()):
        metrics = read_json(exp_dir / 'test_metrics.json', {})
        history = read_json(exp_dir / 'history.json', [])
        config = read_json(exp_dir / 'config.json', {})
        rows.append(make_row(exp_dir.name, exp_dir, metrics, history, config))

    metric_keys = metric_columns(rows)
    fieldnames = ['exp', 'path', 'best_val_loss'] + metric_keys + ['args_summary']
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, '') for key in ['exp', 'path', 'best_val_loss', 'args_summary']}
            metrics = row.get('_metrics', {})
            for key in metric_keys:
                flat[key] = metrics.get(key, '')
            writer.writerow(flat)


if __name__ == '__main__':
    main()
