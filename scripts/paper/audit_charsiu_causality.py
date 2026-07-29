import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description='Audit Charsiu full-WAV vs strict-prefix causal data assumptions.')
    parser.add_argument('--full-data-dir', type=Path, default=REPO_ROOT / 'data' / 'streaming_pcn_gopt_v2_stateful')
    parser.add_argument('--prefix-data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'paper_experiments')
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_manifest(data_dir, split):
    path = Path(data_dir) / f'{split}_manifest.jsonl'
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n = min(p.shape[-1], q.shape[-1])
    if n == 0:
        return 0.0
    p = p[..., :n]
    q = q[..., :n]
    p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-8, None)
    q = q / np.clip(q.sum(axis=-1, keepdims=True), 1e-8, None)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(np.clip(p, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))), axis=-1) + 0.5 * np.sum(
        q * (np.log(np.clip(q, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))),
        axis=-1,
    )


def paired_rows(full_dir, prefix_dir, split, sample_size, seed):
    full_rows = read_manifest(full_dir, split)
    prefix_rows = read_manifest(prefix_dir, split)
    prefix_by_key = {(row.get('utt_id'), int(row.get('chunk_id', -1))): idx for idx, row in enumerate(prefix_rows)}
    pairs = []
    for idx, row in enumerate(full_rows):
        key = (row.get('utt_id'), int(row.get('chunk_id', -1)))
        if key in prefix_by_key:
            pairs.append((idx, prefix_by_key[key], row))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:sample_size]


def compare_npz(full_dir, prefix_dir, split, pairs):
    if not pairs:
        return {'paired_chunks': 0}
    with np.load(Path(full_dir) / f'{split}_chunks.npz', allow_pickle=True) as full_npz, np.load(Path(prefix_dir) / f'{split}_chunks.npz', allow_pickle=True) as prefix_npz:
        js_values = []
        acoustic_js_values = []
        visible_delta = []
        for full_idx, prefix_idx, _ in pairs:
            visible = min(int(full_npz['visible_len'][full_idx]), int(prefix_npz['visible_len'][prefix_idx]))
            if visible <= 0:
                continue
            js = js_divergence(full_npz['cn_post'][full_idx, :visible], prefix_npz['cn_post'][prefix_idx, :visible])
            js_values.extend(np.asarray(js).reshape(-1).tolist())
            acoustic_js = js_divergence(full_npz['acoustic_post'][full_idx, :visible], prefix_npz['acoustic_post'][prefix_idx, :visible])
            acoustic_js_values.extend(np.asarray(acoustic_js).reshape(-1).tolist())
            visible_delta.append(abs(int(full_npz['visible_len'][full_idx]) - int(prefix_npz['visible_len'][prefix_idx])))
    return {
        'paired_chunks': len(pairs),
        'pcn_posterior_js_mean': float(np.mean(js_values)) if js_values else 0.0,
        'pcn_posterior_js_p95': float(np.percentile(js_values, 95)) if js_values else 0.0,
        'acoustic_posterior_js_mean': float(np.mean(acoustic_js_values)) if acoustic_js_values else 0.0,
        'alignment_visible_len_abs_delta_mean': float(np.mean(visible_delta)) if visible_delta else 0.0,
    }


def write_no_overwrite(path, text, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; pass --overwrite to replace.')
    path.write_text(text, encoding='utf-8')


def render_report(payload):
    lines = [
        '# Charsiu Causality Audit',
        '',
        f"Full data: `{payload['full_data_dir']}`",
        f"Full data Charsiu mode: `{payload['full_metadata_charsiu_mode']}`",
        f"Prefix data: `{payload.get('prefix_data_dir') or ''}`",
        '',
    ]
    if payload['full_wav_future_context_risk']:
        lines.append('Existing data should be treated as `full_wav_precomputed`: Charsiu posteriors were generated on the complete WAV and truncated per prefix.')
    else:
        lines.append('Existing data metadata indicates strict prefix Charsiu mode.')
    if payload.get('prefix_comparison'):
        lines.extend(['', '## Full vs Prefix Comparison', ''])
        for split, row in payload['prefix_comparison'].items():
            lines.append(f"- {split}: paired chunks={row.get('paired_chunks', 0)}, acoustic JS mean={row.get('acoustic_posterior_js_mean', 0.0):.6f}.")
    else:
        lines.extend([
            '',
            'No strict-prefix data directory was supplied or available. Generate it with `scripts/server/run_prefix_charsiu_causal_data_252.sh smoke` first, then full shards when an allowed idle GPU is available.',
        ])
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    full_meta = read_json(args.full_data_dir / 'metadata.json', {})
    prefix_meta = read_json(args.prefix_data_dir / 'metadata.json', {}) if args.prefix_data_dir else None
    mode = full_meta.get('charsiu_mode', 'full_wav_precomputed') if isinstance(full_meta, dict) else 'unknown'
    payload = {
        'full_data_dir': str(args.full_data_dir),
        'full_metadata_charsiu_mode': mode,
        'full_wav_future_context_risk': mode in {'full_wav_precomputed', 'unknown'} or mode is None,
        'prefix_data_dir': str(args.prefix_data_dir) if args.prefix_data_dir else None,
        'prefix_metadata': prefix_meta,
        'assertion': 'prefix_recompute mode must call Charsiu only on arrays loaded with duration <= audio_end_t; cache metadata stores source_num_samples and max_allowed_samples.',
    }
    if args.prefix_data_dir and args.prefix_data_dir.exists():
        comparison = {}
        for split in ['train', 'val', 'test']:
            pairs = paired_rows(args.full_data_dir, args.prefix_data_dir, split, args.sample_size, args.seed)
            comparison[split] = compare_npz(args.full_data_dir, args.prefix_data_dir, split, pairs)
        payload['prefix_comparison'] = comparison
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_no_overwrite(args.output_dir / 'causality_audit.json', json.dumps(payload, ensure_ascii=False, indent=2), overwrite=args.overwrite)
    write_no_overwrite(args.output_dir / 'causality_audit_report.md', render_report(payload), overwrite=args.overwrite)
    print(json.dumps({'causality_audit': str(args.output_dir / 'causality_audit.json')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
