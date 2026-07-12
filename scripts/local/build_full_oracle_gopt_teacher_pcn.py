import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def get_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Build full closed-oracle GOPT teacher JSONL for PCN manifests.')
    parser.add_argument('--pcn-data-dir', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--splits', type=str, default='train,val,test')
    parser.add_argument('--oracle-source', choices=['auto', 'closed-gopt-predictions'], default='auto')
    parser.add_argument('--closed-gopt-exp', type=Path, default=None)
    parser.add_argument('--scores-json', type=Path, default=repo_root / 'src' / 'prep_data' / 'scores.json')
    parser.add_argument('--seq-data-dir', type=Path, default=repo_root / 'data' / 'seq_data_librispeech')
    parser.add_argument('--keys-root', type=Path, default=repo_root / 'data' / 'raw_kaldi_gop' / 'librispeech')
    parser.add_argument('--checkpoint', type=Path, default=repo_root / 'pretrained_models' / 'gopt_librispeech' / 'best_audio_model.pth')
    parser.add_argument('--repo-src', type=Path, default=repo_root / 'src')
    parser.add_argument('--multipa-repo-root', type=Path, default=Path(os.environ.get('MULTIPA_REPO_ROOT', '/DATA_2/MultiPA')))
    parser.add_argument('--aligner', type=str, default=str(repo_root / 'server_assets' / 'models' / 'charsiu-en_w2v2_fc_10ms'))
    parser.add_argument('--word-time-cache', type=Path, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--align-device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--time-field', choices=['audio_end', 'commit_time'], default='audio_end')
    parser.add_argument('--min-row-coverage', type=float, default=0.90)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def source_utt_id(row):
    return str(row.get('source_utt_id') or str(row.get('utt_id', '')).split('_c', 1)[0])


def load_key_order(path):
    if not path.exists():
        return []
    order = []
    previous = None
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            utt_id = line.split(',', 1)[0].rsplit('.', 1)[0]
            if utt_id != previous:
                order.append(utt_id)
                previous = utt_id
    return order


def require_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f'Missing {description}: {path}')


def choose_seq_split(manifest_rows, seq_data_dir, keys_root):
    candidates = []
    manifest_ids = [source_utt_id(row) for row in manifest_rows]
    for seq_name, prefix in [('train', 'tr'), ('test', 'te')]:
        required = [
            seq_data_dir / f'{prefix}_feat.npy',
            seq_data_dir / f'{prefix}_label_phn.npy',
            seq_data_dir / f'{prefix}_label_word.npy',
        ]
        if not all(path.exists() for path in required):
            candidates.append((seq_name, prefix, 0, None))
            continue
        ids_path = seq_data_dir / f'{prefix}_utt_ids.txt'
        keys_path = keys_root / f'{prefix}_keys_phn.csv'
        utt_ids = load_key_order(ids_path if ids_path.exists() else keys_path)
        utt_set = set(utt_ids)
        covered = sum(1 for utt_id in manifest_ids if utt_id in utt_set)
        candidates.append((seq_name, prefix, covered, keys_path))
    best = max(candidates, key=lambda item: item[2])
    if best[2] <= 0:
        detail = ', '.join(f'{name}:{covered}' for name, _, covered, _ in candidates)
        raise RuntimeError(f'No closed GOPT sequence data covers this PCN manifest. Coverage by candidate: {detail}')
    return best


def normalize_word_scores(row, manifest_row):
    word_times = []
    groups = manifest_row.get('word_timestamps') or []
    if groups:
        word_times = groups[0] or []
    out = []
    for idx, item in enumerate(row.get('word_scores') or []):
        word_idx = int(item.get('word_id', item.get('word_index', idx)))
        time_item = word_times[word_idx] if 0 <= word_idx < len(word_times) else {}
        out.append(
            {
                'word': str(time_item.get('word', item.get('word', ''))),
                **({'start': float(time_item['start'])} if 'start' in time_item else {}),
                **({'end': float(time_item['end'])} if 'end' in time_item else {}),
                'word_index': word_idx,
                'pred_accuracy': float(item.get('pred_accuracy', 0.0)),
                'pred_stress': float(item.get('pred_stress', 0.0)),
                'pred_total': float(item.get('pred_total', 0.0)),
            }
        )
    return out


def normalize_phone_scores(row, manifest_row):
    slot_times = manifest_row.get('slot_times') or []
    out = []
    for idx, item in enumerate(row.get('phone_scores') or []):
        phone_idx = int(item.get('phone_index', idx))
        time_item = slot_times[phone_idx] if 0 <= phone_idx < len(slot_times) else None
        record = {
            'phone_index': phone_idx,
            'phone': str(item.get('phone', '')),
            'score': float(item.get('score', item.get('phone_score', item.get('pred_accuracy', 0.0)))),
        }
        if isinstance(time_item, (list, tuple)) and len(time_item) >= 2:
            record['start'] = float(time_item[0])
            record['end'] = float(time_item[1])
        elif isinstance(time_item, dict):
            if 'start' in time_item:
                record['start'] = float(time_item['start'])
            if 'end' in time_item:
                record['end'] = float(time_item['end'])
        out.append(record)
    return out


def normalize_eval_row(row, manifest_by_key, split):
    key = (str(row.get('utt_id')), int(row.get('chunk_id', -1)))
    manifest_row = manifest_by_key.get(key, {})
    ok = row.get('status') == 'ok'
    scores = row.get('scores') or {}
    normalized = {
        'split': split,
        'utt_id': str(row.get('utt_id')),
        'chunk_id': int(row.get('chunk_id', -1)),
        'status': 'ok' if ok else 'skip',
        'scores': {
            'accuracy': float(scores.get('accuracy', 0.0)),
            'completeness': float(scores.get('completeness', 0.0)),
            'fluency': float(scores.get('fluency', 0.0)),
            'prosodic': float(scores.get('prosodic', 0.0)),
            'total': float(scores.get('total', 0.0)),
        },
        'word_scores': normalize_word_scores(row, manifest_row) if ok else [],
        'phone_scores': normalize_phone_scores(row, manifest_row) if ok else [],
        'prefix_available': bool(ok),
        'uses_reference_text': bool(row.get('uses_reference_text', ok)),
    }
    if not ok:
        normalized['skip_reason'] = row.get('status', 'unknown')
    return normalized


def run_evaluator(args, split, manifest_path, out_path, seq_name, keys_path):
    evaluator = Path(__file__).resolve().parents[1] / 'eval_gopt_closed_oracle_prefix.py'
    cmd = [
        sys.executable,
        str(evaluator),
        '--prefix-manifest',
        str(manifest_path),
        '--scores-json',
        str(args.scores_json),
        '--seq-data-dir',
        str(args.seq_data_dir),
        '--keys-phn-csv',
        str(keys_path),
        '--seq-split',
        seq_name,
        '--checkpoint',
        str(args.checkpoint),
        '--repo-src',
        str(args.repo_src),
        '--output-jsonl',
        str(out_path),
        '--device',
        args.device,
        '--batch-size',
        str(args.batch_size),
        '--word-count-source',
        'gt_time',
        '--time-field',
        args.time_field,
        '--multipa-repo-root',
        str(args.multipa_repo_root),
        '--aligner',
        str(args.aligner),
        '--align-device',
        args.align_device,
        '--word-time-cache',
        str(args.word_time_cache or (args.output_jsonl.parent / 'gt_word_time_cache')),
    ]
    print(json.dumps({'split': split, 'cmd': cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, check=True)


def normalize_existing_predictions(args, split, manifest_rows):
    if args.closed_gopt_exp is None:
        raise ValueError('--closed-gopt-exp is required when --oracle-source=closed-gopt-predictions')
    predictions = args.closed_gopt_exp / 'predictions.jsonl'
    require_path(predictions, 'closed GOPT predictions.jsonl')
    manifest_keys = {(str(row.get('utt_id')), int(row.get('chunk_id', -1))) for row in manifest_rows}
    selected = []
    for row in read_jsonl(predictions):
        if (str(row.get('utt_id')), int(row.get('chunk_id', -1))) in manifest_keys:
            selected.append(row)
    return selected


def main():
    args = get_args()
    splits = [item.strip() for item in args.splits.split(',') if item.strip()]
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output_jsonl} exists; pass --overwrite to rebuild.')
    require_path(args.pcn_data_dir / 'metadata.json', 'PCN metadata.json')
    require_path(args.scores_json, 'scores JSON')
    require_path(args.seq_data_dir, 'closed GOPT seq-data-dir')
    require_path(args.checkpoint, 'closed GOPT checkpoint')
    require_path(args.repo_src, 'GOPT repo src')
    if args.oracle_source == 'auto':
        require_path(args.multipa_repo_root, 'MultiPA repo root for GT-time alignment')

    all_rows = []
    stats = {}
    tmp_dir = args.output_jsonl.parent / '.oracle_build_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        manifest_path = args.pcn_data_dir / f'{split}_manifest.jsonl'
        require_path(manifest_path, f'{split} PCN manifest')
        manifest_rows = read_jsonl(manifest_path)
        manifest_by_key = {(str(row.get('utt_id')), int(row.get('chunk_id', -1))): row for row in manifest_rows}
        if args.oracle_source == 'closed-gopt-predictions':
            eval_rows = normalize_existing_predictions(args, split, manifest_rows)
        else:
            seq_name, _, covered, keys_path = choose_seq_split(manifest_rows, args.seq_data_dir, args.keys_root)
            if covered < len(manifest_rows) * args.min_row_coverage:
                raise RuntimeError(
                    f'{split}: closed GOPT {seq_name} sequence data covers {covered}/{len(manifest_rows)} rows, '
                    f'below {args.min_row_coverage:.0%}. Refusing to build a partial oracle teacher.'
                )
            out_path = tmp_dir / f'{split}_closed_oracle_eval.jsonl'
            if out_path.exists():
                out_path.unlink()
            run_evaluator(args, split, manifest_path, out_path, seq_name, keys_path)
            eval_rows = read_jsonl(out_path)
        normalized = [normalize_eval_row(row, manifest_by_key, split) for row in eval_rows]
        if len(normalized) < len(manifest_rows) * args.min_row_coverage:
            raise RuntimeError(
                f'{split}: output rows {len(normalized)}/{len(manifest_rows)} below {args.min_row_coverage:.0%}; '
                'this looks like a smoke or partial oracle file.'
            )
        ok_rows = sum(1 for row in normalized if row['status'] == 'ok')
        split_stats = {
            'rows': len(manifest_rows),
            'output_rows': len(normalized),
            'ok_rows': ok_rows,
            'skip_rows': len(normalized) - ok_rows,
            'word_rows': sum(len(row.get('word_scores') or []) for row in normalized),
            'phone_rows': sum(len(row.get('phone_scores') or []) for row in normalized),
        }
        stats[split] = split_stats
        print(json.dumps({'split': split, **split_stats}, ensure_ascii=False), flush=True)
        all_rows.extend(normalized)
    write_jsonl(args.output_jsonl, all_rows)
    summary_path = args.output_jsonl.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output_jsonl': str(args.output_jsonl), 'summary_json': str(summary_path), 'splits': stats}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
