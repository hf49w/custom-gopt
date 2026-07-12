import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


UTT_KEYS = [
    ('accuracy', ['utt_accuracy', 'accuracy']),
    ('completeness', ['utt_completeness', 'completeness']),
    ('fluency', ['utt_fluency', 'fluency']),
    ('prosody', ['utt_prosody', 'utt_prosodic', 'prosody', 'prosodic']),
    ('total', ['utt_total', 'total']),
]
WORD_KEYS = [
    ('accuracy', ['word_accuracy', 'pred_accuracy', 'accuracy']),
    ('stress', ['word_stress', 'pred_stress', 'stress']),
    ('total', ['word_total', 'pred_total', 'total']),
]


def get_args():
    parser = argparse.ArgumentParser(description='Inject closed-oracle GOPT teacher fields into PCN v2 stateful data.')
    parser.add_argument('--data-dir', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--oracle-jsonl', type=Path)
    source.add_argument('--oracle-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits', type=str, default='train,val,test')
    parser.add_argument('--drop-completeness', dest='drop_completeness', action='store_true', default=True)
    parser.add_argument('--keep-completeness', dest='drop_completeness', action='store_false')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not np.isfinite(out):
        return 0.0, 0.0
    return out, 1.0


def score_vector(mapping, key_groups):
    values = []
    mask = []
    for _, candidates in key_groups:
        value = None
        for key in candidates:
            if key in mapping and mapping[key] not in {None, ''}:
                value = mapping[key]
                break
        cur_value, cur_mask = finite_float(value)
        values.append(cur_value)
        mask.append(cur_mask)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def item_time(item, prefix=''):
    start_keys = [f'{prefix}_start', 'start']
    end_keys = [f'{prefix}_end', 'end']
    start = next((item.get(key) for key in start_keys if key in item and item.get(key) not in {None, ''}), None)
    end = next((item.get(key) for key in end_keys if key in item and item.get(key) not in {None, ''}), None)
    start, start_ok = finite_float(start)
    end, end_ok = finite_float(end)
    if start_ok and end_ok:
        return (float(start), float(end))
    return None


def empty_record():
    return {
        'utt_score': np.zeros((5,), dtype=np.float32),
        'utt_dim_mask': np.zeros((5,), dtype=np.float32),
        'prefix_available': True,
        'word_items': [],
        'phone_items': [],
    }


def record_keys(row):
    base = (str(row.get('utt_id')), int(row.get('chunk_id', -1)))
    split = row.get('split')
    if split:
        return [(str(split),) + base, base]
    return [base]


def load_jsonl_records(path):
    records = {}
    with open(path, 'r', encoding='utf-8-sig') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record = empty_record()
            ok = row.get('status', 'ok') == 'ok'
            scores = row.get('scores') or row
            if ok:
                record['utt_score'], record['utt_dim_mask'] = score_vector(scores, UTT_KEYS)
            record['prefix_available'] = bool(row.get('prefix_available', ok))

            word_times = row.get('word_times') or row.get('teacher_word_time') or []
            for idx, item in enumerate((row.get('word_scores') or []) if ok else []):
                if not isinstance(item, dict):
                    continue
                score, mask = score_vector(item, WORD_KEYS)
                time_value = item_time(item, 'word')
                if time_value is None and idx < len(word_times) and word_times[idx] is not None:
                    raw_time = word_times[idx]
                    if isinstance(raw_time, dict):
                        time_value = item_time(raw_time)
                    elif len(raw_time) >= 2:
                        time_value = (float(raw_time[0]), float(raw_time[1]))
                record['word_items'].append({'index': int(item.get('word_index', idx)), 'score': score, 'mask': mask, 'time': time_value})

            phone_times = row.get('phone_times') or []
            for idx, item in enumerate((row.get('phone_scores') or []) if ok else []):
                if isinstance(item, dict):
                    value, valid = finite_float(item.get('phone_score', item.get('score', item.get('pred_accuracy', item.get('total')))))
                    time_value = item_time(item, 'phone')
                else:
                    value, valid = finite_float(item)
                    time_value = None
                if time_value is None and idx < len(phone_times) and phone_times[idx] is not None:
                    raw_time = phone_times[idx]
                    if isinstance(raw_time, dict):
                        time_value = item_time(raw_time)
                    elif len(raw_time) >= 2:
                        time_value = (float(raw_time[0]), float(raw_time[1]))
                record['phone_items'].append({'index': idx, 'score': float(value), 'mask': float(valid), 'time': time_value})
            for key in record_keys(row):
                records[key] = record
    return records


def load_csv_records(path):
    records = defaultdict(empty_record)
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            base_key = (str(row.get('utt_id')), int(float(row.get('chunk_id', -1) or -1)))
            key = (str(row.get('split')),) + base_key if row.get('split') else base_key
            record = records[key]
            utt_score, utt_mask = score_vector(row, UTT_KEYS)
            if utt_mask.sum() > record['utt_dim_mask'].sum():
                record['utt_score'] = utt_score
                record['utt_dim_mask'] = utt_mask
            word_score, word_mask = score_vector(row, WORD_KEYS)
            if word_mask.sum() > 0:
                word_index, ok = finite_float(row.get('word_index', len(record['word_items'])))
                time_value = item_time(row, 'word')
                record['word_items'].append(
                    {'index': int(word_index) if ok else len(record['word_items']), 'score': word_score, 'mask': word_mask, 'time': time_value}
                )
            phone_value, phone_valid = finite_float(row.get('phone_score', row.get('phone_total', row.get('phone'))))
            if phone_valid:
                phone_index, ok = finite_float(row.get('phone_index', len(record['phone_items'])))
                time_value = item_time(row, 'phone')
                record['phone_items'].append(
                    {'index': int(phone_index) if ok else len(record['phone_items']), 'score': phone_value, 'mask': phone_valid, 'time': time_value}
                )
    return dict(records)


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def top_word_times(manifest_row):
    groups = manifest_row.get('word_timestamps') or []
    return groups[0] if groups else []


def align_word_scores(manifest_row, pcn_word_ids, record, seq_len):
    out = np.zeros((seq_len, 3), dtype=np.float32)
    dim_mask = np.zeros((seq_len, 3), dtype=np.float32)
    items = record.get('word_items') or []
    if not items:
        return out, dim_mask
    student_times = top_word_times(manifest_row)
    has_oracle_times = any(item.get('time') is not None for item in items)
    has_student_times = bool(student_times)
    by_index = {int(item.get('index', idx)): item for idx, item in enumerate(items)}
    for slot_idx, word_idx in enumerate(pcn_word_ids.tolist()):
        if slot_idx >= seq_len or word_idx < 0:
            continue
        chosen = None
        if has_oracle_times and has_student_times and word_idx < len(student_times):
            student_time = student_times[word_idx]
            s0 = float(student_time.get('start', 0.0))
            s1 = float(student_time.get('end', s0))
            best_overlap = 0.0
            for item in items:
                cur_time = item.get('time')
                if cur_time is None:
                    continue
                cur_overlap = overlap_ratio(s0, s1, cur_time[0], cur_time[1])
                if cur_overlap > best_overlap:
                    best_overlap = cur_overlap
                    chosen = item
            if best_overlap < 0.2:
                chosen = None
        if chosen is None:
            chosen = by_index.get(int(word_idx))
        if chosen is None:
            continue
        out[slot_idx] = chosen['score']
        dim_mask[slot_idx] = chosen['mask']
    return out, dim_mask


def align_phone_scores(manifest_row, record, seq_len):
    out = np.zeros((seq_len,), dtype=np.float32)
    mask = np.zeros((seq_len,), dtype=np.float32)
    items = record.get('phone_items') or []
    if not items:
        return out, mask
    slot_times = manifest_row.get('slot_times') or []
    has_oracle_times = any(item.get('time') is not None for item in items)
    has_slot_times = bool(slot_times)
    by_index = {int(item.get('index', idx)): item for idx, item in enumerate(items)}
    for slot_idx in range(seq_len):
        chosen = None
        if has_oracle_times and has_slot_times and slot_idx < len(slot_times) and slot_times[slot_idx] is not None:
            slot_time = slot_times[slot_idx]
            s0, s1 = float(slot_time[0]), float(slot_time[1])
            best_overlap = 0.0
            for item in items:
                cur_time = item.get('time')
                if cur_time is None:
                    continue
                cur_overlap = overlap_ratio(s0, s1, cur_time[0], cur_time[1])
                if cur_overlap > best_overlap:
                    best_overlap = cur_overlap
                    chosen = item
            if best_overlap < 0.2:
                chosen = None
        if chosen is None:
            chosen = by_index.get(slot_idx)
        if chosen is None:
            continue
        out[slot_idx] = float(chosen['score'])
        mask[slot_idx] = float(chosen['mask'])
    return out, mask


def lookup_record(records, split, row):
    base = (str(row.get('utt_id')), int(row.get('chunk_id', -1)))
    return records.get((split,) + base) or records.get(base)


def copy_static_files(src_dir, dst_dir, splits):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / 'metadata.json', dst_dir / 'metadata.json')
    for split in splits:
        shutil.copy2(src_dir / f'{split}_manifest.jsonl', dst_dir / f'{split}_manifest.jsonl')


def save_npz_atomic(dst_path, arrays):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f'.{dst_path.name}.',
            suffix='.tmp',
            dir=dst_path.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp, **arrays)
        with np.load(tmp_path) as archive:
            missing = sorted(set(arrays) - set(archive.files))
            if missing:
                raise RuntimeError(f'atomic save validation failed for {dst_path}: missing arrays {missing}')
        os.replace(tmp_path, dst_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def resolve_output_dir(output_dir, overwrite):
    if overwrite and output_dir.exists():
        tmp_dir = output_dir.parent / f'.{output_dir.name}.tmp.{os.getpid()}'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return tmp_dir, output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'{output_dir} exists; pass --overwrite or choose another output dir.')
    return output_dir, None


def commit_output_dir(build_dir, final_dir):
    if final_dir is None:
        return
    backup_dir = final_dir.parent / f'.{final_dir.name}.bak.{os.getpid()}'
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    try:
        if final_dir.exists():
            final_dir.rename(backup_dir)
        build_dir.rename(final_dir)
    except Exception:
        if not final_dir.exists() and backup_dir.exists():
            backup_dir.rename(final_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def main():
    args = get_args()
    splits = [item.strip() for item in args.splits.split(',') if item.strip()]
    output_dir, final_output_dir = resolve_output_dir(args.output_dir, args.overwrite)
    try:
        copy_static_files(args.data_dir, output_dir, splits)
    except Exception:
        if final_output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)
        raise
    records = load_jsonl_records(args.oracle_jsonl) if args.oracle_jsonl else load_csv_records(args.oracle_csv)

    for split in splits:
        archive_path = args.data_dir / f'{split}_chunks.npz'
        manifest = load_jsonl(args.data_dir / f'{split}_manifest.jsonl')
        archive = np.load(archive_path)
        arrays = {name: archive[name] for name in archive.files}
        count, seq_len = arrays['cn_post'].shape[:2]

        oracle_prefix = np.zeros((count, 5), dtype=np.float32)
        oracle_final = np.zeros((count, 5), dtype=np.float32)
        oracle_utt_dim_mask = np.zeros((count, 5), dtype=np.float32)
        oracle_prefix_utt_dim_mask = np.zeros((count, 5), dtype=np.float32)
        oracle_final_utt_dim_mask = np.zeros((count, 5), dtype=np.float32)
        oracle_utt_mask = np.zeros((count,), dtype=np.float32)
        oracle_word_score = np.zeros((count, seq_len, 3), dtype=np.float32)
        oracle_word_dim_mask = np.zeros((count, seq_len, 3), dtype=np.float32)
        oracle_phone_score = np.zeros((count, seq_len), dtype=np.float32)
        oracle_phone_mask = np.zeros((count, seq_len), dtype=np.float32)

        final_by_utt = {}
        for row in manifest:
            record = lookup_record(records, split, row)
            if record is None:
                continue
            utt_id = str(row.get('utt_id'))
            current = final_by_utt.get(utt_id)
            if current is None or bool(row.get('is_final')) or int(row.get('chunk_id', -1)) > current[0]:
                final_by_utt[utt_id] = (int(row.get('chunk_id', -1)), record)

        for idx, row in enumerate(manifest):
            record = lookup_record(records, split, row)
            if record is None:
                continue
            final_record = final_by_utt.get(str(row.get('utt_id')), (None, record))[1]
            dim_mask = record['utt_dim_mask'].copy()
            final_dim_mask = final_record['utt_dim_mask'].copy()
            if args.drop_completeness:
                dim_mask[1] = 0.0
                final_dim_mask[1] = 0.0
            oracle_prefix[idx] = record['utt_score']
            oracle_final[idx] = final_record['utt_score']
            oracle_utt_dim_mask[idx] = dim_mask
            oracle_prefix_utt_dim_mask[idx] = dim_mask if record.get('prefix_available', True) else 0.0
            oracle_final_utt_dim_mask[idx] = final_dim_mask
            oracle_utt_mask[idx] = float((dim_mask.sum() + final_dim_mask.sum()) > 0)
            word_score, word_dim_mask = align_word_scores(row, arrays['pcn_word_id'][idx], record, seq_len)
            phone_score, phone_mask = align_phone_scores(row, record, seq_len)
            oracle_word_score[idx] = word_score
            oracle_word_dim_mask[idx] = word_dim_mask
            oracle_phone_score[idx] = phone_score
            oracle_phone_mask[idx] = phone_mask

        arrays['oracle_phone_score'] = oracle_phone_score
        arrays['oracle_phone_mask'] = oracle_phone_mask
        arrays['oracle_word_score'] = oracle_word_score
        arrays['oracle_word_dim_mask'] = oracle_word_dim_mask
        arrays['oracle_prefix_utt_score'] = oracle_prefix
        arrays['oracle_final_utt_score'] = oracle_final
        arrays['oracle_utt_dim_mask'] = oracle_utt_dim_mask
        arrays['oracle_prefix_utt_dim_mask'] = oracle_prefix_utt_dim_mask
        arrays['oracle_final_utt_dim_mask'] = oracle_final_utt_dim_mask
        arrays['oracle_utt_mask'] = oracle_utt_mask
        dst_path = output_dir / f'{split}_chunks.npz'
        reported_path = (final_output_dir or output_dir) / f'{split}_chunks.npz'
        save_npz_atomic(dst_path, arrays)
        print(
            json.dumps(
                {
                    'split': split,
                    'rows': int(count),
                    'oracle_utt_rows': int(oracle_utt_mask.sum()),
                    'oracle_word_slots': int(np.any(oracle_word_dim_mask > 0, axis=-1).sum()),
                    'oracle_phone_slots': int(oracle_phone_mask.sum()),
                    'oracle_completeness_mask_sum': float(
                        oracle_utt_dim_mask[:, 1].sum()
                        + oracle_prefix_utt_dim_mask[:, 1].sum()
                        + oracle_final_utt_dim_mask[:, 1].sum()
                    ),
                    'output': str(reported_path),
                },
                ensure_ascii=False,
            )
        )
    commit_output_dir(output_dir, final_output_dir)


if __name__ == '__main__':
    main()
