import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description='Inject MultiPA prefix/full-audio teacher scores into streaming_pcn_gopt_v2_stateful NPZ files.'
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--teacher-jsonl', type=Path, required=True, help='Output from MultiPA/eval_multipa_prefix.py.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Default: overwrite data-dir in place after backup.')
    parser.add_argument('--splits', type=str, default='train,val,test')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def teacher_state_dim(rows):
    for row in rows:
        state = row.get('teacher_state_embedding')
        if isinstance(state, list) and state:
            return len(state)
    return 0


def utt_score(row):
    scores = row.get('scores') or {}
    # MultiPA does not expose completeness. Keep that dimension masked out.
    values = np.array(
        [
            float(scores.get('accuracy', 0.0)),
            0.0,
            float(scores.get('fluency', 0.0)),
            float(scores.get('prosodic', 0.0)),
            float(scores.get('total', 0.0)),
        ],
        dtype=np.float32,
    )
    dim_mask = np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return values, dim_mask


def teacher_word_table(row):
    rows = row.get('word_scores') or []
    top_level_times = row.get('teacher_word_time') or []
    scores = []
    dim_masks = []
    times = []
    for idx, item in enumerate(rows):
        cur_scores = []
        cur_mask = []
        for key in ['pred_accuracy', 'pred_stress', 'pred_total']:
            value = item.get(key)
            try:
                value = float(value)
                valid = bool(np.isfinite(value))
            except (TypeError, ValueError):
                value = 0.0
                valid = False
            cur_scores.append(value if valid else 0.0)
            cur_mask.append(1.0 if valid else 0.0)
        scores.append(cur_scores)
        dim_masks.append(cur_mask)
        if 'start' in item and 'end' in item:
            times.append((float(item['start']), float(item['end'])))
        elif idx < len(top_level_times) and top_level_times[idx] is not None:
            times.append((float(top_level_times[idx][0]), float(top_level_times[idx][1])))
        else:
            times.append(None)
    return np.asarray(scores, dtype=np.float32), times, np.asarray(dim_masks, dtype=np.float32)


def overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def align_word_scores(manifest_row, pcn_word_ids, teacher_row, seq_len):
    out = np.zeros((seq_len, 3), dtype=np.float32)
    mask = np.zeros((seq_len,), dtype=np.float32)
    dim_mask = np.zeros((seq_len, 3), dtype=np.float32)
    teacher_scores, teacher_times, teacher_dim_masks = teacher_word_table(teacher_row)
    if teacher_scores.size == 0:
        return out, mask, dim_mask

    top_word_times = []
    word_timestamp_groups = manifest_row.get('word_timestamps') or []
    if word_timestamp_groups:
        top_word_times = word_timestamp_groups[0] or []

    has_teacher_times = any(item is not None for item in teacher_times)
    has_student_times = bool(top_word_times)
    for slot_idx, word_idx in enumerate(pcn_word_ids.tolist()):
        if slot_idx >= seq_len or word_idx < 0:
            continue
        chosen = None
        if has_teacher_times and has_student_times and word_idx < len(top_word_times):
            student_time = top_word_times[word_idx]
            s0 = float(student_time.get('start', 0.0))
            s1 = float(student_time.get('end', s0))
            best_overlap = 0.0
            best_idx = None
            for teacher_idx, teacher_time in enumerate(teacher_times):
                if teacher_time is None:
                    continue
                cur_overlap = overlap_ratio(s0, s1, teacher_time[0], teacher_time[1])
                if cur_overlap > best_overlap:
                    best_overlap = cur_overlap
                    best_idx = teacher_idx
            if best_idx is not None and best_overlap >= 0.2:
                chosen = best_idx
        if chosen is None and word_idx < teacher_scores.shape[0]:
            chosen = int(word_idx)
        if chosen is None:
            continue
        out[slot_idx] = teacher_scores[chosen]
        dim_mask[slot_idx] = teacher_dim_masks[chosen]
        mask[slot_idx] = float(np.any(teacher_dim_masks[chosen] > 0))
    return out, mask, dim_mask


def load_manifest(data_dir, split):
    return load_jsonl(data_dir / f'{split}_manifest.jsonl')


def copy_static_files(src_dir, dst_dir, splits):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ['metadata.json']:
        shutil.copy2(src_dir / name, dst_dir / name)
    for split in splits:
        shutil.copy2(src_dir / f'{split}_manifest.jsonl', dst_dir / f'{split}_manifest.jsonl')


def main():
    args = get_args()
    splits = [item.strip() for item in args.splits.split(',') if item.strip()]
    teacher_rows = load_jsonl(args.teacher_jsonl)
    state_dim = teacher_state_dim(teacher_rows)
    teacher_by_key = {
        (str(row.get('utt_id')), int(row.get('chunk_id', -1))): row
        for row in teacher_rows
        if row.get('status', 'ok') == 'ok'
    }
    output_dir = args.output_dir or args.data_dir
    if output_dir != args.data_dir:
        if output_dir.exists() and args.overwrite:
            shutil.rmtree(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'{output_dir} exists; pass --overwrite or choose another output dir.')
        copy_static_files(args.data_dir, output_dir, splits)

    for split in splits:
        archive_path = args.data_dir / f'{split}_chunks.npz'
        manifest = load_manifest(args.data_dir, split)
        archive = np.load(archive_path)
        arrays = {name: archive[name] for name in archive.files}
        count, seq_len = arrays['cn_post'].shape[:2]
        teacher_prefix = np.zeros((count, 5), dtype=np.float32)
        teacher_final = np.zeros((count, 5), dtype=np.float32)
        teacher_dim_mask = np.zeros((count, 5), dtype=np.float32)
        teacher_mask = np.zeros((count,), dtype=np.float32)
        teacher_word_score = np.zeros((count, seq_len, 3), dtype=np.float32)
        teacher_word_mask = np.zeros((count, seq_len), dtype=np.float32)
        teacher_word_dim_mask = np.zeros((count, seq_len, 3), dtype=np.float32)
        cur_state_dim = state_dim
        if cur_state_dim <= 0 and 'teacher_state_embedding' in arrays:
            cur_state_dim = int(arrays['teacher_state_embedding'].shape[-1])
        teacher_state_embedding = (
            np.zeros((count, cur_state_dim), dtype=np.float32)
            if cur_state_dim > 0
            else None
        )
        teacher_state_mask = np.zeros((count,), dtype=np.float32)

        final_by_utt = {}
        for row in manifest:
            key = (str(row.get('utt_id')), int(row.get('chunk_id', -1)))
            if key not in teacher_by_key:
                continue
            utt_id = key[0]
            current = final_by_utt.get(utt_id)
            if current is None or bool(row.get('is_final')) or int(row.get('chunk_id', -1)) > current[0]:
                final_by_utt[utt_id] = (int(row.get('chunk_id', -1)), teacher_by_key[key])

        for idx, row in enumerate(manifest):
            key = (str(row.get('utt_id')), int(row.get('chunk_id', -1)))
            teacher_row = teacher_by_key.get(key)
            if teacher_row is None:
                continue
            prefix_score, dim_mask = utt_score(teacher_row)
            final_row = final_by_utt.get(key[0], (None, teacher_row))[1]
            final_score, _ = utt_score(final_row)
            teacher_prefix[idx] = prefix_score
            teacher_final[idx] = final_score
            teacher_dim_mask[idx] = dim_mask
            teacher_mask[idx] = 1.0
            word_score, word_mask, word_dim_mask = align_word_scores(
                row,
                arrays['pcn_word_id'][idx],
                teacher_row,
                seq_len,
            )
            teacher_word_score[idx] = word_score
            teacher_word_mask[idx] = word_mask
            teacher_word_dim_mask[idx] = word_dim_mask
            if teacher_state_embedding is not None:
                state = teacher_row.get('teacher_state_embedding')
                if isinstance(state, list) and len(state) == teacher_state_embedding.shape[-1]:
                    state_array = np.asarray(state, dtype=np.float32)
                    if np.isfinite(state_array).all():
                        teacher_state_embedding[idx] = state_array
                        teacher_state_mask[idx] = 1.0

        arrays['teacher_prefix_utt_score'] = teacher_prefix
        arrays['teacher_final_utt_score'] = teacher_final
        arrays['teacher_utt_mask'] = teacher_mask
        arrays['teacher_utt_dim_mask'] = teacher_dim_mask
        arrays['teacher_word_score'] = teacher_word_score
        arrays['teacher_word_mask'] = teacher_word_mask
        arrays['teacher_word_dim_mask'] = teacher_word_dim_mask
        if teacher_state_embedding is not None:
            arrays['teacher_state_embedding'] = teacher_state_embedding
            arrays['teacher_state_mask'] = teacher_state_mask
        else:
            arrays.pop('teacher_state_embedding', None)
            arrays.pop('teacher_state_mask', None)
        dst_path = output_dir / f'{split}_chunks.npz'
        if dst_path.exists() and output_dir == args.data_dir:
            shutil.copy2(dst_path, dst_path.with_suffix('.npz.bak'))
        np.savez_compressed(dst_path, **arrays)
        print(
            json.dumps(
                {
                    'split': split,
                    'rows': int(count),
                    'teacher_rows_used': int(teacher_mask.sum()),
                    'teacher_word_slots': int(teacher_word_mask.sum()),
                    'teacher_word_dimensions': int(teacher_word_dim_mask.sum()),
                    'teacher_state_rows': int(teacher_state_mask.sum()),
                    'teacher_state_dim': int(teacher_state_embedding.shape[-1]) if teacher_state_embedding is not None else 0,
                    'output': str(dst_path),
                },
                ensure_ascii=False,
            )
        )


if __name__ == '__main__':
    main()
