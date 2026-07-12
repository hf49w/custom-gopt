import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_EXP_ROOT = Path('/DATA_2/guest/custom-gopt/exp/pcn_extra_20260704_2130')
FEATURE_NAMES = [
    'slot_duration',
    'slot_log_energy_mean',
    'slot_log_energy_std',
    'slot_log_energy_max',
    'slot_f0_mean',
    'slot_f0_std',
    'slot_f0_max',
    'slot_voiced_ratio',
    'slot_position_in_word',
    'word_phone_count',
    'energy_relative_to_word_mean',
    'duration_relative_to_word_mean',
    'f0_relative_to_word_mean',
    'is_vowel',
    'lexical_stress_0',
    'lexical_stress_1',
    'lexical_stress_2',
]
VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'AX'}
VOICED = VOWELS | {'B', 'D', 'DH', 'G', 'JH', 'L', 'M', 'N', 'NG', 'R', 'V', 'W', 'Y', 'Z', 'ZH'}


def get_args():
    parser = argparse.ArgumentParser(description='Add slot-level prosody/stress-mask fields to PCN v2 NPZ data.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_EXP_ROOT / 'data_streaming_pcn_oracle_gopt_full')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_EXP_ROOT / 'data_streaming_pcn_oracle_gopt_full_slotprosody')
    parser.add_argument('--splits', type=str, default='train,val,test')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, dict):
        start = value.get('start', value.get('phone_start', value.get('word_start')))
        end = value.get('end', value.get('phone_end', value.get('word_end')))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = value[0], value[1]
    else:
        return None
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(start) or not np.isfinite(end):
        return None
    return start, max(end, start)


def first_word_times(row):
    raw = row.get('word_timestamps') or []
    if raw and isinstance(raw[0], list):
        for candidate in raw:
            if candidate:
                return [parse_time(item) for item in candidate]
        return []
    return [parse_time(item) for item in raw]


def times_from_phone_rows(row):
    rows = row.get('hyp_phone_rows') or row.get('phone_rows') or []
    if rows and isinstance(rows[0], list):
        rows = rows[0]
    return [parse_time(item) for item in rows]


def infer_slot_times(row, pcn_word_id, visible_len):
    for key in ['slot_times', 'pcn_slot_times']:
        raw_times = row.get(key) or []
        parsed = [parse_time(item) for item in raw_times]
        if any(item is not None for item in parsed):
            return parsed[:visible_len]
    parsed = times_from_phone_rows(row)
    if any(item is not None for item in parsed):
        return parsed[:visible_len]
    word_times = first_word_times(row)
    if not word_times:
        return []
    out = [None] * visible_len
    word_ids = np.asarray(pcn_word_id[:visible_len], dtype=np.int32)
    for word_idx in sorted(set(int(x) for x in word_ids.tolist() if int(x) >= 0)):
        if word_idx >= len(word_times) or word_times[word_idx] is None:
            continue
        positions = np.flatnonzero(word_ids == word_idx)
        if positions.size <= 0:
            continue
        start, end = word_times[word_idx]
        step = (end - start) / max(int(positions.size), 1)
        for offset, slot_idx in enumerate(positions.tolist()):
            out[slot_idx] = (start + offset * step, start + (offset + 1) * step)
    return out


def clean_phone(phone):
    return ''.join(ch for ch in str(phone).upper() if not ch.isdigit())


def phone_flags(phone):
    phone = clean_phone(phone)
    is_vowel = 1.0 if phone in VOWELS else 0.0
    voiced = 1.0 if phone in VOICED else 0.0
    return is_vowel, voiced


def build_id_to_phone(metadata):
    phn_dict = metadata.get('phn_dict') or {}
    return {int(idx): str(phone) for phone, idx in phn_dict.items()}


def top_phone_ids(cn_post, epsilon_index):
    limit = int(epsilon_index) if epsilon_index is not None and int(epsilon_index) > 0 else max(cn_post.shape[-1] - 1, 1)
    return np.argmax(cn_post[:, :, :limit], axis=-1).astype(np.int32)


def build_features(arrays, manifest, metadata):
    cn_post = arrays['cn_post']
    pcn_word_id = arrays['pcn_word_id']
    visible_len = arrays['visible_len'].astype(np.int32)
    count, seq_len = pcn_word_id.shape
    slot_prosody = np.zeros((count, seq_len, len(FEATURE_NAMES)), dtype=np.float32)
    slot_is_vowel = np.zeros((count, seq_len), dtype=np.float32)
    slot_voiced_ratio = np.zeros((count, seq_len), dtype=np.float32)
    id_to_phone = build_id_to_phone(metadata)
    epsilon_index = metadata.get('epsilon_index', cn_post.shape[-1] - 1)
    top_ids = top_phone_ids(cn_post, epsilon_index)

    for row_idx, row in enumerate(manifest):
        cur_visible = min(int(visible_len[row_idx]), seq_len)
        cur_word_ids = pcn_word_id[row_idx, :cur_visible]
        cur_times = infer_slot_times(row, cur_word_ids, cur_visible)
        durations = np.zeros((cur_visible,), dtype=np.float32)
        has_time = np.zeros((cur_visible,), dtype=bool)
        for slot_idx in range(cur_visible):
            phone = id_to_phone.get(int(top_ids[row_idx, slot_idx]), '')
            is_vowel, voiced = phone_flags(phone)
            slot_is_vowel[row_idx, slot_idx] = is_vowel
            slot_voiced_ratio[row_idx, slot_idx] = voiced
            cur_time = cur_times[slot_idx] if slot_idx < len(cur_times) else None
            if cur_time is None:
                continue
            start, end = cur_time
            duration = max(end - start, 0.0)
            if duration <= 0.0:
                continue
            has_time[slot_idx] = True
            durations[slot_idx] = duration
            slot_prosody[row_idx, slot_idx, 0] = duration
            slot_prosody[row_idx, slot_idx, 7] = voiced
            slot_prosody[row_idx, slot_idx, 13] = is_vowel
        for word_idx in sorted(set(int(x) for x in cur_word_ids.tolist() if int(x) >= 0)):
            positions = np.flatnonzero(cur_word_ids == word_idx)
            timed_positions = [int(pos) for pos in positions.tolist() if has_time[int(pos)]]
            if not timed_positions:
                continue
            mean_duration = float(np.mean(durations[timed_positions]))
            for pos_offset, slot_idx in enumerate(positions.tolist()):
                if not has_time[int(slot_idx)]:
                    continue
                slot_prosody[row_idx, slot_idx, 8] = float(pos_offset / max(len(positions) - 1, 1))
                slot_prosody[row_idx, slot_idx, 9] = float(len(positions))
                slot_prosody[row_idx, slot_idx, 11] = float(durations[slot_idx] / max(mean_duration, 1e-6))
    return slot_prosody, slot_is_vowel, slot_voiced_ratio


def save_npz_atomic(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp, **arrays)
        with np.load(tmp_path) as archive:
            for required in ['slot_prosody', 'slot_is_vowel', 'slot_voiced_ratio']:
                if required not in archive.files:
                    raise RuntimeError(f'{path}: missing {required} after save')
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f'{output_dir} exists; pass --overwrite to rebuild.')
    if output_dir.exists() and overwrite:
        build_dir = output_dir.parent / f'.{output_dir.name}.tmp.{os.getpid()}'
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True)
        return build_dir, output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, None


def commit_output_dir(build_dir, final_dir):
    if final_dir is None:
        return
    backup = final_dir.parent / f'.{final_dir.name}.bak.{os.getpid()}'
    shutil.rmtree(backup, ignore_errors=True)
    try:
        if final_dir.exists():
            final_dir.rename(backup)
        build_dir.rename(final_dir)
    except Exception:
        if not final_dir.exists() and backup.exists():
            backup.rename(final_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def main():
    args = get_args()
    splits = [item.strip() for item in args.splits.split(',') if item.strip()]
    if args.data_dir.resolve() == args.output_dir.resolve():
        raise ValueError('--output-dir must differ from --data-dir.')
    metadata = json.loads((args.data_dir / 'metadata.json').read_text(encoding='utf-8'))
    build_dir, final_dir = prepare_output_dir(args.output_dir, args.overwrite)
    shutil.copy2(args.data_dir / 'metadata.json', build_dir / 'metadata.json')
    summary = {}
    for split in splits:
        manifest = load_jsonl(args.data_dir / f'{split}_manifest.jsonl')
        shutil.copy2(args.data_dir / f'{split}_manifest.jsonl', build_dir / f'{split}_manifest.jsonl')
        with np.load(args.data_dir / f'{split}_chunks.npz') as archive:
            arrays = {name: archive[name] for name in archive.files}
        slot_prosody, slot_is_vowel, slot_voiced_ratio = build_features(arrays, manifest, metadata)
        arrays['slot_prosody'] = slot_prosody
        arrays['slot_is_vowel'] = slot_is_vowel
        arrays['slot_voiced_ratio'] = slot_voiced_ratio
        save_npz_atomic(build_dir / f'{split}_chunks.npz', arrays)
        valid = arrays['visible_len'].astype(np.int32)
        total_slots = int(sum(min(int(item), slot_prosody.shape[1]) for item in valid))
        nonzero = int(np.any(slot_prosody != 0.0, axis=-1).sum())
        summary[split] = {
            'slot_prosody_nonzero_ratio': float(nonzero / max(total_slots, 1)),
            'vowel_slot_count': int(slot_is_vowel.sum()),
            'voiced_slot_count': int((slot_voiced_ratio > 0.3).sum()),
        }
        print(json.dumps({'split': split, **summary[split]}, ensure_ascii=False))

    metadata['slot_prosody'] = FEATURE_NAMES
    metadata['slot_is_vowel'] = '1 for slots whose PCN top phone is ARPABET vowel-like; inferred without ASR regeneration.'
    metadata['slot_voiced_ratio'] = 'Approximate 0/1 voiced flag from PCN top phone class; no Charsiu/Whisper rerun.'
    metadata['slot_prosody_note'] = (
        'Added by augment_slot_prosody_pcn.py from existing manifest/NPZ fields. '
        'Energy, F0, and lexical_stress_* are zero fallback because raw frame-level features are not recomputed.'
    )
    targets = list(metadata.get('targets', []))
    for key in ['slot_prosody', 'slot_is_vowel', 'slot_voiced_ratio']:
        if key not in targets:
            targets.append(key)
    metadata['targets'] = targets
    metadata['slot_prosody_augment_summary'] = summary
    (build_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    commit_output_dir(build_dir, final_dir)


if __name__ == '__main__':
    main()
