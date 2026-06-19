import argparse
import json
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Create a tiny synthetic streaming_pcn_gopt_v1 dataset for smoke tests.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num-train', type=int, default=24)
    parser.add_argument('--num-val', type=int, default=8)
    parser.add_argument('--num-test', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=12)
    parser.add_argument('--phone-dim', type=int, default=43)
    parser.add_argument('--prosody-dim', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


def softmax_rows(rng, shape):
    arr = rng.normal(size=shape).astype(np.float32)
    arr = arr - arr.max(axis=-1, keepdims=True)
    exp_arr = np.exp(arr)
    return exp_arr / np.clip(exp_arr.sum(axis=-1, keepdims=True), 1e-8, None)


def build_split(rng, count, seq_len, phone_dim, prosody_dim):
    cn_post = softmax_rows(rng, (count, seq_len, phone_dim))
    acoustic_post = softmax_rows(rng, (count, seq_len, phone_dim))
    cn_entropy = -np.sum(cn_post * np.log(np.clip(cn_post, 1e-8, None)), axis=-1)
    acoustic_entropy = -np.sum(acoustic_post * np.log(np.clip(acoustic_post, 1e-8, None)), axis=-1)
    cn_top = np.sort(cn_post[:, :, :-1], axis=-1)[:, :, ::-1]
    acoustic_top = np.sort(acoustic_post[:, :, :-1], axis=-1)[:, :, ::-1]
    prefix_stability = rng.uniform(0.0, 1.0, size=(count, seq_len)).astype(np.float32)
    cn_stats = np.stack(
        [
            cn_post[:, :, -1],
            cn_entropy,
            cn_top[:, :, 0],
            cn_top[:, :, 0] - cn_top[:, :, 1],
            prefix_stability,
        ],
        axis=-1,
    ).astype(np.float32)
    acoustic_stats = np.stack(
        [
            acoustic_entropy,
            acoustic_top[:, :, 0] - acoustic_top[:, :, 1],
            rng.uniform(0.02, 0.12, size=(count, seq_len)),
            rng.uniform(0.0, 0.5, size=(count, seq_len)),
        ],
        axis=-1,
    ).astype(np.float32)
    prosody = rng.normal(size=(count, prosody_dim)).astype(np.float32)
    visible_len = rng.integers(low=max(3, seq_len // 3), high=seq_len + 1, size=(count,), dtype=np.int32)
    valid = np.arange(seq_len)[None, :] < visible_len[:, None]
    commit_mask = valid.astype(np.float32)
    pcn_word_id = np.tile(np.arange(seq_len, dtype=np.int32), (count, 1)) // 3
    pcn_word_id[~valid] = -1
    phone_target = np.zeros((count, seq_len, 2), dtype=np.float32) - 1.0
    word_target = np.zeros((count, seq_len, 4), dtype=np.float32) - 1.0
    phone_target[:, :, 0] = rng.integers(0, phone_dim - 1, size=(count, seq_len))
    phone_target[:, :, 1] = rng.uniform(1.0, 5.0, size=(count, seq_len))
    word_target[:, :, 0:3] = rng.uniform(1.0, 5.0, size=(count, seq_len, 3))
    word_target[:, :, 3] = pcn_word_id
    soft_label_weight = commit_mask * rng.uniform(0.1, 1.0, size=(count, seq_len)).astype(np.float32)
    asr_correct_target = (rng.uniform(size=(count, seq_len)) > 0.35).astype(np.float32) * valid
    uncertainty_target = rng.uniform(0.0, 1.0, size=(count, seq_len)).astype(np.float32) * valid
    is_final = np.zeros((count,), dtype=np.int8)
    is_final[::3] = 1
    utt_target = rng.uniform(1.0, 5.0, size=(count, 5)).astype(np.float32)
    coverage_ratio = rng.uniform(0.1, 1.0, size=(count,)).astype(np.float32)
    teacher_mask = (rng.uniform(size=(count,)) > 0.2).astype(np.float32)
    teacher_dim_mask = np.repeat(teacher_mask[:, None], 5, axis=1).astype(np.float32)
    teacher_prefix = np.clip(utt_target + rng.normal(scale=0.2, size=(count, 5)), 1.0, 5.0).astype(np.float32)
    teacher_final = np.clip(utt_target + rng.normal(scale=0.2, size=(count, 5)), 1.0, 5.0).astype(np.float32)
    teacher_word = np.clip(word_target[:, :, 0:3] + rng.normal(scale=0.2, size=(count, seq_len, 3)), 1.0, 5.0).astype(np.float32)
    teacher_word_mask = (soft_label_weight > 0).astype(np.float32)
    return {
        'cn_post': cn_post.astype(np.float32),
        'cn_stats': cn_stats,
        'acoustic_post': acoustic_post.astype(np.float32),
        'acoustic_stats': acoustic_stats,
        'prosody': prosody,
        'pcn_word_id': pcn_word_id,
        'phone_target': phone_target,
        'word_target': word_target,
        'utt_target': utt_target,
        'asr_correct_target': asr_correct_target.astype(np.float32),
        'uncertainty_target': uncertainty_target,
        'soft_label_weight': soft_label_weight,
        'commit_mask': commit_mask,
        'teacher_prefix_utt_score': teacher_prefix,
        'teacher_final_utt_score': teacher_final,
        'teacher_utt_mask': teacher_mask,
        'teacher_utt_dim_mask': teacher_dim_mask,
        'teacher_word_score': teacher_word,
        'teacher_word_mask': teacher_word_mask,
        'coverage_ratio': coverage_ratio,
        'visible_len': visible_len,
        'is_final': is_final,
    }


def save_split(output_dir, split, arrays):
    np.savez_compressed(output_dir / f'{split}_chunks.npz', **arrays)
    with open(output_dir / f'{split}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for idx in range(arrays['cn_post'].shape[0]):
            row = {
                'utt_id': f'{split}_utt_{idx // 4:04d}',
                'chunk_id': int(idx % 4),
                'coverage_ratio': float(arrays['coverage_ratio'][idx]),
                'visible_len': int(arrays['visible_len'][idx]),
                'is_final': bool(arrays['is_final'][idx]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    args = get_args()
    if args.output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counts = {'train': args.num_train, 'val': args.num_val, 'test': args.num_test}
    for split, count in counts.items():
        save_split(args.output_dir, split, build_split(rng, count, args.seq_len, args.phone_dim, args.prosody_dim))
    metadata = {
        'schema': 'streaming_pcn_gopt_v1',
        'seq_len': int(args.seq_len),
        'phone_dim': int(args.phone_dim),
        'epsilon_index': int(args.phone_dim - 1),
        'prosody': [f'p{i}' for i in range(args.prosody_dim)],
        'phn_dict': {f'P{i}': i for i in range(args.phone_dim - 1)},
        'synthetic': True,
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
