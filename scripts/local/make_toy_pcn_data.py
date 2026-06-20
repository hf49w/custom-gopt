import argparse
import json
import shutil
from pathlib import Path

import numpy as np


SCHEMA = 'streaming_pcn_gopt_v2_stateful'


def get_args():
    parser = argparse.ArgumentParser(description='Create a tiny synthetic streaming_pcn_gopt_v2_stateful dataset.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num-train', type=int, default=8)
    parser.add_argument('--num-val', type=int, default=3)
    parser.add_argument('--num-test', type=int, default=3)
    parser.add_argument('--chunks-per-utt', type=int, default=4)
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


def build_split(rng, utt_count, chunks_per_utt, seq_len, phone_dim, prosody_dim):
    count = utt_count * chunks_per_utt
    cn_post = softmax_rows(rng, (count, seq_len, phone_dim))
    acoustic_post = softmax_rows(rng, (count, seq_len, phone_dim))
    cn_entropy = -np.sum(cn_post * np.log(np.clip(cn_post, 1e-8, None)), axis=-1)
    acoustic_entropy = -np.sum(acoustic_post * np.log(np.clip(acoustic_post, 1e-8, None)), axis=-1)
    cn_top = np.sort(cn_post[:, :, :-1], axis=-1)[:, :, ::-1]
    acoustic_top = np.sort(acoustic_post[:, :, :-1], axis=-1)[:, :, ::-1]
    cn_stats = np.stack(
        [
            cn_post[:, :, -1],
            cn_entropy,
            cn_top[:, :, 0],
            cn_top[:, :, 0] - cn_top[:, :, 1],
            np.zeros((count, seq_len), dtype=np.float32),
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
    visible_len = np.full((count,), seq_len, dtype=np.int32)
    pcn_word_id = np.tile(np.arange(seq_len, dtype=np.int32), (count, 1)) // 3
    phone_target = np.zeros((count, seq_len, 2), dtype=np.float32) - 1.0
    word_target = np.zeros((count, seq_len, 4), dtype=np.float32) - 1.0
    phone_target[:, :, 0] = rng.integers(0, phone_dim - 1, size=(count, seq_len))
    phone_target[:, :, 1] = rng.uniform(1.0, 5.0, size=(count, seq_len))
    word_target[:, :, 0:3] = rng.uniform(1.0, 5.0, size=(count, seq_len, 3))
    word_target[:, :, 3] = pcn_word_id

    cumulative_commit_mask = np.zeros((count, seq_len), dtype=np.float32)
    new_commit_mask = np.zeros((count, seq_len), dtype=np.float32)
    mapped_old_slot = np.zeros((count, seq_len), dtype=np.int32) - 1
    previous_chunk_id = np.zeros((count,), dtype=np.int32) - 1
    utterance_index = np.zeros((count,), dtype=np.int32)
    state_reset = np.zeros((count,), dtype=np.int8)
    chunk_id = np.zeros((count,), dtype=np.int32)
    is_final = np.zeros((count,), dtype=np.int8)
    prefix_stability = np.zeros((count,), dtype=np.float32)
    new_committed_word_count = np.zeros((count,), dtype=np.int32)
    cumulative_committed_word_count = np.zeros((count,), dtype=np.int32)

    slots_per_chunk = max(1, seq_len // chunks_per_utt)
    for utt in range(utt_count):
        prev_cum = np.zeros((seq_len,), dtype=np.float32)
        for chunk in range(chunks_per_utt):
            idx = utt * chunks_per_utt + chunk
            utterance_index[idx] = utt
            chunk_id[idx] = chunk
            previous_chunk_id[idx] = chunk - 1
            state_reset[idx] = 1 if chunk == 0 else 0
            is_final[idx] = 1 if chunk == chunks_per_utt - 1 else 0
            end_slot = seq_len if is_final[idx] else min(seq_len, (chunk + 1) * slots_per_chunk)
            cumulative_commit_mask[idx, :end_slot] = 1.0
            new_commit_mask[idx] = np.clip(cumulative_commit_mask[idx] - prev_cum, 0.0, 1.0)
            if chunk > 0:
                old = np.flatnonzero(prev_cum > 0)
                mapped_old_slot[idx, old] = old
            prev_cum = cumulative_commit_mask[idx].copy()
            prefix_stability[idx] = float(chunk / max(chunks_per_utt - 1, 1))
            new_committed_word_count[idx] = len(set(pcn_word_id[idx][new_commit_mask[idx] > 0].tolist()))
            cumulative_committed_word_count[idx] = len(set(pcn_word_id[idx][cumulative_commit_mask[idx] > 0].tolist()))
            cn_stats[idx, :, 4] = prefix_stability[idx]

    soft_label_weight = cumulative_commit_mask * rng.uniform(0.1, 1.0, size=(count, seq_len)).astype(np.float32)
    asr_correct_target = (rng.uniform(size=(count, seq_len)) > 0.35).astype(np.float32)
    uncertainty_target = rng.uniform(0.0, 1.0, size=(count, seq_len)).astype(np.float32)
    confidence_target = np.clip(soft_label_weight * (1.0 - cn_entropy / np.log(phone_dim)), 0.0, 1.0).astype(np.float32)
    confidence_loss_mask = cumulative_commit_mask.copy()
    abstention_target = ((confidence_target < 0.25) | (uncertainty_target > 0.7)).astype(np.float32) * cumulative_commit_mask
    abstention_loss_mask = cumulative_commit_mask.copy()

    utt_target = rng.uniform(1.0, 5.0, size=(count, 5)).astype(np.float32)
    coverage_ratio = (chunk_id + 1).astype(np.float32) / float(chunks_per_utt)
    teacher_mask = (rng.uniform(size=(count,)) > 0.2).astype(np.float32)
    teacher_dim_mask = np.repeat(teacher_mask[:, None], 5, axis=1).astype(np.float32)
    teacher_prefix = np.clip(utt_target + rng.normal(scale=0.2, size=(count, 5)), 1.0, 5.0).astype(np.float32)
    teacher_final = np.clip(utt_target + rng.normal(scale=0.2, size=(count, 5)), 1.0, 5.0).astype(np.float32)
    teacher_word = np.clip(word_target[:, :, 0:3] + rng.normal(scale=0.2, size=(count, seq_len, 3)), 1.0, 5.0).astype(np.float32)
    teacher_word_mask = cumulative_commit_mask.copy()
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
        'commit_mask': cumulative_commit_mask,
        'cumulative_commit_mask': cumulative_commit_mask,
        'new_commit_mask': new_commit_mask,
        'mapped_old_slot': mapped_old_slot,
        'confidence_target': confidence_target,
        'confidence_loss_mask': confidence_loss_mask,
        'abstention_target': abstention_target,
        'abstention_loss_mask': abstention_loss_mask,
        'teacher_prefix_utt_score': teacher_prefix,
        'teacher_final_utt_score': teacher_final,
        'teacher_utt_mask': teacher_mask,
        'teacher_utt_dim_mask': teacher_dim_mask,
        'teacher_word_score': teacher_word,
        'teacher_word_mask': teacher_word_mask,
        'coverage_ratio': coverage_ratio,
        'visible_len': visible_len,
        'is_final': is_final,
        'previous_chunk_id': previous_chunk_id,
        'utterance_index': utterance_index,
        'state_reset': state_reset,
        'chunk_id': chunk_id,
        'new_committed_word_count': new_committed_word_count,
        'cumulative_committed_word_count': cumulative_committed_word_count,
        'prefix_stability': prefix_stability,
    }


def save_split(output_dir, split, arrays):
    np.savez_compressed(output_dir / f'{split}_chunks.npz', **arrays)
    with open(output_dir / f'{split}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for idx in range(arrays['cn_post'].shape[0]):
            row = {
                'utt_id': f'{split}_utt_{int(arrays["utterance_index"][idx]):04d}',
                'chunk_id': int(arrays['chunk_id'][idx]),
                'previous_chunk_id': int(arrays['previous_chunk_id'][idx]),
                'utterance_index': int(arrays['utterance_index'][idx]),
                'state_reset': int(arrays['state_reset'][idx]),
                'coverage_ratio': float(arrays['coverage_ratio'][idx]),
                'visible_len': int(arrays['visible_len'][idx]),
                'is_final': bool(arrays['is_final'][idx]),
                'new_committed_word_count': int(arrays['new_committed_word_count'][idx]),
                'cumulative_committed_word_count': int(arrays['cumulative_committed_word_count'][idx]),
                'commit_alignment_diagnostics': {
                    'mapped_old_slots': np.flatnonzero(arrays['mapped_old_slot'][idx] >= 0).astype(int).tolist(),
                    'new_slots': np.flatnonzero(arrays['new_commit_mask'][idx] > 0).astype(int).tolist(),
                    'dropped_or_revised_slots': [],
                },
                'word_timestamps': [],
                'timestamp_source': ['toy'],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    args = get_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counts = {'train': args.num_train, 'val': args.num_val, 'test': args.num_test}
    for split, count in counts.items():
        save_split(
            args.output_dir,
            split,
            build_split(rng, count, args.chunks_per_utt, args.seq_len, args.phone_dim, args.prosody_dim),
        )
    metadata = {
        'schema': SCHEMA,
        'pcn_type': 'nbest_derived_phone_confusion_network',
        'seq_len': int(args.seq_len),
        'phone_dim': int(args.phone_dim),
        'epsilon_index': int(args.phone_dim - 1),
        'prosody': [f'p{i}' for i in range(args.prosody_dim)],
        'phn_dict': {f'P{i}': i for i in range(args.phone_dim - 1)},
        'synthetic': True,
        'targets': [
            'cumulative_commit_mask',
            'new_commit_mask',
            'mapped_old_slot',
            'confidence_target',
            'confidence_loss_mask',
            'abstention_target',
            'abstention_loss_mask',
        ],
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
