import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from build_charsiu_seq_data import (
    EPS,
    align_reference_utterance,
    build_model_phone_map,
    build_reference_records,
    load_official_charsiu_aligner,
    resolve_dataset_splits,
)


def get_args():
    parser = argparse.ArgumentParser(description='Build chunked streaming GOPT data from Charsiu-aligned phone features.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json')
    parser.add_argument('--train-scp', type=str, default=None)
    parser.add_argument('--val-scp', type=str, default=None)
    parser.add_argument('--test-scp', type=str, default=None)
    parser.add_argument('--val-speaker-ratio', type=float, default=0.5, help='When --val-scp is not set, hold out this fraction of original test speakers for validation.')
    parser.add_argument('--split-seed', type=int, default=1337)
    parser.add_argument('--output-dir', type=str, default='data/streaming_charsiu_tiny')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--charsiu-src-dir', type=str, default=os.environ.get('CHARSIU_SRC_DIR'))
    parser.add_argument('--charsiu-lang', type=str, default=os.environ.get('CHARSIU_LANG', 'en'))
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--chunk-sec', type=float, default=0.64, help='Committed audio prefix step in seconds.')
    parser.add_argument('--right-context-sec', type=float, default=0.16, help='Visible future audio in seconds.')
    parser.add_argument('--min-sil-frames', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def build_phone_vocab(scores, utt_ids):
    phn_dict = {}
    for utt_id in utt_ids:
        for record in build_reference_records(scores[utt_id]):
            if record['phone'] not in phn_dict:
                phn_dict[record['phone']] = len(phn_dict)
    return phn_dict


def align_split(split_items, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict):
    aligned = []
    skipped = []
    for utt_id, audio_path in tqdm(split_items, desc='align'):
        result = align_utterance(
            utt_id=utt_id,
            audio_path=audio_path,
            scores=scores,
            charsiu=charsiu,
            sample_rate=sample_rate,
            device=device,
            phone_to_frame_id=phone_to_frame_id,
            phn_dict=phn_dict,
        )
        if 'skip_reason' in result:
            skipped.append(result)
        else:
            aligned.append(result)
    return aligned, skipped


def align_utterance(utt_id, audio_path, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict):
    return align_reference_utterance(
        utt_id=utt_id,
        audio_path=audio_path,
        scores=scores,
        charsiu=charsiu,
        sample_rate=sample_rate,
        device=device,
        phone_to_frame_id=phone_to_frame_id,
        phn_dict=phn_dict,
    )


def infer_seq_len(*aligned_splits):
    longest = 0
    for split_items in aligned_splits:
        for item in split_items:
            longest = max(longest, len(item['phones']))
    return longest


def train_norm_from_aligned(aligned_train):
    feats = []
    for item in aligned_train:
        for phone in item['phones']:
            feats.append(phone['feature'])
    if not feats:
        raise ValueError('No aligned phone features found in training split.')
    feat = np.stack(feats).astype(np.float32)
    return float(feat.mean()), float(feat.std() + EPS)


def commit_schedule(final_time, chunk_sec):
    commit_times = []
    cur = chunk_sec
    while cur < final_time - 1e-6:
        commit_times.append(float(cur))
        cur += chunk_sec
    commit_times.append(float(final_time))
    return commit_times


def build_chunk_arrays(aligned_items, seq_len, feat_dim, chunk_sec, right_context_sec):
    feat_rows = []
    phn_id_rows = []
    phn_score_rows = []
    word_rows = []
    utt_rows = []
    phone_loss_rows = []
    word_loss_rows = []
    utt_loss_rows = []
    is_final_rows = []
    visible_len_rows = []
    manifest = []

    for item in aligned_items:
        phones = item['phones']
        if not phones:
            continue

        final_time = max(phones[-1]['end_time'], 1e-4)
        for chunk_id, commit_time in enumerate(commit_schedule(final_time, chunk_sec)):
            is_final = int(abs(commit_time - final_time) < 1e-5)
            audio_end = final_time if is_final else min(final_time, commit_time + right_context_sec)

            visible_phones = [phone for phone in phones if phone['end_time'] <= audio_end + 1e-6]
            if not visible_phones:
                continue

            cur_feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
            cur_phn_id = np.zeros((seq_len,), dtype=np.int64) - 1
            cur_phn_score = np.zeros((seq_len,), dtype=np.float32) - 1
            cur_word = np.zeros((seq_len, 4), dtype=np.float32) - 1
            cur_phone_loss = np.zeros((seq_len,), dtype=np.float32)
            cur_word_loss = np.zeros((seq_len,), dtype=np.float32)
            cur_utt = np.array(
                [
                    item['utt_scores']['accuracy'],
                    item['utt_scores']['completeness'],
                    item['utt_scores']['fluency'],
                    item['utt_scores']['prosodic'],
                    item['utt_scores']['total'],
                ],
                dtype=np.float32,
            )

            for tok_idx, phone in enumerate(visible_phones):
                cur_feat[tok_idx] = phone['feature']
                cur_phn_id[tok_idx] = phone['phone_id']
                cur_phn_score[tok_idx] = phone['phone_score']
                cur_word[tok_idx, 0] = phone['word_accuracy']
                cur_word[tok_idx, 1] = phone['word_stress']
                cur_word[tok_idx, 2] = phone['word_total']
                cur_word[tok_idx, 3] = phone['word_id']

                if phone['end_time'] <= commit_time + 1e-6:
                    cur_phone_loss[tok_idx] = 1.0
                if item['word_end_times'][phone['word_id']] <= commit_time + 1e-6:
                    cur_word_loss[tok_idx] = 1.0

            utt_loss = float(is_final)
            if cur_phone_loss.sum() == 0 and cur_word_loss.sum() == 0 and utt_loss == 0:
                continue

            feat_rows.append(cur_feat)
            phn_id_rows.append(cur_phn_id)
            phn_score_rows.append(cur_phn_score)
            word_rows.append(cur_word)
            utt_rows.append(cur_utt)
            phone_loss_rows.append(cur_phone_loss)
            word_loss_rows.append(cur_word_loss)
            utt_loss_rows.append(utt_loss)
            is_final_rows.append(is_final)
            visible_len_rows.append(len(visible_phones))
            manifest.append(
                {
                    'utt_id': item['utt_id'],
                    'chunk_id': int(chunk_id),
                    'commit_time': float(commit_time),
                    'audio_end': float(audio_end),
                    'visible_phone_count': int(len(visible_phones)),
                    'committed_phone_count': int(cur_phone_loss.sum()),
                    'committed_word_phone_count': int(cur_word_loss.sum()),
                    'is_final': bool(is_final),
                }
            )

    if not feat_rows:
        raise ValueError('No streaming chunks were generated.')

    return {
        'feat': np.stack(feat_rows).astype(np.float32),
        'phn_id': np.stack(phn_id_rows).astype(np.int64),
        'phn_score': np.stack(phn_score_rows).astype(np.float32),
        'word_label': np.stack(word_rows).astype(np.float32),
        'utt_label': np.stack(utt_rows).astype(np.float32),
        'phone_loss_mask': np.stack(phone_loss_rows).astype(np.float32),
        'word_loss_mask': np.stack(word_loss_rows).astype(np.float32),
        'utt_loss_mask': np.array(utt_loss_rows, dtype=np.float32),
        'is_final': np.array(is_final_rows, dtype=np.int8),
        'visible_len': np.array(visible_len_rows, dtype=np.int32),
        'manifest': manifest,
    }


def save_chunk_split(prefix, arrays, output_dir):
    np.savez_compressed(
        output_dir / f'{prefix}_chunks.npz',
        feat=arrays['feat'],
        phn_id=arrays['phn_id'],
        phn_score=arrays['phn_score'],
        word_label=arrays['word_label'],
        utt_label=arrays['utt_label'],
        phone_loss_mask=arrays['phone_loss_mask'],
        word_loss_mask=arrays['word_loss_mask'],
        utt_loss_mask=arrays['utt_loss_mask'],
        is_final=arrays['is_final'],
        visible_len=arrays['visible_len'],
    )
    with open(output_dir / f'{prefix}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for row in arrays['manifest']:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    args = get_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild.')
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = Path(args.scores_json)
    if not scores_path.is_absolute():
        scores_path = Path.cwd() / scores_path

    with open(scores_path, 'r', encoding='utf-8') as handle:
        scores = json.load(handle)

    train_items, val_items, test_items, split_meta = resolve_dataset_splits(
        dataset_root=dataset_root,
        train_scp=args.train_scp,
        val_scp=args.val_scp,
        test_scp=args.test_scp,
        val_ratio=args.val_speaker_ratio,
        split_seed=args.split_seed,
    )
    utt_ids = [utt_id for utt_id, _ in train_items + val_items + test_items if utt_id in scores]
    phn_dict = build_phone_vocab(scores, utt_ids)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    charsiu = load_official_charsiu_aligner(
        model_name=args.aligner_model,
        device=device,
        sample_rate=args.sample_rate,
        sil_threshold=args.min_sil_frames,
        lang=args.charsiu_lang,
        charsiu_src_dir=args.charsiu_src_dir,
    )
    phone_to_frame_id, id2label, silence_ids = build_model_phone_map(charsiu)
    feat_dim = int(charsiu.aligner.config.num_labels) + 4

    aligned_train, skipped_train = align_split(
        train_items, scores, charsiu, args.sample_rate, device, phone_to_frame_id, phn_dict,
    )
    aligned_val, skipped_val = align_split(
        val_items, scores, charsiu, args.sample_rate, device, phone_to_frame_id, phn_dict,
    )
    aligned_test, skipped_test = align_split(
        test_items, scores, charsiu, args.sample_rate, device, phone_to_frame_id, phn_dict,
    )

    seq_len = infer_seq_len(aligned_train, aligned_val, aligned_test)
    train_norm_mean, train_norm_std = train_norm_from_aligned(aligned_train)
    train_arrays = build_chunk_arrays(aligned_train, seq_len, feat_dim, args.chunk_sec, args.right_context_sec)
    val_arrays = build_chunk_arrays(aligned_val, seq_len, feat_dim, args.chunk_sec, args.right_context_sec)
    test_arrays = build_chunk_arrays(aligned_test, seq_len, feat_dim, args.chunk_sec, args.right_context_sec)

    save_chunk_split('train', train_arrays, output_dir)
    save_chunk_split('val', val_arrays, output_dir)
    save_chunk_split('test', test_arrays, output_dir)

    metadata = {
        'aligner_model': args.aligner_model,
        'charsiu_src_dir': args.charsiu_src_dir,
        'charsiu_lang': args.charsiu_lang,
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'split_meta': split_meta,
        'chunk_sec': float(args.chunk_sec),
        'right_context_sec': float(args.right_context_sec),
        'seq_len': int(seq_len),
        'feat_dim': int(feat_dim),
        'phn_dict': phn_dict,
        'phn_num': int(len(phn_dict) + 1),
        'train_norm_mean': float(train_norm_mean),
        'train_norm_std': float(train_norm_std),
        'num_frame_labels': int(charsiu.aligner.config.num_labels),
        'frame_id2label': {str(key): str(value) for key, value in id2label.items()},
        'silence_ids': [int(x) for x in silence_ids],
        'train_chunks': int(train_arrays['feat'].shape[0]),
        'val_chunks': int(val_arrays['feat'].shape[0]),
        'test_chunks': int(test_arrays['feat'].shape[0]),
        'train_final_chunks': int(train_arrays['is_final'].sum()),
        'val_final_chunks': int(val_arrays['is_final'].sum()),
        'test_final_chunks': int(test_arrays['is_final'].sum()),
        'skipped_train': skipped_train,
        'skipped_val': skipped_val,
        'skipped_test': skipped_test,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
