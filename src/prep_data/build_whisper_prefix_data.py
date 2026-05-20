import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from build_charsiu_seq_data import (
    build_model_phone_map,
    build_reference_word_records,
    load_frame_model,
    parse_wav_scp,
)
from build_streaming_charsiu_data import align_split, commit_schedule


def get_args():
    parser = argparse.ArgumentParser(description='Build prefix-level streaming ASR manifests from offline Charsiu alignments.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json')
    parser.add_argument('--train-scp', type=str, default=None)
    parser.add_argument('--test-scp', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='data/streaming_whisper_prefix')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--chunk-sec', type=float, default=0.64, help='Committed prefix step in seconds.')
    parser.add_argument('--right-context-sec', type=float, default=0.16, help='Visible future audio in seconds.')
    parser.add_argument('--min-sil-frames', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def build_prefix_rows(aligned_items, scores, chunk_sec, right_context_sec):
    rows = []
    for item in aligned_items:
        utt_id = item['utt_id']
        gold_words = build_reference_word_records(scores[utt_id])
        word_end_times = item['word_end_times']
        if not gold_words:
            continue

        final_time = max(item['phones'][-1]['end_time'], 1e-4)
        prev_target_text = ''
        for chunk_id, commit_time in enumerate(commit_schedule(final_time, chunk_sec)):
            is_final = abs(commit_time - final_time) < 1e-5
            audio_end = final_time if is_final else min(final_time, commit_time + right_context_sec)

            committed_words = [word for word in gold_words if word_end_times.get(word['word_id'], 0.0) <= commit_time + 1e-6]
            visible_words = [word for word in gold_words if word_end_times.get(word['word_id'], 0.0) <= audio_end + 1e-6]
            if not visible_words:
                continue

            target_text = ' '.join(word['display_text'] for word in committed_words).strip()
            visible_text = ' '.join(word['display_text'] for word in visible_words).strip()
            full_text = ' '.join(word['display_text'] for word in gold_words).strip()
            if not target_text:
                continue

            rows.append({
                'utt_id': utt_id,
                'chunk_id': int(chunk_id),
                'audio_path': item['audio_path'],
                'audio_start': 0.0,
                'audio_end': float(audio_end),
                'commit_time': float(commit_time),
                'right_context_sec': float(max(audio_end - commit_time, 0.0)),
                'target_text': target_text,
                'visible_text': visible_text,
                'prompt_text': prev_target_text,
                'full_text': full_text,
                'committed_word_ids': [int(word['word_id']) for word in committed_words],
                'visible_word_ids': [int(word['word_id']) for word in visible_words],
                'committed_word_count': int(len(committed_words)),
                'visible_word_count': int(len(visible_words)),
                'is_final': bool(is_final),
            })
            prev_target_text = target_text
    return rows


def save_rows(prefix, rows, output_dir):
    with open(output_dir / f'{prefix}_prefix.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    args = get_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild.')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scp = Path(args.train_scp) if args.train_scp else dataset_root / 'train' / 'wav.scp'
    test_scp = Path(args.test_scp) if args.test_scp else dataset_root / 'test' / 'wav.scp'
    scores_path = Path(args.scores_json)
    if not scores_path.is_absolute():
        scores_path = Path.cwd() / scores_path

    with open(scores_path, 'r', encoding='utf-8') as handle:
        scores = json.load(handle)

    train_items = parse_wav_scp(train_scp, dataset_root)
    test_items = parse_wav_scp(test_scp, dataset_root)
    utt_ids = [utt_id for utt_id, _ in train_items + test_items if utt_id in scores]

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(args.aligner_model)
    model = load_frame_model(args.aligner_model).to(device).eval()
    phone_to_frame_id, _, silence_ids = build_model_phone_map(model)
    phn_dict = {}
    for utt_id in utt_ids:
        for word in build_reference_word_records(scores[utt_id]):
            for phone in word['phones']:
                if phone not in phn_dict:
                    phn_dict[phone] = len(phn_dict)

    aligned_train, skipped_train = align_split(
        train_items, scores, processor, model, args.sample_rate, device, args.min_sil_frames, phone_to_frame_id, silence_ids, phn_dict,
    )
    aligned_test, skipped_test = align_split(
        test_items, scores, processor, model, args.sample_rate, device, args.min_sil_frames, phone_to_frame_id, silence_ids, phn_dict,
    )

    train_rows = build_prefix_rows(aligned_train, scores, args.chunk_sec, args.right_context_sec)
    test_rows = build_prefix_rows(aligned_test, scores, args.chunk_sec, args.right_context_sec)
    save_rows('train', train_rows, output_dir)
    save_rows('test', test_rows, output_dir)

    metadata = {
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'aligner_model': args.aligner_model,
        'chunk_sec': float(args.chunk_sec),
        'right_context_sec': float(args.right_context_sec),
        'sample_rate': int(args.sample_rate),
        'train_rows': int(len(train_rows)),
        'test_rows': int(len(test_rows)),
        'train_final_rows': int(sum(1 for row in train_rows if row['is_final'])),
        'test_final_rows': int(sum(1 for row in test_rows if row['is_final'])),
        'skipped_train': skipped_train,
        'skipped_test': skipped_test,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
