import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, pipeline

from build_charsiu_seq_data import (
    EPS,
    audio_logits,
    build_model_phone_map,
    build_reference_word_records,
    load_frame_model,
    monotonic_align,
    normalize_phone,
    normalize_word,
    parse_wav_scp,
    segment_feature,
)
from build_streaming_charsiu_data import commit_schedule


def get_args():
    parser = argparse.ArgumentParser(description='Build streaming GOPT chunks from ASR hypotheses instead of gold text.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json')
    parser.add_argument('--train-scp', type=str, default=None)
    parser.add_argument('--test-scp', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='data/streaming_asr_gopt')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--asr-model', type=str, default='openai/whisper-base')
    parser.add_argument('--timestamp-backend', type=str, default='transformers', choices=['transformers', 'whisper_timestamped'])
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--chunk-sec', type=float, default=0.64)
    parser.add_argument('--right-context-sec', type=float, default=0.16)
    parser.add_argument('--min-sil-frames', type=int, default=4)
    parser.add_argument('--min-utt-match-ratio', type=float, default=0.5, help='Minimum matched committed-word ratio required to keep utterance loss on final chunk.')
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--best-of', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def build_phone_vocab(scores, utt_ids):
    phn_dict = {}
    for utt_id in utt_ids:
        for word in build_reference_word_records(scores[utt_id]):
            for phone in word['phones']:
                if phone not in phn_dict:
                    phn_dict[phone] = len(phn_dict)
    return phn_dict


def build_word_lexicon(scores, utt_ids):
    votes = defaultdict(Counter)
    for utt_id in utt_ids:
        for word in build_reference_word_records(scores[utt_id]):
            if word['text'] and word['phones']:
                votes[word['text']][tuple(word['phones'])] += 1
    lexicon = {}
    for word, counter in votes.items():
        phones = max(counter.items(), key=lambda item: (item[1], -len(item[0])))[0]
        lexicon[word] = list(phones)
    return lexicon


def longest_common_prefix_len(prev_words, curr_words):
    limit = min(len(prev_words), len(curr_words))
    idx = 0
    while idx < limit and prev_words[idx]['text'] == curr_words[idx]['text']:
        idx += 1
    return idx


def lcs_align_words(asr_words, gold_words):
    n = len(asr_words)
    m = len(gold_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if asr_words[i]['text'] == gold_words[j]['text']:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    mapping = [-1] * n
    i = 0
    j = 0
    while i < n and j < m:
        if asr_words[i]['text'] == gold_words[j]['text']:
            mapping[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return mapping


def build_transformers_asr(asr_model, language, device):
    if device.startswith('cuda'):
        pipe_device = int(device.split(':', 1)[1]) if ':' in device else 0
    else:
        pipe_device = -1
    return pipeline(
        'automatic-speech-recognition',
        model=asr_model,
        tokenizer=asr_model,
        feature_extractor=asr_model,
        device=pipe_device,
    ), {
        'return_timestamps': 'word',
        'generate_kwargs': {
            'language': language,
            'task': 'transcribe',
        },
    }


def build_whisper_timestamped_asr(asr_model, device):
    try:
        import whisper_timestamped as whisper
    except ImportError as exc:
        raise ImportError('whisper-timestamped is required for --timestamp-backend whisper_timestamped') from exc
    return whisper.load_model(asr_model, device=device), whisper


def extract_words_from_transformers(result):
    words = []
    for chunk in result.get('chunks', []):
        text = normalize_word(chunk.get('text', ''))
        ts = chunk.get('timestamp') or (None, None)
        if not text or ts[0] is None or ts[1] is None:
            continue
        words.append({
            'text': text,
            'display_text': text.lower(),
            'start': float(ts[0]),
            'end': float(ts[1]),
            'confidence': None,
        })
    return words


def extract_words_from_whisper_timestamped(result):
    words = []
    for segment in result.get('segments', []):
        for word in segment.get('words', []):
            text = normalize_word(word.get('text', ''))
            start = word.get('start', None)
            end = word.get('end', None)
            if not text or start is None or end is None:
                continue
            words.append({
                'text': text,
                'display_text': text.lower(),
                'start': float(start),
                'end': float(end),
                'confidence': None if word.get('confidence') is None else float(word['confidence']),
            })
    return words


def transcribe_audio_prefix(audio_prefix, sample_rate, backend_name, backend_model, backend_kwargs, language, beam_size, best_of):
    if backend_name == 'transformers':
        result = backend_model(
            {
                'array': audio_prefix,
                'sampling_rate': sample_rate,
            },
            **backend_kwargs,
        )
        return extract_words_from_transformers(result)

    whisper = backend_kwargs['module']
    result = whisper.transcribe(
        backend_model,
        audio_prefix,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
    )
    return extract_words_from_whisper_timestamped(result)


def select_visible_frames(probs, keep_mask, audio_end, frame_step):
    frame_limit = max(int(math.ceil(audio_end / max(frame_step, EPS))), 1)
    frame_mask = np.zeros(len(probs), dtype=bool)
    frame_mask[: min(frame_limit, len(probs))] = True
    visible_mask = keep_mask & frame_mask
    return np.flatnonzero(visible_mask), probs[visible_mask]


def align_gold_utterance(utt_id, audio_path, scores, processor, model, sample_rate, device, min_sil_frames):
    if utt_id not in scores or not audio_path.exists():
        return None

    gold_words = build_reference_word_records(scores[utt_id])
    probs, audio_duration = audio_logits(audio_path, processor, model, sample_rate, device)
    frame_step = audio_duration / max(len(probs), 1)
    frame_labels = np.argmax(probs, axis=-1)
    phone_to_frame_id, _, silence_ids = build_model_phone_map(model)
    keep_mask = np.ones(len(probs), dtype=bool)
    if silence_ids:
        silence_ids = set(int(x) for x in silence_ids)
        start = 0
        while start < len(frame_labels):
            cur = int(frame_labels[start])
            end = start + 1
            while end < len(frame_labels) and int(frame_labels[end]) == cur:
                end += 1
            if cur in silence_ids and (end - start) >= min_sil_frames:
                keep_mask[start:end] = False
            start = end
        if not np.any(keep_mask):
            keep_mask[:] = True

    word_end_times = {}
    word_start_times = {}
    visible_phone_seq = []
    for word in gold_words:
        visible_phone_seq.extend(word['phones'])
    missing_phone = next((phone for phone in visible_phone_seq if phone not in phone_to_frame_id), None)
    if missing_phone is not None:
        return {'utt_id': utt_id, 'skip_reason': f'phone_not_in_model_vocab:{missing_phone}'}

    kept_indices = np.flatnonzero(keep_mask)
    kept_probs = probs[keep_mask]
    try:
        phone_ids = [phone_to_frame_id[phone] for phone in visible_phone_seq]
        path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), phone_ids)
    except Exception as exc:
        return {'utt_id': utt_id, 'skip_reason': f'gold_alignment_failed:{exc}'}

    phone_offset = 0
    for word in gold_words:
        word_frames = []
        for _ in word['phones']:
            tok_frames = kept_indices[path == phone_offset]
            if tok_frames.size == 0:
                return {'utt_id': utt_id, 'skip_reason': f'empty_gold_phone_segment:{phone_offset}'}
            word_frames.append(tok_frames)
            phone_offset += 1
        start_time = float(word_frames[0][0] * frame_step)
        end_time = float((word_frames[-1][-1] + 1) * frame_step)
        word_start_times[word['word_id']] = start_time
        word_end_times[word['word_id']] = end_time

    return {
        'utt_id': utt_id,
        'audio_path': str(audio_path),
        'audio_duration': float(audio_duration),
        'frame_step': float(frame_step),
        'probs': probs,
        'keep_mask': keep_mask,
        'word_start_times': word_start_times,
        'word_end_times': word_end_times,
        'gold_words': gold_words,
        'utt_scores': {
            'accuracy': float(scores[utt_id]['accuracy']),
            'completeness': float(scores[utt_id]['completeness']),
            'fluency': float(scores[utt_id]['fluency']),
            'prosodic': float(scores[utt_id]['prosodic']),
            'total': float(scores[utt_id]['total']),
        },
    }


def align_split(split_items, scores, processor, model, sample_rate, device, min_sil_frames):
    aligned = []
    skipped = []
    for utt_id, audio_path in tqdm(split_items, desc='gold-align'):
        result = align_gold_utterance(utt_id, audio_path, scores, processor, model, sample_rate, device, min_sil_frames)
        if result is None:
            skipped.append({'utt_id': utt_id, 'skip_reason': 'missing_scores_or_audio'})
        elif 'skip_reason' in result:
            skipped.append(result)
        else:
            aligned.append(result)
    return aligned, skipped


def build_chunk_examples(aligned_items, asr_backend_name, asr_backend_model, asr_backend_kwargs, args, lexicon, phn_dict, phone_to_frame_id):
    examples = []
    skipped_chunks = []
    for item in tqdm(aligned_items, desc='asr-chunks'):
        utt_id = item['utt_id']
        audio, _ = librosa.load(item['audio_path'], sr=args.sample_rate, mono=True)
        gold_words = item['gold_words']
        gold_word_mapping = {word['word_id']: word for word in gold_words}
        final_time = max(item['word_end_times'].values()) if item['word_end_times'] else item['audio_duration']
        prev_visible_words = []

        for chunk_id, commit_time in enumerate(commit_schedule(final_time, args.chunk_sec)):
            is_final = abs(commit_time - final_time) < 1e-5
            audio_end = final_time if is_final else min(final_time, commit_time + args.right_context_sec)
            audio_prefix = audio[: int(max(audio_end, 1e-4) * args.sample_rate)]
            asr_words = transcribe_audio_prefix(
                audio_prefix=audio_prefix,
                sample_rate=args.sample_rate,
                backend_name=asr_backend_name,
                backend_model=asr_backend_model,
                backend_kwargs=asr_backend_kwargs,
                language=args.language,
                beam_size=args.beam_size,
                best_of=args.best_of,
            )
            if not asr_words:
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': 'empty_asr_hypothesis'})
                prev_visible_words = []
                continue

            timestamp_commit_len = sum(1 for word in asr_words if word['end'] <= commit_time + 1e-6)
            stable_prefix_len = longest_common_prefix_len(prev_visible_words, asr_words) if prev_visible_words else timestamp_commit_len
            committed_len = timestamp_commit_len if is_final else min(timestamp_commit_len, stable_prefix_len)
            prev_visible_words = asr_words

            asr_to_gold = lcs_align_words(asr_words, gold_words)
            pseudo_records = []
            matched_committed_words = 0
            committed_lexicon_words = 0
            visible_word_local_id = 0

            for asr_idx, asr_word in enumerate(asr_words):
                phones = lexicon.get(asr_word['text'])
                if not phones:
                    continue
                committed = asr_idx < committed_len
                if committed:
                    committed_lexicon_words += 1

                gold_idx = asr_to_gold[asr_idx]
                gold_word = None
                word_match = False
                if gold_idx >= 0 and gold_words[gold_idx]['text'] == asr_word['text']:
                    gold_word = gold_words[gold_idx]
                    if gold_word['phones'] == phones:
                        word_match = True
                if committed and word_match:
                    matched_committed_words += 1

                for phone_idx, phone in enumerate(phones):
                    if phone not in phone_to_frame_id or phone not in phn_dict:
                        continue
                    phone_score = -1.0
                    word_accuracy = -1.0
                    word_stress = -1.0
                    word_total = -1.0
                    loss_ok = False
                    if word_match and gold_word is not None and phone_idx < len(gold_word['phone_scores']):
                        phone_score = float(gold_word['phone_scores'][phone_idx])
                        word_accuracy = float(gold_word['accuracy'])
                        word_stress = float(gold_word['stress'])
                        word_total = float(gold_word['total'])
                        loss_ok = committed
                    pseudo_records.append({
                        'phone': phone,
                        'phone_id': int(phn_dict[phone]),
                        'phone_score': phone_score,
                        'word_local_id': int(visible_word_local_id),
                        'word_accuracy': word_accuracy,
                        'word_stress': word_stress,
                        'word_total': word_total,
                        'word_match': bool(word_match),
                        'committed': bool(committed),
                        'word_end': float(asr_word['end']),
                        'phone_loss_ok': bool(loss_ok),
                        'display_word': asr_word['display_text'],
                    })
                visible_word_local_id += 1

            if not pseudo_records:
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': 'no_alignable_asr_phones'})
                continue

            kept_indices, kept_probs = select_visible_frames(item['probs'], item['keep_mask'], audio_end, item['frame_step'])
            if kept_probs.shape[0] < len(pseudo_records):
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': 'not_enough_frames_for_asr_phones'})
                continue

            try:
                phone_ids = [phone_to_frame_id[record['phone']] for record in pseudo_records]
                path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), phone_ids)
            except Exception as exc:
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': f'asr_alignment_failed:{exc}'})
                continue

            segments = []
            for tok_idx, record in enumerate(pseudo_records):
                tok_frames = kept_indices[path == tok_idx]
                if tok_frames.size == 0:
                    segments = []
                    break
                target_id = phone_to_frame_id[record['phone']]
                feature = segment_feature(item['probs'][tok_frames], target_id, item['frame_step'])
                end_time = float((int(tok_frames[-1]) + 1) * item['frame_step'])
                segments.append({
                    'feature': feature.astype(np.float32),
                    'phone_id': int(record['phone_id']),
                    'phone_score': float(record['phone_score']),
                    'word_id': int(record['word_local_id']),
                    'word_accuracy': float(record['word_accuracy']),
                    'word_stress': float(record['word_stress']),
                    'word_total': float(record['word_total']),
                    'phone_loss_mask': float(record['phone_loss_ok'] and end_time <= commit_time + 1e-6),
                    'word_loss_mask': float(record['phone_loss_ok'] and record['word_end'] <= commit_time + 1e-6),
                    'end_time': end_time,
                })
            if not segments:
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': 'empty_asr_phone_segment'})
                continue

            matched_ratio = float(matched_committed_words) / float(max(committed_lexicon_words, 1))
            utt_loss_mask = float(is_final and matched_ratio >= args.min_utt_match_ratio)
            phone_loss_count = sum(segment['phone_loss_mask'] for segment in segments)
            word_loss_count = sum(segment['word_loss_mask'] for segment in segments)
            if phone_loss_count == 0 and word_loss_count == 0 and utt_loss_mask == 0:
                skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': 'no_supervised_tokens_after_matching'})
                continue
            examples.append({
                'utt_id': utt_id,
                'chunk_id': int(chunk_id),
                'audio_end': float(audio_end),
                'commit_time': float(commit_time),
                'is_final': bool(is_final),
                'utt_scores': item['utt_scores'],
                'segments': segments,
                'matched_ratio': matched_ratio,
                'matched_committed_words': int(matched_committed_words),
                'committed_lexicon_words': int(committed_lexicon_words),
                'utt_loss_mask': utt_loss_mask,
            })
    return examples, skipped_chunks


def infer_seq_len(examples):
    return max(len(example['segments']) for example in examples)


def train_norm_from_examples(examples):
    feats = []
    for example in examples:
        for segment in example['segments']:
            feats.append(segment['feature'])
    feat = np.stack(feats).astype(np.float32)
    return float(feat.mean()), float(feat.std() + EPS)


def build_arrays(examples, seq_len, feat_dim):
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

    for example in examples:
        cur_feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
        cur_phn_id = np.zeros((seq_len,), dtype=np.int64) - 1
        cur_phn_score = np.zeros((seq_len,), dtype=np.float32) - 1
        cur_word = np.zeros((seq_len, 4), dtype=np.float32) - 1
        cur_phone_loss = np.zeros((seq_len,), dtype=np.float32)
        cur_word_loss = np.zeros((seq_len,), dtype=np.float32)
        cur_utt = np.array(
            [
                example['utt_scores']['accuracy'],
                example['utt_scores']['completeness'],
                example['utt_scores']['fluency'],
                example['utt_scores']['prosodic'],
                example['utt_scores']['total'],
            ],
            dtype=np.float32,
        )

        for tok_idx, segment in enumerate(example['segments']):
            cur_feat[tok_idx] = segment['feature']
            cur_phn_id[tok_idx] = segment['phone_id']
            cur_phn_score[tok_idx] = segment['phone_score']
            cur_word[tok_idx, 0] = segment['word_accuracy']
            cur_word[tok_idx, 1] = segment['word_stress']
            cur_word[tok_idx, 2] = segment['word_total']
            cur_word[tok_idx, 3] = segment['word_id']
            cur_phone_loss[tok_idx] = segment['phone_loss_mask']
            cur_word_loss[tok_idx] = segment['word_loss_mask']

        feat_rows.append(cur_feat)
        phn_id_rows.append(cur_phn_id)
        phn_score_rows.append(cur_phn_score)
        word_rows.append(cur_word)
        utt_rows.append(cur_utt)
        phone_loss_rows.append(cur_phone_loss)
        word_loss_rows.append(cur_word_loss)
        utt_loss_rows.append(example['utt_loss_mask'])
        is_final_rows.append(int(example['is_final']))
        visible_len_rows.append(len(example['segments']))
        manifest.append({
            'utt_id': example['utt_id'],
            'chunk_id': example['chunk_id'],
            'commit_time': example['commit_time'],
            'audio_end': example['audio_end'],
            'visible_phone_count': int(len(example['segments'])),
            'committed_phone_count': int(cur_phone_loss.sum()),
            'committed_word_phone_count': int(cur_word_loss.sum()),
            'matched_ratio': float(example['matched_ratio']),
            'matched_committed_words': int(example['matched_committed_words']),
            'committed_lexicon_words': int(example['committed_lexicon_words']),
            'utt_loss_mask': float(example['utt_loss_mask']),
            'is_final': bool(example['is_final']),
        })

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
    phn_dict = build_phone_vocab(scores, utt_ids)
    lexicon = build_word_lexicon(scores, utt_ids)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    aligner_processor = AutoProcessor.from_pretrained(args.aligner_model)
    aligner_model = load_frame_model(args.aligner_model).to(device).eval()
    phone_to_frame_id, id2label, silence_ids = build_model_phone_map(aligner_model)
    feat_dim = int(aligner_model.config.num_labels) + 4

    if args.timestamp_backend == 'transformers':
        asr_backend_model, asr_backend_kwargs = build_transformers_asr(args.asr_model, args.language, device)
        asr_backend_name = 'transformers'
    else:
        asr_model, whisper_module = build_whisper_timestamped_asr(args.asr_model, device)
        asr_backend_model = asr_model
        asr_backend_kwargs = {'module': whisper_module}
        asr_backend_name = 'whisper_timestamped'

    aligned_train, skipped_train = align_split(
        train_items, scores, aligner_processor, aligner_model, args.sample_rate, device, args.min_sil_frames,
    )
    aligned_test, skipped_test = align_split(
        test_items, scores, aligner_processor, aligner_model, args.sample_rate, device, args.min_sil_frames,
    )

    train_examples, skipped_train_chunks = build_chunk_examples(
        aligned_items=aligned_train,
        asr_backend_name=asr_backend_name,
        asr_backend_model=asr_backend_model,
        asr_backend_kwargs=asr_backend_kwargs,
        args=args,
        lexicon=lexicon,
        phn_dict=phn_dict,
        phone_to_frame_id=phone_to_frame_id,
    )
    test_examples, skipped_test_chunks = build_chunk_examples(
        aligned_items=aligned_test,
        asr_backend_name=asr_backend_name,
        asr_backend_model=asr_backend_model,
        asr_backend_kwargs=asr_backend_kwargs,
        args=args,
        lexicon=lexicon,
        phn_dict=phn_dict,
        phone_to_frame_id=phone_to_frame_id,
    )

    if not train_examples or not test_examples:
        raise ValueError('No ASR-driven streaming chunks were generated.')

    seq_len = infer_seq_len(train_examples + test_examples)
    train_norm_mean, train_norm_std = train_norm_from_examples(train_examples)
    train_arrays = build_arrays(train_examples, seq_len, feat_dim)
    test_arrays = build_arrays(test_examples, seq_len, feat_dim)

    save_chunk_split('train', train_arrays, output_dir)
    save_chunk_split('test', test_arrays, output_dir)

    metadata = {
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'aligner_model': args.aligner_model,
        'asr_model': args.asr_model,
        'timestamp_backend': args.timestamp_backend,
        'chunk_sec': float(args.chunk_sec),
        'right_context_sec': float(args.right_context_sec),
        'seq_len': int(seq_len),
        'feat_dim': int(feat_dim),
        'phn_dict': phn_dict,
        'phn_num': int(len(phn_dict) + 1),
        'train_norm_mean': float(train_norm_mean),
        'train_norm_std': float(train_norm_std),
        'num_frame_labels': int(aligner_model.config.num_labels),
        'frame_id2label': {str(key): str(value) for key, value in id2label.items()},
        'silence_ids': [int(x) for x in silence_ids],
        'lexicon_size': int(len(lexicon)),
        'train_chunks': int(train_arrays['feat'].shape[0]),
        'test_chunks': int(test_arrays['feat'].shape[0]),
        'train_final_chunks': int(train_arrays['is_final'].sum()),
        'test_final_chunks': int(test_arrays['is_final'].sum()),
        'skipped_train': skipped_train,
        'skipped_test': skipped_test,
        'skipped_train_chunks': skipped_train_chunks,
        'skipped_test_chunks': skipped_test_chunks,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
