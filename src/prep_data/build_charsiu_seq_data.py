import argparse
import json
import math
import os
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForAudioFrameClassification
except ImportError:
    AutoModelForAudioFrameClassification = None

try:
    from transformers import Wav2Vec2ForFrameClassification
except ImportError:
    Wav2Vec2ForFrameClassification = None


EPS = 1e-8


def get_args():
    parser = argparse.ArgumentParser(description='Build GOPT sequence data from Charsiu frame logits without Kaldi.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json', help='SpeechOcean phone/word/utt labels.')
    parser.add_argument('--train-scp', type=str, default=None, help='Optional explicit train wav.scp path.')
    parser.add_argument('--test-scp', type=str, default=None, help='Optional explicit test wav.scp path.')
    parser.add_argument('--output-dir', type=str, default='data/seq_data_charsiu_tiny', help='Directory to save sequence npy files.')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms', help='HF model used for frame classification.')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default=None, help='cuda / cpu. Defaults to cuda if available.')
    parser.add_argument('--max-seq-len', type=int, default=0, help='0 means infer from data.')
    parser.add_argument('--min-sil-frames', type=int, default=4, help='Consecutive silence frames to strip before alignment.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def normalize_phone(phone):
    phone = phone.upper().strip()
    phone = phone.replace('[', '').replace(']', '')
    phone = re.sub(r'[_\d].*', '', phone)
    return phone


def normalize_word(word):
    word = word.upper().strip()
    word = word.replace('[', '').replace(']', '')
    word = re.sub(r"[^A-Z']", '', word)
    return word


def display_word(word):
    return normalize_word(word).lower()


def parse_wav_scp(path, dataset_root):
    items = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            utt_id, rel_path = line.split(maxsplit=1)
            rel_path = rel_path.replace('/', os.sep)
            abs_path = Path(dataset_root) / rel_path
            items.append((utt_id, abs_path))
    return items


def build_reference_records(utt_info):
    phones = []
    for word_id, word in enumerate(utt_info['words']):
        phone_scores = word['phones-accuracy']
        if len(word['phones']) != len(phone_scores):
            raise ValueError('Phone and phone-score lengths mismatch.')
        for phone, phone_score in zip(word['phones'], phone_scores):
            phones.append({
                'phone': normalize_phone(phone),
                'phone_score': float(phone_score),
                'word_id': int(word_id),
                'word_accuracy': float(word['accuracy']),
                'word_stress': float(word['stress']),
                'word_total': float(word['total']),
            })
    return phones


def build_reference_word_records(utt_info):
    words = []
    for word_id, word in enumerate(utt_info['words']):
        phones = [normalize_phone(phone) for phone in word['phones']]
        phone_scores = [float(score) for score in word['phones-accuracy']]
        if len(phones) != len(phone_scores):
            raise ValueError('Phone and phone-score lengths mismatch.')
        words.append({
            'word_id': int(word_id),
            'text': normalize_word(word.get('text', '')),
            'display_text': display_word(word.get('text', '')),
            'phones': phones,
            'phone_scores': phone_scores,
            'accuracy': float(word['accuracy']),
            'stress': float(word['stress']),
            'total': float(word['total']),
        })
    return words


def words_to_text(words, key='display_text'):
    tokens = [word[key] for word in words if word.get(key)]
    return ' '.join(tokens).strip()


def infer_seq_len(scores, utt_ids, max_seq_len):
    longest = 0
    for utt_id in utt_ids:
        longest = max(longest, len(build_reference_records(scores[utt_id])))
    if max_seq_len > 0:
        if longest > max_seq_len:
            raise ValueError(f'max_seq_len={max_seq_len} is smaller than longest phone sequence={longest}.')
        return max_seq_len
    return longest


def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(np.sum(exp_x, axis=-1, keepdims=True), a_min=EPS, a_max=None)


def load_frame_model(model_name):
    if AutoModelForAudioFrameClassification is not None:
        return AutoModelForAudioFrameClassification.from_pretrained(model_name)
    if Wav2Vec2ForFrameClassification is not None:
        return Wav2Vec2ForFrameClassification.from_pretrained(model_name)
    raise ImportError('No audio frame classification model class is available in transformers.')


def build_model_phone_map(model):
    id2label = model.config.id2label
    phone_to_id = {}
    sil_ids = []
    for idx, label in id2label.items():
        norm = normalize_phone(str(label))
        if not norm:
            continue
        if 'SIL' in norm:
            sil_ids.append(int(idx))
        if norm not in phone_to_id:
            phone_to_id[norm] = int(idx)
    return phone_to_id, id2label, sil_ids


def build_silence_mask(frame_labels, silence_ids, min_sil_frames):
    if not silence_ids:
        return np.ones(len(frame_labels), dtype=bool)
    silence_ids = set(int(x) for x in silence_ids)
    keep = np.ones(len(frame_labels), dtype=bool)
    start = 0
    while start < len(frame_labels):
        current = int(frame_labels[start])
        end = start + 1
        while end < len(frame_labels) and int(frame_labels[end]) == current:
            end += 1
        if current in silence_ids and (end - start) >= min_sil_frames:
            keep[start:end] = False
        start = end
    if not np.any(keep):
        keep[:] = True
    return keep


def monotonic_align(neg_log_probs, phone_ids):
    frame_num = neg_log_probs.shape[0]
    token_num = len(phone_ids)
    if frame_num < token_num:
        raise ValueError(f'Not enough frames ({frame_num}) to align {token_num} phones.')

    dp = np.full((frame_num, token_num), np.inf, dtype=np.float32)
    back = np.zeros((frame_num, token_num), dtype=np.int8)
    dp[0, 0] = neg_log_probs[0, phone_ids[0]]

    for t in range(1, frame_num):
        max_token = min(t, token_num - 1)
        for tok in range(max_token + 1):
            stay_cost = dp[t - 1, tok]
            move_cost = dp[t - 1, tok - 1] if tok > 0 else np.inf
            if move_cost < stay_cost:
                dp[t, tok] = move_cost + neg_log_probs[t, phone_ids[tok]]
                back[t, tok] = 1
            else:
                dp[t, tok] = stay_cost + neg_log_probs[t, phone_ids[tok]]

    path = np.zeros(frame_num, dtype=np.int32)
    tok = token_num - 1
    for t in range(frame_num - 1, -1, -1):
        path[t] = tok
        if t > 0 and back[t, tok] == 1:
            tok -= 1
    return path


def segment_feature(segment_probs, target_idx, frame_step):
    mean_probs = np.mean(segment_probs, axis=0)
    target_prob = float(mean_probs[target_idx])
    if mean_probs.shape[0] > 1:
        other_probs = np.delete(mean_probs, target_idx)
        max_other = float(np.max(other_probs))
    else:
        max_other = 0.0
    entropy = float(-(mean_probs * np.log(np.clip(mean_probs, EPS, None))).sum())
    duration = float(segment_probs.shape[0] * frame_step)
    stats = np.array([target_prob, target_prob - max_other, entropy, duration], dtype=np.float32)
    return np.concatenate([mean_probs.astype(np.float32), stats], axis=0)


def audio_logits(audio_path, processor, model, sample_rate, device):
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0].detach().cpu().numpy()
    return softmax_np(logits), len(audio) / sample_rate


def build_seq_arrays(split_items, scores, seq_len, feat_dim, phone_to_id, sil_ids, processor, model, sample_rate, device, min_sil_frames, phn_dict):
    feat_rows = []
    phn_rows = []
    word_rows = []
    utt_rows = []
    report = {'kept': 0, 'skipped': []}

    for utt_id, audio_path in tqdm(split_items, desc='building'):
        if utt_id not in scores:
            report['skipped'].append({'utt_id': utt_id, 'reason': 'missing_scores'})
            continue
        if not audio_path.exists():
            report['skipped'].append({'utt_id': utt_id, 'reason': 'missing_audio', 'audio_path': str(audio_path)})
            continue

        ref_records = build_reference_records(scores[utt_id])
        ref_phones = [record['phone'] for record in ref_records]
        missing_phone = next((phone for phone in ref_phones if phone not in phone_to_id), None)
        if missing_phone is not None:
            report['skipped'].append({'utt_id': utt_id, 'reason': 'phone_not_in_model_vocab', 'phone': missing_phone})
            continue

        probs, audio_duration = audio_logits(audio_path, processor, model, sample_rate, device)
        frame_step = audio_duration / max(len(probs), 1)
        frame_labels = np.argmax(probs, axis=-1)
        keep_mask = build_silence_mask(frame_labels, sil_ids, min_sil_frames)
        kept_indices = np.flatnonzero(keep_mask)
        kept_probs = probs[keep_mask]

        try:
            phone_ids = [phone_to_id[phone] for phone in ref_phones]
            path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), phone_ids)
        except Exception as exc:
            report['skipped'].append({'utt_id': utt_id, 'reason': 'alignment_failed', 'detail': str(exc)})
            continue

        cur_feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
        cur_phn = np.zeros((seq_len, 2), dtype=np.float32) - 1
        cur_word = np.zeros((seq_len, 4), dtype=np.float32) - 1
        cur_utt = np.zeros((5,), dtype=np.float32)
        success = True
        for tok_idx, record in enumerate(ref_records):
            tok_frames = kept_indices[path == tok_idx]
            if tok_frames.size == 0:
                report['skipped'].append({'utt_id': utt_id, 'reason': 'empty_phone_segment', 'token_index': tok_idx})
                success = False
                break
            cur_probs = probs[tok_frames]
            target_id = phone_to_id[record['phone']]
            cur_feat[tok_idx, :] = segment_feature(cur_probs, target_id, frame_step)
            cur_phn[tok_idx, 0] = phn_dict[record['phone']]
            cur_phn[tok_idx, 1] = record['phone_score']
            cur_word[tok_idx, 0] = record['word_accuracy']
            cur_word[tok_idx, 1] = record['word_stress']
            cur_word[tok_idx, 2] = record['word_total']
            cur_word[tok_idx, 3] = record['word_id']

        if success:
            utt = scores[utt_id]
            cur_utt[0] = float(utt['accuracy'])
            cur_utt[1] = float(utt['completeness'])
            cur_utt[2] = float(utt['fluency'])
            cur_utt[3] = float(utt['prosodic'])
            cur_utt[4] = float(utt['total'])
            feat_rows.append(cur_feat)
            phn_rows.append(cur_phn)
            word_rows.append(cur_word)
            utt_rows.append(cur_utt)
            report['kept'] += 1
            continue

    if not feat_rows:
        raise ValueError('No utterances were kept for this split.')

    feat = np.stack(feat_rows).astype(np.float32)
    phn_label = np.stack(phn_rows).astype(np.float32)
    word_label = np.stack(word_rows).astype(np.float32)
    utt_label = np.stack(utt_rows).astype(np.float32)
    return feat, phn_label, word_label, utt_label, report


def scalar_train_norm(feat, phn_label):
    valid_mask = phn_label[:, :, 0] >= 0
    valid_feat = feat[valid_mask]
    if valid_feat.size == 0:
        raise ValueError('No valid features were generated for the training split.')
    return float(valid_feat.mean()), float(valid_feat.std() + EPS)


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
    seq_len = infer_seq_len(scores, utt_ids, args.max_seq_len)

    all_phones = []
    for utt_id in utt_ids:
        all_phones.extend(record['phone'] for record in build_reference_records(scores[utt_id]))
    phn_dict = {}
    for phone in all_phones:
        if phone not in phn_dict:
            phn_dict[phone] = len(phn_dict)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(args.aligner_model)
    model = load_frame_model(args.aligner_model).to(device).eval()

    phone_to_id, id2label, sil_ids = build_model_phone_map(model)
    feat_dim = int(model.config.num_labels) + 4

    tr_feat, tr_phn, tr_word, tr_utt, tr_report = build_seq_arrays(
        train_items, scores, seq_len, feat_dim, phone_to_id, sil_ids,
        processor, model, args.sample_rate, device, args.min_sil_frames, phn_dict,
    )
    te_feat, te_phn, te_word, te_utt, te_report = build_seq_arrays(
        test_items, scores, seq_len, feat_dim, phone_to_id, sil_ids,
        processor, model, args.sample_rate, device, args.min_sil_frames, phn_dict,
    )

    norm_mean, norm_std = scalar_train_norm(tr_feat, tr_phn)

    np.save(output_dir / 'tr_feat.npy', tr_feat)
    np.save(output_dir / 'tr_label_phn.npy', tr_phn)
    np.save(output_dir / 'tr_label_word.npy', tr_word)
    np.save(output_dir / 'tr_label_utt.npy', tr_utt)
    np.save(output_dir / 'te_feat.npy', te_feat)
    np.save(output_dir / 'te_label_phn.npy', te_phn)
    np.save(output_dir / 'te_label_word.npy', te_word)
    np.save(output_dir / 'te_label_utt.npy', te_utt)

    metadata = {
        'aligner_model': args.aligner_model,
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'seq_len': seq_len,
        'feat_dim': feat_dim,
        'phn_dict': phn_dict,
        'phn_num': len(phn_dict) + 1,
        'train_norm_mean': norm_mean,
        'train_norm_std': norm_std,
        'num_frame_labels': int(model.config.num_labels),
        'frame_id2label': {str(key): str(value) for key, value in id2label.items()},
        'silence_ids': sil_ids,
        'train_report': tr_report,
        'test_report': te_report,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
