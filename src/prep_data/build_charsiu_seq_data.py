import argparse
import importlib.util
import json
import math
import os
import re
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm


EPS = 1e-8


def get_args():
    parser = argparse.ArgumentParser(description='Build GOPT sequence data from Charsiu frame logits without Kaldi.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json', help='SpeechOcean phone/word/utt labels.')
    parser.add_argument('--train-scp', type=str, default=None, help='Optional explicit train wav.scp path.')
    parser.add_argument('--test-scp', type=str, default=None, help='Optional explicit test wav.scp path.')
    parser.add_argument('--output-dir', type=str, default='data/seq_data_charsiu_tiny', help='Directory to save sequence npy files.')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms', help='HF model used for frame classification.')
    parser.add_argument('--charsiu-src-dir', type=str, default=os.environ.get('CHARSIU_SRC_DIR'), help='Local official Charsiu repo root or src directory.')
    parser.add_argument('--charsiu-lang', type=str, default=os.environ.get('CHARSIU_LANG', 'en'), help='Charsiu language code, e.g. en or zh.')
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


def speaker_id_from_audio_path(audio_path):
    return Path(audio_path).parent.name.upper()


def split_items_by_speaker_holdout(items, holdout_ratio=0.1, split_seed=1337):
    if not items:
        return [], []
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError(f'holdout_ratio must be in (0, 1), got {holdout_ratio}')

    speaker_to_items = {}
    for utt_id, audio_path in items:
        speaker_id = speaker_id_from_audio_path(audio_path)
        speaker_to_items.setdefault(speaker_id, []).append((utt_id, audio_path))

    speakers = sorted(speaker_to_items.keys())
    rng = random.Random(split_seed)
    rng.shuffle(speakers)

    target_holdout_utts = max(1, int(round(len(items) * holdout_ratio)))
    holdout_speakers = []
    holdout_count = 0
    for speaker_id in speakers:
        if holdout_count >= target_holdout_utts and holdout_speakers:
            break
        holdout_speakers.append(speaker_id)
        holdout_count += len(speaker_to_items[speaker_id])

    holdout_speaker_set = set(holdout_speakers)
    remain_split = []
    holdout_split = []
    for utt_id, audio_path in items:
        if speaker_id_from_audio_path(audio_path) in holdout_speaker_set:
            holdout_split.append((utt_id, audio_path))
        else:
            remain_split.append((utt_id, audio_path))

    if not remain_split or not holdout_split:
        raise ValueError(
            f'Failed to create non-empty split. remain={len(remain_split)} holdout={len(holdout_split)} '
            f'from total={len(items)} with holdout_ratio={holdout_ratio}'
        )
    return remain_split, holdout_split


def resolve_dataset_splits(dataset_root, train_scp=None, val_scp=None, test_scp=None, val_ratio=0.1, split_seed=1337):
    train_scp = Path(train_scp) if train_scp else Path(dataset_root) / 'train' / 'wav.scp'
    test_scp = Path(test_scp) if test_scp else Path(dataset_root) / 'test' / 'wav.scp'
    if val_scp:
        val_scp = Path(val_scp)
        train_items = parse_wav_scp(train_scp, dataset_root)
        val_items = parse_wav_scp(val_scp, dataset_root)
        test_items = parse_wav_scp(test_scp, dataset_root)
        split_meta = {
            'split_strategy': 'explicit_val_scp',
            'train_scp': str(train_scp),
            'val_scp': str(val_scp),
            'test_scp': str(test_scp),
        }
    else:
        train_items = parse_wav_scp(train_scp, dataset_root)
        all_test_items = parse_wav_scp(test_scp, dataset_root)
        test_items, val_items = split_items_by_speaker_holdout(
            all_test_items,
            holdout_ratio=val_ratio,
            split_seed=split_seed,
        )
        split_meta = {
            'split_strategy': 'speaker_holdout_from_original_test',
            'train_scp': str(train_scp),
            'val_scp': None,
            'test_scp': str(test_scp),
            'val_ratio': float(val_ratio),
            'split_seed': int(split_seed),
            'train_speakers': sorted({speaker_id_from_audio_path(path) for _, path in train_items}),
            'original_test_speakers': sorted({speaker_id_from_audio_path(path) for _, path in all_test_items}),
            'val_speakers': sorted({speaker_id_from_audio_path(path) for _, path in val_items}),
            'test_speakers': sorted({speaker_id_from_audio_path(path) for _, path in test_items}),
        }
    return train_items, val_items, test_items, split_meta


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


def build_reference_text(utt_info):
    return words_to_text(build_reference_word_records(utt_info), key='display_text')


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


def resolve_charsiu_src_dir(charsiu_src_dir):
    if not charsiu_src_dir:
        return None
    src_dir = Path(charsiu_src_dir)
    if (src_dir / 'src' / 'Charsiu.py').exists():
        return src_dir / 'src'
    if (src_dir / 'Charsiu.py').exists():
        return src_dir
    raise FileNotFoundError(f'Charsiu source directory not found: {charsiu_src_dir}')


def import_charsiu_forced_aligner(charsiu_src_dir=None):
    src_dir = resolve_charsiu_src_dir(charsiu_src_dir)
    if src_dir is not None:
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    try:
        from Charsiu import charsiu_forced_aligner
    except ImportError as exc:
        raise ImportError(
            'Official Charsiu is not importable. Clone https://github.com/lingjzhu/charsiu and '
            'pass --charsiu-src-dir or export CHARSIU_SRC_DIR to its repo root.'
        ) from exc
    return charsiu_forced_aligner


def resolve_charsiu_tokenizer_dir(charsiu_src_dir=None):
    candidates = []
    for name in ['CHARSIU_TOKENIZER_EN_CMU', 'CHARSU_TOKENIZER_EN_CMU']:
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    if charsiu_src_dir:
        src_dir = Path(charsiu_src_dir)
        candidates.extend([src_dir / 'local', src_dir.parent / 'local'])
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = repo_root / 'server_assets' / 'hf_cache' / 'transformers' / 'models--charsiu--tokenizer_en_cmu'
    refs_main = cache_root / 'refs' / 'main'
    if refs_main.exists():
        try:
            revision = refs_main.read_text(encoding='utf-8').strip()
            candidates.append(cache_root / 'snapshots' / revision)
        except Exception:
            pass
    for path in candidates:
        if (path / 'vocab.json').exists():
            return path
    return None


def patch_charsiu_tokenizer_loader(charsiu_src_dir=None):
    tokenizer_dir = resolve_charsiu_tokenizer_dir(charsiu_src_dir)
    if tokenizer_dir is None:
        return None
    try:
        from transformers import Wav2Vec2CTCTokenizer
    except Exception:
        return tokenizer_dir
    if getattr(Wav2Vec2CTCTokenizer, '_custom_gopt_charsiu_tokenizer_patch', False):
        return tokenizer_dir
    original_from_pretrained = Wav2Vec2CTCTokenizer.from_pretrained

    def local_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) == 'charsiu/tokenizer_en_cmu':
            pretrained_model_name_or_path = str(tokenizer_dir)
            kwargs['local_files_only'] = True
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    Wav2Vec2CTCTokenizer.from_pretrained = classmethod(local_from_pretrained)
    Wav2Vec2CTCTokenizer._custom_gopt_charsiu_tokenizer_patch = True
    return tokenizer_dir


def patch_local_transformers_torch_load_guard(model_name):
    model_path = Path(str(model_name))
    if not model_path.exists() or not (model_path / 'pytorch_model.bin').exists():
        return False
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return False
    if getattr(modeling_utils, '_custom_gopt_local_torch_load_guard_patch', False):
        return True

    def allow_local_torch_load():
        return None

    modeling_utils.check_torch_load_is_safe = allow_local_torch_load
    import_utils.check_torch_load_is_safe = allow_local_torch_load
    modeling_utils._custom_gopt_local_torch_load_guard_patch = True
    return True


def load_official_charsiu_aligner(model_name, device, sample_rate, sil_threshold, lang='en', charsiu_src_dir=None):
    patch_charsiu_tokenizer_loader(charsiu_src_dir)
    patch_local_transformers_torch_load_guard(model_name)
    charsiu_forced_aligner = import_charsiu_forced_aligner(charsiu_src_dir)
    aligner = charsiu_forced_aligner(
        aligner=model_name,
        lang=lang,
        sampling_rate=sample_rate,
        device=device,
        sil_threshold=sil_threshold,
    )
    aligner._custom_charsiu_src_dir = charsiu_src_dir
    return aligner


def import_official_charsiu_forced_align(charsiu_src_dir=None):
    src_dir = resolve_charsiu_src_dir(charsiu_src_dir)
    if src_dir is None:
        raise ImportError('Official Charsiu source directory is required to import forced_align.')
    utils_path = src_dir / 'utils.py'
    if not utils_path.exists():
        raise FileNotFoundError(f'Charsiu utils.py not found under {src_dir}')
    module_name = 'official_charsiu_utils'
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import official Charsiu utils from {utils_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forced_align


def build_model_phone_map(model_or_charsiu):
    processor = None
    model = model_or_charsiu
    charsiu_src_dir = getattr(model_or_charsiu, '_custom_charsiu_src_dir', None)
    if hasattr(model_or_charsiu, 'charsiu_processor'):
        processor = model_or_charsiu.charsiu_processor
        model = model_or_charsiu.aligner
    elif hasattr(model_or_charsiu, 'processor') and hasattr(model_or_charsiu, 'mapping_phone2id'):
        processor = model_or_charsiu

    if processor is not None:
        id2label = None
        src_dir = resolve_charsiu_src_dir(charsiu_src_dir)
        if src_dir is not None:
            vocab_path = src_dir / 'vocab-ctc.json'
            if vocab_path.exists():
                vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
                id2label = {int(idx): str(label) for label, idx in vocab.items()}
        if id2label is None:
            num_labels = getattr(getattr(model, 'config', None), 'num_labels', None)
            if num_labels is not None:
                id2label = {int(idx): str(processor.mapping_id2phone(int(idx))) for idx in range(int(num_labels))}
            else:
                vocab = processor.processor.tokenizer.get_vocab()
                id2label = {int(idx): str(label) for label, idx in vocab.items()}
    else:
        id2label = getattr(model.config, 'id2label', {}) or {}

    phone_to_id = {}
    sil_ids = []
    for idx, label in sorted(id2label.items(), key=lambda item: int(item[0])):
        idx = int(idx)
        norm = normalize_phone(str(label))
        if not norm:
            continue
        if 'SIL' in norm or (processor is not None and str(label) == getattr(processor, 'sil', None)):
            sil_ids.append(idx)
        if norm not in phone_to_id:
            phone_to_id[norm] = idx
    return phone_to_id, id2label, sil_ids


def build_silence_keep_mask(charsiu_aligner, probs):
    sil_mask = charsiu_aligner._get_sil_mask(probs)
    keep_mask = sil_mask != charsiu_aligner.charsiu_processor.sil_idx
    if not np.any(keep_mask):
        keep_mask[:] = True
    return keep_mask


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
    if hasattr(processor, 'audio_preprocess'):
        inputs = processor.audio_preprocess(audio, sr=sample_rate)
        inputs = torch.Tensor(inputs).unsqueeze(0).to(device)
        model_inputs = {'input_values': inputs}
    else:
        model_inputs = processor(audio, sampling_rate=sample_rate, return_tensors='pt')
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    with torch.no_grad():
        logits = model(**model_inputs).logits[0].detach().cpu().numpy()
    return softmax_np(logits), len(audio) / sample_rate


def time_to_frame_bounds(start_time, end_time, frame_step, num_frames):
    start_idx = max(int(math.floor(start_time / max(frame_step, EPS))), 0)
    end_idx = max(int(math.ceil(end_time / max(frame_step, EPS))), start_idx + 1)
    end_idx = min(end_idx, num_frames)
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, num_frames)
        start_idx = max(end_idx - 1, 0)
    return start_idx, end_idx


def normalize_aligned_phones(aligned_phones):
    rows = []
    for start_time, end_time, phone in aligned_phones:
        norm_phone = normalize_phone(phone)
        if not norm_phone or norm_phone == 'SIL':
            continue
        rows.append((float(start_time), float(end_time), norm_phone))
    return rows


def align_reference_utterance(utt_id, audio_path, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict):
    if utt_id not in scores:
        return {'utt_id': utt_id, 'skip_reason': 'missing_scores'}
    if not audio_path.exists():
        return {'utt_id': utt_id, 'skip_reason': 'missing_audio', 'audio_path': str(audio_path)}

    ref_words = build_reference_word_records(scores[utt_id])
    ref_records = build_reference_records(scores[utt_id])
    if not ref_words or not ref_records:
        return {'utt_id': utt_id, 'skip_reason': 'empty_reference_text'}

    ref_phones = [record['phone'] for record in ref_records]
    for phone in ref_phones:
        if phone not in phone_to_frame_id or phone not in phn_dict:
            return {'utt_id': utt_id, 'skip_reason': f'phone_not_in_model_vocab:{phone}'}

    phone_groups = [tuple(word['phones']) for word in ref_words if word['phones']]
    if not phone_groups:
        return {'utt_id': utt_id, 'skip_reason': 'empty_reference_phones'}

    try:
        phone_ids = charsiu.charsiu_processor.get_phone_ids(phone_groups)
        forced_align = import_official_charsiu_forced_align(getattr(charsiu, '_custom_charsiu_src_dir', None))
    except Exception as exc:
        return {'utt_id': utt_id, 'skip_reason': f'charsiu_phone_ids_failed:{exc}'}

    target_phone_ids = list(phone_ids[1:-1])
    if len(target_phone_ids) != len(ref_records):
        return {
            'utt_id': utt_id,
            'skip_reason': f'charsiu_phone_id_count_mismatch:ref={len(ref_records)} ids={len(target_phone_ids)}',
        }

    probs, audio_duration = audio_logits(audio_path, charsiu.charsiu_processor, charsiu.aligner, sample_rate, device)
    keep_mask = build_silence_keep_mask(charsiu, probs)
    kept_indices = np.flatnonzero(keep_mask)
    kept_probs = probs[keep_mask]
    if kept_probs.shape[0] < len(target_phone_ids):
        return {
            'utt_id': utt_id,
            'skip_reason': f'not_enough_nonsil_frames:frames={kept_probs.shape[0]} phones={len(target_phone_ids)}',
        }
    try:
        aligned_phone_ids = np.asarray(forced_align(kept_probs, target_phone_ids), dtype=np.int32)
    except Exception as exc:
        return {'utt_id': utt_id, 'skip_reason': f'charsiu_forced_align_failed:{exc}'}

    frame_step = audio_duration / max(len(probs), 1)

    phone_segments = []
    word_end_times = {}
    for phone_idx, record in enumerate(ref_records):
        phone = record['phone']
        token_frames = np.flatnonzero(aligned_phone_ids == phone_idx)
        if token_frames.size == 0:
            return {'utt_id': utt_id, 'skip_reason': f'empty_phone_alignment:{phone_idx}:{phone}'}
        kept_segment_probs = kept_probs[token_frames]
        start_frame = int(kept_indices[token_frames[0]])
        end_frame = int(kept_indices[token_frames[-1]]) + 1
        start_time = float(start_frame * frame_step)
        end_time = float(min(audio_duration, end_frame * frame_step))
        if end_time <= start_time:
            end_time = float(min(audio_duration, start_time + frame_step))

        segment_probs = kept_segment_probs
        if segment_probs.size == 0:
            return {'utt_id': utt_id, 'skip_reason': f'empty_phone_segment:{record["word_id"]}'}
        target_id = phone_to_frame_id[phone]
        feature = segment_feature(segment_probs, target_id, frame_step)
        segment = {
            'phone': phone,
            'phone_id': int(phn_dict[phone]),
            'phone_score': float(record['phone_score']),
            'word_id': int(record['word_id']),
            'word_accuracy': float(record['word_accuracy']),
            'word_stress': float(record['word_stress']),
            'word_total': float(record['word_total']),
            'start_time': float(start_time),
            'end_time': float(end_time),
            'feature': feature.astype(np.float32),
        }
        phone_segments.append(segment)
        word_end_times[segment['word_id']] = max(word_end_times.get(segment['word_id'], 0.0), float(end_time))

    return {
        'utt_id': utt_id,
        'audio_path': str(audio_path),
        'audio_duration': float(audio_duration),
        'phones': phone_segments,
        'word_end_times': word_end_times,
        'utt_scores': {
            'accuracy': float(scores[utt_id]['accuracy']),
            'completeness': float(scores[utt_id]['completeness']),
            'fluency': float(scores[utt_id]['fluency']),
            'prosodic': float(scores[utt_id]['prosodic']),
            'total': float(scores[utt_id]['total']),
        },
    }


def build_seq_arrays(split_items, scores, seq_len, feat_dim, charsiu, sample_rate, device, phone_to_id, phn_dict):
    feat_rows = []
    phn_rows = []
    word_rows = []
    utt_rows = []
    report = {'kept': 0, 'skipped': []}

    for utt_id, audio_path in tqdm(split_items, desc='building'):
        aligned = align_reference_utterance(
            utt_id=utt_id,
            audio_path=audio_path,
            scores=scores,
            charsiu=charsiu,
            sample_rate=sample_rate,
            device=device,
            phone_to_frame_id=phone_to_id,
            phn_dict=phn_dict,
        )
        if 'skip_reason' in aligned:
            report['skipped'].append({'utt_id': utt_id, 'reason': aligned['skip_reason']})
            continue

        cur_feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
        cur_phn = np.zeros((seq_len, 2), dtype=np.float32) - 1
        cur_word = np.zeros((seq_len, 4), dtype=np.float32) - 1
        cur_utt = np.zeros((5,), dtype=np.float32)
        for tok_idx, phone in enumerate(aligned['phones']):
            cur_feat[tok_idx, :] = phone['feature']
            cur_phn[tok_idx, 0] = phone['phone_id']
            cur_phn[tok_idx, 1] = phone['phone_score']
            cur_word[tok_idx, 0] = phone['word_accuracy']
            cur_word[tok_idx, 1] = phone['word_stress']
            cur_word[tok_idx, 2] = phone['word_total']
            cur_word[tok_idx, 3] = phone['word_id']

        cur_utt[0] = aligned['utt_scores']['accuracy']
        cur_utt[1] = aligned['utt_scores']['completeness']
        cur_utt[2] = aligned['utt_scores']['fluency']
        cur_utt[3] = aligned['utt_scores']['prosodic']
        cur_utt[4] = aligned['utt_scores']['total']
        feat_rows.append(cur_feat)
        phn_rows.append(cur_phn)
        word_rows.append(cur_word)
        utt_rows.append(cur_utt)
        report['kept'] += 1

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
    charsiu = load_official_charsiu_aligner(
        model_name=args.aligner_model,
        device=device,
        sample_rate=args.sample_rate,
        sil_threshold=args.min_sil_frames,
        lang=args.charsiu_lang,
        charsiu_src_dir=args.charsiu_src_dir,
    )

    phone_to_id, id2label, sil_ids = build_model_phone_map(charsiu)
    feat_dim = int(charsiu.aligner.config.num_labels) + 4

    tr_feat, tr_phn, tr_word, tr_utt, tr_report = build_seq_arrays(
        train_items, scores, seq_len, feat_dim, charsiu, args.sample_rate, device, phone_to_id, phn_dict,
    )
    te_feat, te_phn, te_word, te_utt, te_report = build_seq_arrays(
        test_items, scores, seq_len, feat_dim, charsiu, args.sample_rate, device, phone_to_id, phn_dict,
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
        'charsiu_src_dir': args.charsiu_src_dir,
        'charsiu_lang': args.charsiu_lang,
        'num_frame_labels': int(charsiu.aligner.config.num_labels),
        'frame_id2label': {str(key): str(value) for key, value in id2label.items()},
        'silence_ids': sil_ids,
        'train_report': tr_report,
        'test_report': te_report,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
