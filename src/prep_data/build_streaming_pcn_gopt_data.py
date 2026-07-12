import argparse
import gc
import json
import math
import os
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from build_charsiu_seq_data import (
    EPS,
    build_model_phone_map,
    load_official_charsiu_aligner,
    monotonic_align,
    normalize_phone,
    normalize_word,
    resolve_dataset_splits,
)
from build_streaming_asr_gopt_data import (
    align_gold_utterance,
    build_phone_vocab,
    build_word_lexicon,
    lcs_align_words,
    select_visible_frames,
)
from build_streaming_charsiu_data import commit_schedule


EPS_TOKEN = '<eps>'
PCN_SCHEMA = 'streaming_pcn_gopt_v2_stateful'
PCN_TYPE = 'nbest_derived_phone_confusion_network'


def get_args():
    parser = argparse.ArgumentParser(
        description='Build streaming PCN/N-best GOPT data. GT is used only for offline supervision.'
    )
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json')
    parser.add_argument('--train-scp', type=str, default=None)
    parser.add_argument('--val-scp', type=str, default=None)
    parser.add_argument('--test-scp', type=str, default=None)
    parser.add_argument('--val-speaker-ratio', type=float, default=0.5)
    parser.add_argument('--split-seed', type=int, default=1337)
    parser.add_argument('--output-dir', type=str, default='data/streaming_pcn_gopt')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--charsiu-src-dir', type=str, default=os.environ.get('CHARSIU_SRC_DIR'))
    parser.add_argument('--charsiu-lang', type=str, default=os.environ.get('CHARSIU_LANG', 'en'))
    parser.add_argument('--asr-model', type=str, default='openai/whisper-base')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--chunk-sec', type=float, default=0.64)
    parser.add_argument('--right-context-sec', type=float, default=0.16)
    parser.add_argument('--min-sil-frames', type=int, default=4)
    parser.add_argument('--nbest', type=int, default=5)
    parser.add_argument('--beam-size', type=int, default=8)
    parser.add_argument('--asr-max-new-tokens', type=int, default=128)
    parser.add_argument('--asr-no-repeat-ngram-size', type=int, default=0)
    parser.add_argument('--asr-max-words', type=int, default=64, help='Only triggers strict repeat checks; length alone is not rejected.')
    parser.add_argument('--asr-max-visible-phones', type=int, default=100, help='Only triggers strict repeat checks; length alone is not rejected.')
    parser.add_argument('--asr-max-phone-ratio', type=float, default=3.0, help='Hard reject a hypothesis when ASR phones exceed this multiple of reference phones.')
    parser.add_argument('--asr-repeat-ngram-min-repeats', type=int, default=4)
    parser.add_argument('--asr-repeat-max-ngram-size', type=int, default=12)
    parser.add_argument('--asr-repeat-ngram-coverage', type=float, default=0.6)
    parser.add_argument('--asr-repeat-token-ratio', type=float, default=0.5)
    parser.add_argument('--max-seq-len', type=int, default=0, help='0 means infer from generated PCN slots.')
    parser.add_argument('--target-splits', type=str, default='train,val,test')
    parser.add_argument('--resume', action='store_true', help='Reuse per-utterance progress files already written under output-dir/progress.')
    parser.add_argument('--finalize-only', action='store_true', help='Only merge existing progress files into NPZ/manifest outputs.')
    parser.add_argument('--skip-finalize', action='store_true', help='Only write per-utterance progress files; do not build final NPZ/manifest outputs.')
    parser.add_argument('--num-shards', type=int, default=1, help='Number of utterance shards for this generation run.')
    parser.add_argument('--shard-index', type=int, default=0, help='Shard index to process, in [0, num-shards).')
    parser.add_argument('--include-slot-prosody', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def parse_target_splits(raw_value):
    valid = ['train', 'val', 'test']
    selected = [item.strip() for item in raw_value.split(',') if item.strip()]
    if not selected:
        raise ValueError('target_splits cannot be empty.')
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f'Unknown target split(s): {unknown}')
    return selected


def validate_shard_args(args):
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1.')
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('--shard-index must satisfy 0 <= shard-index < num-shards.')
    if args.finalize_only and args.skip_finalize:
        raise ValueError('--finalize-only and --skip-finalize cannot be used together.')


def progress_path(output_dir, split_name, utt_id):
    return output_dir / 'progress' / split_name / f'{utt_id}.pkl'


def write_progress_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def read_progress_record(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def make_progress_record(split_name, utterance_index, utt_id, audio_path, status, examples=None, skipped_chunks=None, skip_record=None):
    return {
        'schema': PCN_SCHEMA,
        'pcn_type': PCN_TYPE,
        'split': split_name,
        'utterance_index': int(utterance_index),
        'utt_id': utt_id,
        'audio_path': str(audio_path),
        'status': status,
        'examples': examples or [],
        'skipped_chunks': skipped_chunks or [],
        'skip_record': skip_record,
    }


def softmax_1d(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32)
    pos_inf = np.isposinf(arr)
    if np.any(pos_inf):
        out = np.zeros_like(arr, dtype=np.float64)
        out[pos_inf] = 1.0 / float(np.sum(pos_inf))
        return out.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    min_finite = np.min(arr[finite])
    arr = np.where(finite, arr, min_finite - 50.0)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    return (exp_arr / np.clip(exp_arr.sum(), EPS, None)).astype(np.float32)


def entropy_np(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(np.clip(probs, EPS, None))))


def top_margin(probs):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return 0.0, 0.0
    top = np.sort(probs)[::-1]
    top1 = float(top[0])
    top2 = float(top[1]) if top.size > 1 else 0.0
    return top1, float(top1 - top2)


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), EPS, None)
    q = q / np.clip(q.sum(), EPS, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(m, EPS, None))))
    kl_qm = np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(np.clip(m, EPS, None))))
    return float(0.5 * (kl_pm + kl_qm))


def safe_json(value):
    if isinstance(value, dict):
        return {str(key): safe_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def non_overlapping_occurrences(tokens, pattern):
    positions = []
    pattern_size = len(pattern)
    cursor = 0
    while cursor <= len(tokens) - pattern_size:
        if tokens[cursor : cursor + pattern_size] == pattern:
            positions.append(cursor)
            cursor += pattern_size
        else:
            cursor += 1
    return positions


def find_repeated_ngram(
    words,
    min_repeats,
    max_ngram_size=12,
    strict=False,
    min_coverage=0.6,
    dominant_token_ratio=0.5,
):
    if min_repeats <= 0:
        return None
    tokens = [str(word['text']) for word in words]
    if not tokens:
        return None

    for ngram_size in range(1, min(max_ngram_size, len(tokens)) + 1):
        span = ngram_size * min_repeats
        for start in range(0, len(tokens) - span + 1):
            pattern = tokens[start : start + ngram_size]
            repeats = 1
            cursor = start + ngram_size
            while tokens[cursor : cursor + ngram_size] == pattern:
                repeats += 1
                cursor += ngram_size
            if repeats >= min_repeats:
                return {
                    'start': int(start),
                    'ngram_size': int(ngram_size),
                    'repeats': int(repeats),
                    'pattern': pattern,
                    'detection': 'contiguous_cycle',
                }

    if not strict:
        return None

    token_counts = Counter(tokens)
    dominant_token, dominant_count = token_counts.most_common(1)[0]
    token_ratio = float(dominant_count) / float(len(tokens))
    if dominant_count >= max(min_repeats * 2, 8) and token_ratio >= dominant_token_ratio:
        return {
            'start': None,
            'ngram_size': 1,
            'repeats': int(dominant_count),
            'pattern': [dominant_token],
            'coverage': token_ratio,
            'detection': 'dominant_token',
        }

    max_size = min(max_ngram_size, len(tokens) // max(min_repeats, 1))
    for ngram_size in range(max_size, 1, -1):
        candidates = Counter(
            tuple(tokens[start : start + ngram_size])
            for start in range(0, len(tokens) - ngram_size + 1)
        )
        for pattern, raw_count in candidates.most_common():
            if raw_count < min_repeats:
                break
            positions = non_overlapping_occurrences(tokens, list(pattern))
            if len(positions) < min_repeats:
                continue
            coverage = float(len(positions) * ngram_size) / float(len(tokens))
            if coverage >= min_coverage:
                return {
                    'start': int(positions[0]),
                    'ngram_size': int(ngram_size),
                    'repeats': int(len(positions)),
                    'pattern': list(pattern),
                    'coverage': coverage,
                    'detection': 'high_coverage_periodicity',
                }
    return None


class PhoneMapper:
    def __init__(self, lexicon, phn_dict):
        self.lexicon = lexicon
        self.phn_dict = phn_dict
        self.cache = {}
        self.fallback_phone = 'AH' if 'AH' in phn_dict else (next(iter(phn_dict)) if phn_dict else None)
        self.g2p = None
        try:
            from g2p_en import G2p

            self.g2p = G2p()
        except Exception:
            pass

    def word_to_phones(self, word):
        word = normalize_word(word)
        if word in self.lexicon:
            return list(self.lexicon[word]), 'lexicon'
        if word in self.cache:
            return self.cache[word]
        if self.g2p is not None:
            try:
                phones = []
                for raw_phone in self.g2p(word.lower()):
                    phone = normalize_phone(raw_phone)
                    if phone and phone in self.phn_dict:
                        phones.append(phone)
                if phones:
                    result = (phones, 'g2p_en')
                    self.cache[word] = result
                    return result
            except Exception:
                self.g2p = None
        if self.fallback_phone is None:
            result = ([], 'unmapped')
        else:
            approx_len = max(1, int(math.ceil(len(re.sub(r'[^A-Z]', '', word)) / 3.0)))
            result = ([self.fallback_phone] * approx_len, 'fallback_phone')
        self.cache[word] = result
        return result

    def words_to_phone_sequence(self, words):
        phones = []
        phone_to_word = []
        source_counts = Counter()
        for word_idx, word in enumerate(words):
            cur_phones, source = self.word_to_phones(word)
            source_counts[source] += 1
            for phone in cur_phones:
                phones.append(phone)
                phone_to_word.append(word_idx)
        return phones, phone_to_word, dict(source_counts)


class WhisperNBestGenerator:
    def __init__(self, model_name, language, device, nbest, beam_size, max_new_tokens, no_repeat_ngram_size):
        self.processor = AutoProcessor.from_pretrained(model_name)
        torch_dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.language = language
        self.device = device
        self.nbest = int(nbest)
        self.beam_size = max(int(beam_size), int(nbest))
        self.max_new_tokens = int(max_new_tokens)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.forced_decoder_ids = None
        if hasattr(self.processor, 'get_decoder_prompt_ids'):
            try:
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task='transcribe',
                )
            except Exception:
                self.forced_decoder_ids = None

    @torch.inference_mode()
    def generate(self, audio, sample_rate, audio_end):
        inputs = self.processor(
            np.asarray(audio, dtype=np.float32),
            sampling_rate=sample_rate,
            return_tensors='pt',
            return_attention_mask=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        for key in ('input_features', 'input_values'):
            if key in inputs and torch.is_floating_point(inputs[key]):
                inputs[key] = inputs[key].to(dtype=self.model_dtype)
        generate_kwargs = {
            'num_beams': self.beam_size,
            'num_return_sequences': self.nbest,
            'max_new_tokens': self.max_new_tokens,
            'return_dict_in_generate': True,
            'output_scores': True,
            'do_sample': False,
        }
        if self.no_repeat_ngram_size > 0:
            generate_kwargs['no_repeat_ngram_size'] = self.no_repeat_ngram_size
        if self.forced_decoder_ids is not None:
            generate_kwargs['forced_decoder_ids'] = self.forced_decoder_ids
        output = self.model.generate(**inputs, **generate_kwargs)
        if torch.is_tensor(output):
            sequences = output
            sequence_scores = [0.0 for _ in range(int(sequences.shape[0]))]
            transition_scores = []
        else:
            sequences = output.sequences
            if getattr(output, 'sequences_scores', None) is not None:
                sequence_scores = output.sequences_scores.detach().float().cpu().numpy().tolist()
            else:
                sequence_scores = [0.0 for _ in range(int(sequences.shape[0]))]
            transition_scores = self._transition_scores(output)
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)

        rows = []
        for rank, (text, sequence_score) in enumerate(zip(texts, sequence_scores)):
            token_ids = sequences[rank].detach().cpu().tolist()
            token_logprobs = (
                transition_scores[rank]
                if rank < len(transition_scores)
                else [float(sequence_score) for _ in token_ids]
            )
            token_rows, word_rows = self._token_word_scores(token_ids, token_logprobs)
            words = [row['word'] for row in word_rows]
            length_norm = (
                float(np.mean([row['logprob'] for row in token_rows]))
                if token_rows
                else float(sequence_score)
            )
            rows.append({
                'rank': int(rank),
                'text': ' '.join(words).lower(),
                'words': words,
                'logprob': float(length_norm),
                'sequence_score': float(sequence_score),
                'length_normalized_sequence_score': float(length_norm),
                'token_ids': [int(row['token_id']) for row in token_rows],
                'token_logprobs': [float(row['logprob']) for row in token_rows],
                'token_confidences': [float(row['confidence']) for row in token_rows],
                'word_token_ranges': [[int(row['token_start']), int(row['token_end'])] for row in word_rows],
                'word_logprobs': [float(row['logprob']) for row in word_rows],
                'word_confidences': [float(row['confidence']) for row in word_rows],
                'word_timestamps': estimate_word_timestamps(words, audio_end),
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return rows

    def _transition_scores(self, output):
        if getattr(output, 'scores', None) is None:
            return []
        try:
            sequences = output.sequences.detach().cpu()
            step_scores = [score.detach().float().cpu() for score in output.scores]
            if not step_scores:
                return []
            beam_indices = getattr(output, 'beam_indices', None)
            if beam_indices is not None:
                beam_indices = beam_indices.detach().cpu()
            generated_steps = len(step_scores)
            start = max(0, int(sequences.shape[1]) - generated_steps)
            rows = []
            for seq_idx in range(int(sequences.shape[0])):
                cur = []
                for step, logits in enumerate(step_scores):
                    token_pos = start + step
                    if token_pos >= int(sequences.shape[1]):
                        break
                    token_id = int(sequences[seq_idx, token_pos])
                    if token_id < 0 or token_id >= int(logits.shape[-1]):
                        cur.append(0.0)
                        continue
                    score_row = min(seq_idx, int(logits.shape[0]) - 1)
                    if beam_indices is not None and seq_idx < int(beam_indices.shape[0]) and token_pos < int(beam_indices.shape[1]):
                        candidate = int(beam_indices[seq_idx, token_pos])
                        if 0 <= candidate < int(logits.shape[0]):
                            score_row = candidate
                    log_probs = torch.log_softmax(logits[score_row], dim=-1)
                    cur.append(float(log_probs[token_id].item()))
                rows.append(cur)
            return rows
        except Exception:
            return []

    def _token_word_scores(self, token_ids, token_logprobs):
        tokenizer = self.processor.tokenizer
        special_ids = set(getattr(tokenizer, 'all_special_ids', []) or [])
        if len(token_logprobs) != len(token_ids):
            aligned = [0.0 for _ in token_ids]
            offset = max(0, len(token_ids) - len(token_logprobs))
            for idx, value in enumerate(token_logprobs[-len(token_ids) :]):
                pos = offset + idx
                if pos < len(aligned):
                    aligned[pos] = float(value)
            token_logprobs = aligned
        token_rows = []
        words = []
        current = []
        current_token_ids = []
        current_logprobs = []
        current_start = None

        def flush():
            nonlocal current, current_token_ids, current_logprobs, current_start
            word = normalize_word(''.join(current))
            if word:
                avg_logprob = float(np.mean(current_logprobs)) if current_logprobs else 0.0
                words.append({
                    'word': word,
                    'token_start': int(current_start if current_start is not None else 0),
                    'token_end': int((current_token_ids[-1] + 1) if current_token_ids else 0),
                    'logprob': avg_logprob,
                    'confidence': float(np.clip(np.exp(avg_logprob), 0.0, 1.0)),
                })
            current = []
            current_token_ids = []
            current_logprobs = []
            current_start = None

        for pos, token_id in enumerate(token_ids):
            if int(token_id) in special_ids:
                continue
            logprob = float(token_logprobs[pos]) if pos < len(token_logprobs) else 0.0
            try:
                text = tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except TypeError:
                text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
            if not text:
                continue
            token_rows.append({
                'token_id': int(token_id),
                'logprob': logprob,
                'confidence': float(np.clip(np.exp(logprob), 0.0, 1.0)),
            })
            for char in text:
                if char.isalpha() or char == "'":
                    if current_start is None:
                        current_start = pos
                    current.append(char)
                    current_token_ids.append(pos)
                    current_logprobs.append(logprob)
                else:
                    flush()
        flush()
        return token_rows, words


def normalize_text_to_words(text):
    tokens = []
    for token in re.findall(r"[A-Za-z']+", text):
        norm = normalize_word(token)
        if norm:
            tokens.append(norm)
    return tokens


def estimate_word_timestamps(words, audio_end):
    if not words:
        return []
    step = float(audio_end) / max(len(words), 1)
    rows = []
    for idx, word in enumerate(words):
        rows.append({
            'word': word.lower(),
            'start': float(idx * step),
            'end': float(min(audio_end, (idx + 1) * step)),
            'source': 'duration_proportional_estimate',
        })
    return rows


def hypothesis_phone_rows(row, phone_mapper, phn_dict):
    phones = []
    phone_ids = []
    phone_to_word = []
    phone_confidences = []
    source_counts = Counter()
    word_confidences = row.get('word_confidences') or [1.0 for _ in row.get('words', [])]
    for word_idx, word in enumerate(row.get('words', [])):
        cur_phones, source = phone_mapper.word_to_phones(word)
        source_counts[source] += 1
        word_conf = float(word_confidences[word_idx]) if word_idx < len(word_confidences) else 1.0
        word_conf = float(np.clip(word_conf, 0.0, 1.0))
        for phone in cur_phones:
            if phone not in phn_dict:
                continue
            phones.append(phone)
            phone_ids.append(int(phn_dict[phone]))
            phone_to_word.append(int(word_idx))
            phone_confidences.append(word_conf)
    return {
        'phones': phones,
        'phone_ids': phone_ids,
        'phone_to_word': phone_to_word,
        'phone_confidences': phone_confidences,
        'source_counts': dict(source_counts),
    }


def align_hypothesis_with_charsiu(item, row, phone_mapper, phn_dict, phone_to_frame_id, audio_end):
    hyp = hypothesis_phone_rows(row, phone_mapper, phn_dict)
    words = row.get('words', [])
    if not words or not hyp['phone_ids']:
        row['word_timestamps'] = []
        row['timestamp_source'] = 'empty_hypothesis'
        row['phone_times'] = []
        row['phone_acoustic_supports'] = []
        return row

    kept_indices, kept_probs = select_visible_frames(item['probs'], item['keep_mask'], audio_end, item['frame_step'])
    frame_phone_ids = []
    id_to_phone = {idx: phone for phone, idx in phn_dict.items()}
    try:
        for phone_idx in hyp['phone_ids']:
            phone = id_to_phone[int(phone_idx)]
            frame_phone_ids.append(phone_to_frame_id[phone])
        if kept_probs.shape[0] < len(frame_phone_ids):
            raise ValueError('not_enough_visible_frames_for_hypothesis')
        path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), frame_phone_ids)
        phone_times = []
        phone_acoustic_supports = []
        for phone_pos, frame_phone_id in enumerate(frame_phone_ids):
            local_frames = kept_indices[path == phone_pos]
            if local_frames.size == 0:
                raise ValueError('empty_aligned_phone_span')
            start_time = float(local_frames.min() * item['frame_step'])
            end_time = float((local_frames.max() + 1) * item['frame_step'])
            support = float(np.mean(item['probs'][local_frames, frame_phone_id]))
            phone_times.append((start_time, min(float(audio_end), end_time)))
            phone_acoustic_supports.append(float(np.clip(support, 0.0, 1.0)))
        word_rows = []
        for word_idx, word in enumerate(words):
            positions = [idx for idx, cur_word in enumerate(hyp['phone_to_word']) if cur_word == word_idx]
            if not positions:
                continue
            start = min(phone_times[idx][0] for idx in positions)
            end = max(phone_times[idx][1] for idx in positions)
            word_rows.append({
                'word': word,
                'start': float(start),
                'end': float(end),
                'source': 'charsiu_hypothesis_alignment',
                'word_logprob': float(row.get('word_logprobs', [0.0] * len(words))[word_idx])
                if word_idx < len(row.get('word_logprobs', []))
                else 0.0,
                'word_confidence': float(row.get('word_confidences', [1.0] * len(words))[word_idx])
                if word_idx < len(row.get('word_confidences', []))
                else 1.0,
            })
        row['word_timestamps'] = word_rows
        row['timestamp_source'] = 'charsiu_hypothesis_alignment'
        row['phone_times'] = phone_times
        row['phone_acoustic_supports'] = phone_acoustic_supports
    except Exception as exc:
        row['word_timestamps'] = estimate_word_timestamps(words, audio_end)
        row['timestamp_source'] = 'duration_proportional_fallback'
        row['timestamp_fallback_reason'] = repr(exc)
        phone_times = []
        phone_acoustic_supports = []
        for phone_idx, word_idx in enumerate(hyp['phone_to_word']):
            if word_idx < len(row['word_timestamps']):
                word_time = row['word_timestamps'][word_idx]
                phone_times.append((float(word_time['start']), float(word_time['end'])))
            else:
                phone_times.append((0.0, float(audio_end)))
            phone_acoustic_supports.append(0.0)
        row['phone_times'] = phone_times
        row['phone_acoustic_supports'] = phone_acoustic_supports
    row.update(hyp)
    return row


def interval_overlap_ratio(a, b):
    if a is None or b is None:
        return 0.0
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))
    union = max(float(a1), float(b1)) - min(float(a0), float(b0))
    if union <= 0:
        return 0.0
    return inter / union


def align_token_sequences(
    ref_tokens,
    hyp_tokens,
    mismatch_cost=1.0,
    ref_times=None,
    hyp_times=None,
    hyp_confidences=None,
    time_overlap_reward=0.0,
):
    n = len(ref_tokens)
    m = len(hyp_tokens)
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    back = {}
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + 1.0
        back[(i, 0)] = (i - 1, 0, 'delete')
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + 1.0
        back[(0, j)] = (0, j - 1, 'insert')
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0.0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else mismatch_cost
            if ref_times is not None and hyp_times is not None:
                sub_cost -= float(time_overlap_reward) * interval_overlap_ratio(ref_times[i - 1], hyp_times[j - 1])
            if hyp_confidences is not None:
                sub_cost -= 0.1 * float(np.clip(hyp_confidences[j - 1], 0.0, 1.0))
            sub_cost = max(0.0, sub_cost)
            candidates = [
                (dp[i - 1, j - 1] + sub_cost, i - 1, j - 1, 'match'),
                (dp[i - 1, j] + 1.0, i - 1, j, 'delete'),
                (dp[i, j - 1] + 1.0, i, j - 1, 'insert'),
            ]
            cost, prev_i, prev_j, op = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost
            back[(i, j)] = (prev_i, prev_j, op)

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        prev_i, prev_j, op = back[(i, j)]
        if op == 'match':
            pairs.append((i - 1, j - 1))
        elif op == 'delete':
            pairs.append((i - 1, None))
        else:
            pairs.append((None, j - 1))
        i, j = prev_i, prev_j
    pairs.reverse()
    return pairs


def slot_top_phone(slot_counts, eps_index):
    best_idx = None
    best_mass = -1.0
    for phone_idx, mass in slot_counts.items():
        if phone_idx == eps_index:
            continue
        if mass > best_mass:
            best_idx = phone_idx
            best_mass = mass
    return best_idx


def slot_time(slot):
    mass = max(float(slot.get('time_mass', 0.0)), EPS)
    if mass <= EPS:
        return None
    return float(slot.get('start_sum', 0.0) / mass), float(slot.get('end_sum', 0.0) / mass)


def add_slot_phone_mass(slot, phone_idx, phone_mass, eps_index, eps_mass=0.0, phone_time=None, confidence=1.0):
    if phone_mass > 0:
        slot['counts'][int(phone_idx)] += float(phone_mass)
    if eps_mass > 0:
        slot['counts'][eps_index] += float(eps_mass)
    if phone_time is not None and phone_mass > 0:
        slot['start_sum'] = slot.get('start_sum', 0.0) + float(phone_time[0]) * float(phone_mass)
        slot['end_sum'] = slot.get('end_sum', 0.0) + float(phone_time[1]) * float(phone_mass)
        slot['time_mass'] = slot.get('time_mass', 0.0) + float(phone_mass)
    slot['confidence_sum'] = slot.get('confidence_sum', 0.0) + float(confidence) * max(float(phone_mass), 0.0)
    slot['confidence_mass'] = slot.get('confidence_mass', 0.0) + max(float(phone_mass), 0.0)


def insert_slot(slots, insert_at, eps_index, prior_mass):
    slot = {'counts': Counter(), 'start_sum': 0.0, 'end_sum': 0.0, 'time_mass': 0.0}
    if prior_mass > 0:
        slot['counts'][eps_index] += float(prior_mass)
    slots.insert(insert_at, slot)
    return slot


def build_pcn_from_hypotheses(hypotheses, phone_mapper, phn_dict):
    eps_index = len(phn_dict)
    phone_dim = len(phn_dict) + 1
    hyp_logprobs = [float(row.get('length_normalized_sequence_score', row.get('logprob', 0.0))) for row in hypotheses]
    hyp_weights = softmax_1d(hyp_logprobs)
    id_to_phone = {idx: phone for phone, idx in phn_dict.items()}

    hyp_phone_rows = []
    for row, weight in zip(hypotheses, hyp_weights):
        if 'phone_ids' not in row:
            row = align_hypothesis_with_charsiu(
                item={'probs': np.zeros((0, phone_dim), dtype=np.float32), 'keep_mask': np.zeros((0,), dtype=np.int32), 'frame_step': 0.02},
                row=row,
                phone_mapper=phone_mapper,
                phn_dict=phn_dict,
                phone_to_frame_id={},
                audio_end=0.0,
            )
        phone_ids = [int(phone_idx) for phone_idx in row.get('phone_ids', [])]
        phone_confidences = [float(x) for x in row.get('phone_confidences', [1.0] * len(phone_ids))]
        phone_acoustic = [float(x) for x in row.get('phone_acoustic_supports', [0.0] * len(phone_ids))]
        phone_times = row.get('phone_times', [None] * len(phone_ids))
        hyp_phone_rows.append({
            'phones': row.get('phones', []),
            'phone_ids': phone_ids,
            'phone_to_word': list(row.get('phone_to_word', [])),
            'phone_confidences': phone_confidences,
            'phone_acoustic_supports': phone_acoustic,
            'phone_times': phone_times,
            'weight': float(weight),
            'source_counts': row.get('source_counts', {}),
        })

    slots = []
    total_mass = 0.0
    for hyp_idx, hyp in enumerate(hyp_phone_rows):
        phone_ids = hyp['phone_ids']
        weight = float(hyp['weight'])
        if hyp_idx == 0:
            for local_idx, phone_idx in enumerate(phone_ids):
                slot = {'counts': Counter(), 'start_sum': 0.0, 'end_sum': 0.0, 'time_mass': 0.0}
                quality = float(np.clip(
                    0.5 * hyp['phone_confidences'][local_idx] + 0.5 * hyp['phone_acoustic_supports'][local_idx],
                    0.0,
                    1.0,
                ))
                add_slot_phone_mass(
                    slot,
                    phone_idx,
                    weight * quality,
                    eps_index,
                    eps_mass=weight * (1.0 - quality),
                    phone_time=hyp['phone_times'][local_idx] if local_idx < len(hyp['phone_times']) else None,
                    confidence=hyp['phone_confidences'][local_idx],
                )
                slots.append(slot)
            total_mass += weight
            continue

        consensus_slot_ids = []
        consensus_phone_ids = []
        for slot_idx, slot in enumerate(slots):
            top_idx = slot_top_phone(slot['counts'], eps_index)
            if top_idx is not None:
                consensus_slot_ids.append(slot_idx)
                consensus_phone_ids.append(top_idx)
        consensus_times = [slot_time(slots[slot_idx]) for slot_idx in consensus_slot_ids]
        pairs = align_token_sequences(
            consensus_phone_ids,
            phone_ids,
            ref_times=consensus_times,
            hyp_times=hyp.get('phone_times'),
            hyp_confidences=hyp.get('phone_confidences'),
            time_overlap_reward=0.3,
        )
        cursor = 0
        matched_slots = set()
        for ref_pos, hyp_pos in pairs:
            if ref_pos is None:
                insert_at = consensus_slot_ids[cursor] if cursor < len(consensus_slot_ids) else len(slots)
                slot = insert_slot(slots, insert_at, eps_index, total_mass)
                quality = float(np.clip(
                    0.5 * hyp['phone_confidences'][hyp_pos] + 0.5 * hyp['phone_acoustic_supports'][hyp_pos],
                    0.0,
                    1.0,
                ))
                add_slot_phone_mass(
                    slot,
                    phone_ids[hyp_pos],
                    weight * quality,
                    eps_index,
                    eps_mass=weight * (1.0 - quality),
                    phone_time=hyp['phone_times'][hyp_pos] if hyp_pos < len(hyp['phone_times']) else None,
                    confidence=hyp['phone_confidences'][hyp_pos],
                )
                cursor += 1
            else:
                slot_idx = consensus_slot_ids[ref_pos]
                matched_slots.add(slot_idx)
                cursor = ref_pos + 1
                if hyp_pos is None:
                    slots[slot_idx]['counts'][eps_index] += weight
                else:
                    quality = float(np.clip(
                        0.5 * hyp['phone_confidences'][hyp_pos] + 0.5 * hyp['phone_acoustic_supports'][hyp_pos],
                        0.0,
                        1.0,
                    ))
                    add_slot_phone_mass(
                        slots[slot_idx],
                        phone_ids[hyp_pos],
                        weight * quality,
                        eps_index,
                        eps_mass=weight * (1.0 - quality),
                        phone_time=hyp['phone_times'][hyp_pos] if hyp_pos < len(hyp['phone_times']) else None,
                        confidence=hyp['phone_confidences'][hyp_pos],
                    )
        for slot_idx, slot in enumerate(slots):
            if slot_idx not in matched_slots and sum(slot['counts'].values()) < total_mass + weight - EPS:
                if slot_top_phone(slot['counts'], eps_index) is not None:
                    slot['counts'][eps_index] += weight
        total_mass += weight

    if not slots:
        slot = {'counts': Counter()}
        slot['counts'][eps_index] = 1.0
        slots = [slot]
        total_mass = 1.0

    cn_post = np.zeros((len(slots), phone_dim), dtype=np.float32)
    top_phone_ids = []
    slot_times = []
    slot_confidences = []
    for slot_idx, slot in enumerate(slots):
        slot_mass = sum(slot['counts'].values())
        if slot_mass < total_mass:
            slot['counts'][eps_index] += total_mass - slot_mass
        for phone_idx, mass in slot['counts'].items():
            cn_post[slot_idx, int(phone_idx)] = float(mass) / max(float(total_mass), EPS)
        cn_post[slot_idx] /= np.clip(cn_post[slot_idx].sum(), EPS, None)
        top_phone_ids.append(slot_top_phone(slot['counts'], eps_index))
        slot_times.append(slot_time(slot))
        conf_mass = max(float(slot.get('confidence_mass', 0.0)), EPS)
        slot_confidences.append(float(np.clip(slot.get('confidence_sum', 0.0) / conf_mass, 0.0, 1.0)))

    return {
        'cn_post': cn_post,
        'top_phone_ids': top_phone_ids,
        'slot_times': slot_times,
        'slot_confidences': slot_confidences,
        'hyp_phone_rows': hyp_phone_rows,
        'hyp_weights': hyp_weights.tolist(),
        'id_to_phone': id_to_phone,
        'eps_index': eps_index,
    }


def filter_looping_hypotheses(hypotheses, phone_mapper, gold_words, args):
    reference_phone_count = sum(len(word['phones']) for word in gold_words)
    kept = []
    filtered = []
    for row in hypotheses:
        phones, _, source_counts = phone_mapper.words_to_phone_sequence(row['words'])
        asr_phone_count = len(phones)
        phone_ratio = float(asr_phone_count / max(reference_phone_count, 1))
        long_word_trigger = args.asr_max_words > 0 and len(row['words']) > args.asr_max_words
        long_phone_trigger = args.asr_max_visible_phones > 0 and asr_phone_count > args.asr_max_visible_phones
        strict_repeat_check = bool(long_word_trigger or long_phone_trigger)
        repeat_info = find_repeated_ngram(
            [{'text': word} for word in row['words']],
            min_repeats=args.asr_repeat_ngram_min_repeats,
            max_ngram_size=args.asr_repeat_max_ngram_size,
            strict=strict_repeat_check,
            min_coverage=args.asr_repeat_ngram_coverage,
            dominant_token_ratio=args.asr_repeat_token_ratio,
        )
        diagnostics = {
            'rank': int(row.get('rank', -1)),
            'word_count': int(len(row['words'])),
            'asr_phone_count': int(asr_phone_count),
            'reference_phone_count': int(reference_phone_count),
            'phone_ratio': phone_ratio,
            'long_word_trigger': bool(long_word_trigger),
            'long_phone_trigger': bool(long_phone_trigger),
            'strict_repeat_check': bool(strict_repeat_check),
            'phone_source_counts': source_counts,
        }
        if repeat_info is not None:
            filtered.append({
                **diagnostics,
                'reason': 'repetitive_asr_hypothesis',
                'repeat_info': repeat_info,
                'text': row.get('text', ''),
            })
            continue
        if args.asr_max_phone_ratio > 0 and phone_ratio > args.asr_max_phone_ratio:
            filtered.append({
                **diagnostics,
                'reason': 'asr_phone_ratio_outlier',
                'text': row.get('text', ''),
            })
            continue
        row = dict(row)
        row['asr_phone_count'] = int(asr_phone_count)
        row['phone_ratio'] = phone_ratio
        row['phone_source_counts'] = source_counts
        kept.append(row)
    if not kept:
        kept = []
    return kept, filtered


def top_hyp_word_ids_for_slots(pcn):
    slot_word_ids = np.zeros((len(pcn['top_phone_ids']),), dtype=np.int32) - 1
    if not pcn['hyp_phone_rows']:
        return slot_word_ids
    top_hyp = pcn['hyp_phone_rows'][0]
    hyp_phone_ids = list(top_hyp['phone_ids'])
    hyp_phone_to_word = list(top_hyp['phone_to_word'])
    consensus_slot_ids = []
    consensus_phone_ids = []
    for slot_idx, phone_idx in enumerate(pcn['top_phone_ids']):
        if phone_idx is not None:
            consensus_slot_ids.append(slot_idx)
            consensus_phone_ids.append(phone_idx)
    pairs = align_token_sequences(consensus_phone_ids, hyp_phone_ids)
    for ref_pos, hyp_pos in pairs:
        if ref_pos is None or hyp_pos is None:
            continue
        slot_idx = consensus_slot_ids[ref_pos]
        if hyp_pos < len(hyp_phone_to_word):
            slot_word_ids[slot_idx] = int(hyp_phone_to_word[hyp_pos])
    return slot_word_ids


def pcn_stats(cn_post, top_phone_ids, prev_top_phone_ids, eps_index):
    stats = np.zeros((cn_post.shape[0], 5), dtype=np.float32)
    stable_prefix = 0
    for idx, phone_idx in enumerate(top_phone_ids):
        if idx < len(prev_top_phone_ids) and phone_idx == prev_top_phone_ids[idx]:
            stable_prefix += 1
        else:
            break
    prefix_stability = float(stable_prefix) / float(max(len(top_phone_ids), 1))
    for idx, probs in enumerate(cn_post):
        non_eps = np.delete(probs, eps_index)
        top1, margin = top_margin(non_eps)
        stats[idx] = np.array(
            [
                float(probs[eps_index]),
                entropy_np(probs),
                top1,
                margin,
                prefix_stability,
            ],
            dtype=np.float32,
        )
    return stats, prefix_stability


def validate_pcn(pcn):
    cn_post = np.asarray(pcn['cn_post'], dtype=np.float32)
    if not np.isfinite(cn_post).all():
        raise ValueError('pcn_contains_nan_or_inf')
    sums = cn_post.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        raise ValueError(f'pcn_posterior_not_normalized:max_abs_err={float(np.max(np.abs(sums - 1.0)))}')
    eps_index = int(pcn['eps_index'])
    if eps_index < 0 or eps_index >= cn_post.shape[-1]:
        raise ValueError('invalid_epsilon_index')
    eps_probs = cn_post[:, eps_index]
    if np.any(eps_probs < -1e-6) or np.any(eps_probs > 1.0 + 1e-6):
        raise ValueError('epsilon_probability_out_of_range')
    for row in pcn.get('hyp_phone_rows', []):
        for value in row.get('phone_confidences', []):
            if not np.isfinite(value) or value < -1e-6 or value > 1.0 + 1e-6:
                raise ValueError('phone_confidence_out_of_range')
    for value in pcn.get('slot_confidences', []):
        if not np.isfinite(value) or value < -1e-6 or value > 1.0 + 1e-6:
            raise ValueError('slot_confidence_out_of_range')


def cumulative_candidate_mask_from_asr(pcn_word_id, word_timestamps, commit_time, audio_end, is_final):
    mask = np.zeros((pcn_word_id.shape[0],), dtype=np.float32)
    committed_words = set()
    for word_idx, word_time in enumerate(word_timestamps or []):
        end_time = float(word_time.get('end', 0.0))
        if end_time <= commit_time + 1e-6 or (is_final and end_time <= audio_end + 1e-6):
            committed_words.add(int(word_idx))
    for slot_idx, word_idx in enumerate(pcn_word_id.tolist()):
        if int(word_idx) in committed_words:
            mask[slot_idx] = 1.0
    return mask


def align_previous_commits(prev_state, current_top_phone_ids):
    if prev_state is None:
        return {}, [], []
    prev_top_phone_ids = prev_state.get('top_phone_ids', [])
    prev_committed = set(int(x) for x in prev_state.get('committed_slots', []))
    pairs = align_token_sequences(prev_top_phone_ids, current_top_phone_ids, mismatch_cost=1.0)
    prev_to_current = {}
    for prev_idx, cur_idx in pairs:
        if prev_idx is None or cur_idx is None:
            continue
        if prev_idx < len(prev_top_phone_ids) and cur_idx < len(current_top_phone_ids):
            if prev_top_phone_ids[prev_idx] == current_top_phone_ids[cur_idx]:
                prev_to_current[int(prev_idx)] = int(cur_idx)
    mapped_old = sorted(prev_to_current[idx] for idx in prev_committed if idx in prev_to_current)
    dropped = sorted(idx for idx in prev_committed if idx not in prev_to_current)
    return prev_to_current, mapped_old, dropped


def build_stateful_commit_masks(prev_state, pcn, pcn_word_id, word_timestamps, commit_time, audio_end, is_final):
    candidate = cumulative_candidate_mask_from_asr(pcn_word_id, word_timestamps, commit_time, audio_end, is_final)
    prev_to_current, mapped_old_slots, dropped_or_revised = align_previous_commits(prev_state, pcn['top_phone_ids'])
    current_to_prev = np.zeros((len(pcn['top_phone_ids']),), dtype=np.int32) - 1
    for prev_idx, cur_idx in prev_to_current.items():
        if 0 <= cur_idx < current_to_prev.shape[0]:
            current_to_prev[cur_idx] = int(prev_idx)
    mapped_old_set = set(mapped_old_slots)
    cumulative = candidate.copy()
    for slot_idx in mapped_old_set:
        if 0 <= slot_idx < cumulative.shape[0]:
            cumulative[slot_idx] = 1.0

    new_mask = np.zeros_like(cumulative)
    for word_idx in sorted(set(int(x) for x in pcn_word_id.tolist() if int(x) >= 0)):
        positions = np.flatnonzero(pcn_word_id == word_idx)
        if positions.size == 0:
            continue
        word_committed = bool(np.all(candidate[positions] > 0))
        if not word_committed:
            continue
        if any(int(pos) in mapped_old_set for pos in positions.tolist()):
            continue
        new_mask[positions] = 1.0

    committed_slots = sorted(np.flatnonzero(cumulative > 0).astype(int).tolist())
    new_slots = sorted(np.flatnonzero(new_mask > 0).astype(int).tolist())
    diagnostics = {
        'mapped_old_slots': mapped_old_slots,
        'new_slots': new_slots,
        'dropped_or_revised_slots': dropped_or_revised,
    }
    next_state = {
        'top_phone_ids': list(pcn['top_phone_ids']),
        'committed_slots': committed_slots,
    }
    return cumulative.astype(np.float32), new_mask.astype(np.float32), current_to_prev, diagnostics, next_state


def acoustic_distribution_for_frames(frame_probs, phone_to_frame_id, phn_dict, phone_dim):
    out = np.zeros((phone_dim,), dtype=np.float32)
    if frame_probs.size == 0:
        return out
    mean_probs = frame_probs.mean(axis=0)
    for phone, phone_idx in phn_dict.items():
        frame_idx = phone_to_frame_id.get(phone)
        if frame_idx is not None and frame_idx < mean_probs.shape[0]:
            out[int(phone_idx)] = float(mean_probs[frame_idx])
    if out.sum() > 0:
        out /= out.sum()
    return out


def build_acoustic_evidence(item, cn_post, top_phone_ids, phone_to_frame_id, phn_dict, audio_end):
    phone_dim = cn_post.shape[-1]
    acoustic_post = np.zeros_like(cn_post, dtype=np.float32)
    acoustic_stats = np.zeros((cn_post.shape[0], 4), dtype=np.float32)
    kept_indices, kept_probs = select_visible_frames(item['probs'], item['keep_mask'], audio_end, item['frame_step'])
    alignable = [
        (slot_idx, phone_idx)
        for slot_idx, phone_idx in enumerate(top_phone_ids)
        if phone_idx is not None
    ]
    if kept_probs.shape[0] < len(alignable) or not alignable:
        for slot_idx in range(cn_post.shape[0]):
            acoustic_stats[slot_idx, 3] = js_divergence(cn_post[slot_idx], acoustic_post[slot_idx])
        return acoustic_post, acoustic_stats, 0

    id_to_phone = {idx: phone for phone, idx in phn_dict.items()}
    frame_phone_ids = [phone_to_frame_id[id_to_phone[phone_idx]] for _, phone_idx in alignable]
    try:
        path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), frame_phone_ids)
    except Exception:
        for slot_idx in range(cn_post.shape[0]):
            acoustic_stats[slot_idx, 3] = js_divergence(cn_post[slot_idx], acoustic_post[slot_idx])
        return acoustic_post, acoustic_stats, 0

    for local_idx, (slot_idx, _) in enumerate(alignable):
        tok_frames = kept_indices[path == local_idx]
        if tok_frames.size == 0:
            acoustic_stats[slot_idx, 3] = js_divergence(cn_post[slot_idx], acoustic_post[slot_idx])
            continue
        dist = acoustic_distribution_for_frames(
            item['probs'][tok_frames],
            phone_to_frame_id,
            phn_dict,
            phone_dim,
        )
        acoustic_post[slot_idx] = dist
        non_eps = dist[:-1]
        _, margin = top_margin(non_eps)
        duration = float(tok_frames.size * item['frame_step'])
        acoustic_stats[slot_idx] = np.array(
            [
                entropy_np(dist),
                margin,
                duration,
                js_divergence(cn_post[slot_idx], dist),
            ],
            dtype=np.float32,
        )
    return acoustic_post, acoustic_stats, int(kept_probs.shape[0])


def compute_prosody(audio, sample_rate, audio_end, word_count, phone_count):
    audio = np.asarray(audio, dtype=np.float32)
    duration = float(max(audio_end, EPS))
    if audio.size == 0:
        return np.zeros((14,), dtype=np.float32)
    hop_length = max(int(0.01 * sample_rate), 1)
    frame_length = max(int(0.025 * sample_rate), hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    log_energy = np.log(np.clip(rms, EPS, None))
    energy_threshold = np.percentile(rms, 20) if rms.size else 0.0
    silence = rms <= max(energy_threshold, EPS)
    silence_ratio = float(np.mean(silence)) if silence.size else 0.0
    pause_count, longest_pause = count_pauses(silence, hop_length / sample_rate)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced = np.isfinite(f0)
        f0_values = f0[voiced]
        voiced_prob_mean = float(np.nanmean(voiced_prob)) if voiced_prob is not None else float(np.mean(voiced))
    except Exception:
        f0_values = np.array([], dtype=np.float32)
        voiced = np.zeros_like(rms, dtype=bool)
        voiced_prob_mean = 0.0
    f0_mean = float(np.mean(f0_values)) if f0_values.size else 0.0
    f0_std = float(np.std(f0_values)) if f0_values.size else 0.0
    f0_slope = slope(f0_values)
    energy_mean = float(np.mean(log_energy)) if log_energy.size else 0.0
    energy_std = float(np.std(log_energy)) if log_energy.size else 0.0
    energy_slope = slope(log_energy)
    voiced_duration = max(float(np.mean(voiced)) * duration, EPS) if voiced.size else EPS
    return np.array(
        [
            duration,
            f0_mean,
            f0_std,
            f0_slope,
            voiced_prob_mean,
            energy_mean,
            energy_std,
            energy_slope,
            silence_ratio,
            float(pause_count),
            float(longest_pause),
            float(word_count) / duration,
            float(phone_count) / duration,
            float(phone_count) / voiced_duration,
        ],
        dtype=np.float32,
    )


def slot_prosody_feature_names():
    return [
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


def compute_slot_prosody(audio, sample_rate, slot_times, pcn_word_id, top_phone_ids, phn_dict):
    feature_names = slot_prosody_feature_names()
    slot_count = len(top_phone_ids)
    out = np.zeros((slot_count, len(feature_names)), dtype=np.float32)
    if slot_count <= 0:
        return out
    audio = np.asarray(audio, dtype=np.float32)
    id_to_phone = {int(idx): phone for phone, idx in phn_dict.items()}
    hop_length = max(int(0.01 * sample_rate), 1)
    frame_length = max(int(0.025 * sample_rate), hop_length)
    if audio.size:
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        log_energy = np.log(np.clip(rms, EPS, None))
        frame_times = librosa.frames_to_time(np.arange(rms.shape[0]), sr=sample_rate, hop_length=hop_length)
    else:
        log_energy = np.zeros((0,), dtype=np.float32)
        frame_times = np.zeros((0,), dtype=np.float32)
    try:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception:
        f0 = np.zeros_like(log_energy, dtype=np.float32) * np.nan
    if f0 is None:
        f0 = np.zeros_like(log_energy, dtype=np.float32) * np.nan
    f0 = np.asarray(f0, dtype=np.float32)
    common_frames = min(frame_times.shape[0], log_energy.shape[0], f0.shape[0])
    frame_times = frame_times[:common_frames]
    log_energy = log_energy[:common_frames]
    f0 = f0[:common_frames]

    durations = np.zeros((slot_count,), dtype=np.float32)
    energy_means = np.zeros((slot_count,), dtype=np.float32)
    f0_means = np.zeros((slot_count,), dtype=np.float32)
    pcn_word_id = np.asarray(pcn_word_id, dtype=np.int32)
    for slot_idx in range(slot_count):
        if slot_idx < len(slot_times) and slot_times[slot_idx] is not None:
            start, end = slot_times[slot_idx]
            start = float(start)
            end = float(max(end, start))
        else:
            start = 0.0
            end = 0.0
        durations[slot_idx] = max(end - start, 0.0)
        frame_mask = (frame_times >= start) & (frame_times <= end) if frame_times.size else np.zeros((0,), dtype=bool)
        energy_values = log_energy[frame_mask]
        f0_values = f0[frame_mask]
        f0_values = f0_values[np.isfinite(f0_values)]
        if energy_values.size:
            out[slot_idx, 1] = float(np.mean(energy_values))
            out[slot_idx, 2] = float(np.std(energy_values))
            out[slot_idx, 3] = float(np.max(energy_values))
            energy_means[slot_idx] = out[slot_idx, 1]
        if f0_values.size:
            out[slot_idx, 4] = float(np.mean(f0_values))
            out[slot_idx, 5] = float(np.std(f0_values))
            out[slot_idx, 6] = float(np.max(f0_values))
            out[slot_idx, 7] = float(f0_values.size / max(frame_mask.sum(), 1))
            f0_means[slot_idx] = out[slot_idx, 4]
        out[slot_idx, 0] = durations[slot_idx]
        word_idx = int(pcn_word_id[slot_idx]) if slot_idx < pcn_word_id.shape[0] else -1
        same_word = np.flatnonzero(pcn_word_id == word_idx) if word_idx >= 0 else np.asarray([slot_idx], dtype=np.int64)
        if same_word.size:
            position = np.flatnonzero(same_word == slot_idx)
            out[slot_idx, 8] = float(position[0] / max(same_word.size - 1, 1)) if position.size else 0.0
            out[slot_idx, 9] = float(same_word.size)
        phone_idx = top_phone_ids[slot_idx] if slot_idx < len(top_phone_ids) else None
        phone = id_to_phone.get(int(phone_idx), '') if phone_idx is not None else ''
        out[slot_idx, 13] = 1.0 if phone[:1].upper() in {'A', 'E', 'I', 'O', 'U'} else 0.0

    for slot_idx in range(slot_count):
        word_idx = int(pcn_word_id[slot_idx]) if slot_idx < pcn_word_id.shape[0] else -1
        same_word = np.flatnonzero(pcn_word_id == word_idx) if word_idx >= 0 else np.asarray([slot_idx], dtype=np.int64)
        if same_word.size:
            word_energy = float(np.mean(energy_means[same_word]))
            word_duration = float(np.mean(durations[same_word]))
            word_f0 = float(np.mean(f0_means[same_word]))
            out[slot_idx, 10] = out[slot_idx, 1] - word_energy
            out[slot_idx, 11] = durations[slot_idx] / max(word_duration, EPS)
            out[slot_idx, 12] = out[slot_idx, 4] - word_f0
    return out.astype(np.float32)


def slope(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x = x - x.mean()
    y = values - values.mean()
    denom = float(np.sum(x * x))
    if denom <= EPS:
        return 0.0
    return float(np.sum(x * y) / denom)


def count_pauses(silence_mask, frame_step):
    pause_count = 0
    longest = 0.0
    cur = 0
    for value in silence_mask.tolist():
        if value:
            cur += 1
        elif cur > 0:
            pause_count += 1
            longest = max(longest, cur * frame_step)
            cur = 0
    if cur > 0:
        pause_count += 1
        longest = max(longest, cur * frame_step)
    return pause_count, longest


def build_gt_phone_rows(gold_words, gold_phone_segments):
    rows = []
    cursor = 0
    for word in gold_words:
        for phone_idx, phone in enumerate(word['phones']):
            segment = gold_phone_segments[cursor] if cursor < len(gold_phone_segments) else {}
            rows.append({
                'phone': phone,
                'phone_score': float(word['phone_scores'][phone_idx]),
                'word_id': int(word['word_id']),
                'word_text': word['text'],
                'word_accuracy': float(word['accuracy']),
                'word_stress': float(word['stress']),
                'word_total': float(word['total']),
                'end_time': float(segment.get('end_time', 0.0)),
            })
            cursor += 1
    return rows


def words_seen_in_nbest(hypotheses, gold_words):
    seen = set()
    gold_word_rows = [{'text': word['text']} for word in gold_words]
    for hyp in hypotheses:
        hyp_word_rows = [{'text': word} for word in hyp['words']]
        mapping = lcs_align_words(hyp_word_rows, gold_word_rows)
        for hyp_idx, gold_idx in enumerate(mapping):
            if gold_idx >= 0 and hyp_word_rows[hyp_idx]['text'] == gold_word_rows[gold_idx]['text']:
                seen.add(int(gold_words[gold_idx]['word_id']))
    return seen


def build_targets(item, cn_post, acoustic_post, acoustic_stats, top_phone_ids, hypotheses, phn_dict, cumulative_commit_mask):
    gt_rows = build_gt_phone_rows(item['gold_words'], item.get('gold_phone_segments', []))
    gt_tokens = [row['phone'] for row in gt_rows]
    id_to_phone = {idx: phone for phone, idx in phn_dict.items()}
    slot_tokens = [
        id_to_phone[phone_idx] if phone_idx is not None else EPS_TOKEN
        for phone_idx in top_phone_ids
    ]
    pairs = align_token_sequences(gt_tokens, slot_tokens)
    slot_to_gt = {}
    for gt_idx, slot_idx in pairs:
        if gt_idx is not None and slot_idx is not None:
            slot_to_gt[slot_idx] = gt_idx

    seen_word_ids = words_seen_in_nbest(hypotheses, item['gold_words'])
    phone_target = np.zeros((len(top_phone_ids), 2), dtype=np.float32) - 1.0
    word_target = np.zeros((len(top_phone_ids), 4), dtype=np.float32) - 1.0
    asr_correct_target = np.zeros((len(top_phone_ids),), dtype=np.float32)
    uncertainty_target = np.ones((len(top_phone_ids),), dtype=np.float32)
    soft_label_weight = np.zeros((len(top_phone_ids),), dtype=np.float32)
    confidence_target = np.zeros((len(top_phone_ids),), dtype=np.float32)
    confidence_loss_mask = np.zeros((len(top_phone_ids),), dtype=np.float32)
    abstention_target = np.zeros((len(top_phone_ids),), dtype=np.float32)
    abstention_loss_mask = np.zeros((len(top_phone_ids),), dtype=np.float32)

    for slot_idx, gt_idx in slot_to_gt.items():
        gt = gt_rows[gt_idx]
        gt_phone_id = phn_dict.get(gt['phone'])
        if gt_phone_id is None:
            continue
        committed = bool(slot_idx < len(cumulative_commit_mask) and cumulative_commit_mask[slot_idx] > 0)
        gt_word_seen = gt['word_id'] in seen_word_ids
        top_phone = slot_tokens[slot_idx]
        exact = top_phone == gt['phone']
        gt_posterior = float(cn_post[slot_idx, gt_phone_id])
        entropy = entropy_np(cn_post[slot_idx])
        entropy_norm = entropy / math.log(max(cn_post.shape[-1], 2))
        acoustic_support = float(acoustic_post[slot_idx, gt_phone_id]) if acoustic_post.shape[0] > slot_idx else 0.0
        js_div = float(acoustic_stats[slot_idx, 3]) if acoustic_stats.shape[0] > slot_idx else 1.0
        phone_target[slot_idx] = np.array([float(gt_phone_id), float(gt['phone_score'])], dtype=np.float32)
        word_target[slot_idx] = np.array(
            [
                float(gt['word_accuracy']),
                float(gt['word_stress']),
                float(gt['word_total']),
                float(gt['word_id']),
            ],
            dtype=np.float32,
        )
        asr_correct_target[slot_idx] = float(exact and gt_word_seen)
        uncertainty_target[slot_idx] = 1.0 if not gt_word_seen else float(np.clip(entropy_norm, 0.0, 1.0))
        if committed and gt_word_seen and exact:
            soft_label_weight[slot_idx] = 1.0
        elif committed and gt_word_seen and gt_posterior > 0:
            soft_label_weight[slot_idx] = gt_posterior
        else:
            soft_label_weight[slot_idx] = 0.0
        if committed:
            abstention_loss_mask[slot_idx] = 1.0
            confidence_loss_mask[slot_idx] = float(gt_word_seen)
            confidence_target[slot_idx] = float(
                np.clip(soft_label_weight[slot_idx] * (1.0 - np.clip(entropy_norm, 0.0, 1.0)) * acoustic_support, 0.0, 1.0)
            )
            abstention_target[slot_idx] = float(
                (not gt_word_seen)
                or soft_label_weight[slot_idx] < 0.05
                or (entropy_norm > 0.75 and js_div > 0.35)
                or ((not exact) and acoustic_support < 0.2)
            )

    return {
        'phone_target': phone_target,
        'word_target': word_target,
        'asr_correct_target': asr_correct_target,
        'uncertainty_target': uncertainty_target,
        'soft_label_weight': soft_label_weight,
        'confidence_target': confidence_target,
        'confidence_loss_mask': confidence_loss_mask,
        'abstention_target': abstention_target,
        'abstention_loss_mask': abstention_loss_mask,
        'gt_word_seen_count': int(len(seen_word_ids)),
    }


def build_examples_for_utterance(item, asr_generator, phone_mapper, phn_dict, phone_to_frame_id, args, utterance_index=0):
    examples = []
    skipped_chunks = []
    utt_id = item['utt_id']
    audio, _ = librosa.load(item['audio_path'], sr=args.sample_rate, mono=True)
    final_time = max(item['word_end_times'].values()) if item['word_end_times'] else item['audio_duration']
    prev_top_phone_ids = []
    prev_commit_state = None
    previous_chunk_id = -1

    for chunk_id, commit_time in enumerate(commit_schedule(final_time, args.chunk_sec)):
        is_final = abs(commit_time - final_time) < 1e-5
        audio_end = final_time if is_final else min(final_time, commit_time + args.right_context_sec)
        audio_prefix = audio[: int(max(audio_end, 1e-4) * args.sample_rate)]
        raw_hypotheses = asr_generator.generate(audio_prefix, args.sample_rate, audio_end)
        hypotheses, filtered_hypotheses = filter_looping_hypotheses(
            raw_hypotheses,
            phone_mapper,
            item['gold_words'],
            args,
        )
        if not hypotheses:
            skipped_chunks.append({
                'utt_id': utt_id,
                'chunk_id': int(chunk_id),
                'reason': 'all_nbest_hypotheses_filtered',
                'filtered_hypotheses': filtered_hypotheses,
            })
            continue
        hypotheses = [
            align_hypothesis_with_charsiu(
                item=item,
                row=dict(row),
                phone_mapper=phone_mapper,
                phn_dict=phn_dict,
                phone_to_frame_id=phone_to_frame_id,
                audio_end=audio_end,
            )
            for row in hypotheses
        ]
        pcn = build_pcn_from_hypotheses(hypotheses, phone_mapper, phn_dict)
        validate_pcn(pcn)
        pcn_word_id = top_hyp_word_ids_for_slots(pcn)
        cn_stats_arr, prefix_stability = pcn_stats(
            pcn['cn_post'],
            pcn['top_phone_ids'],
            prev_top_phone_ids,
            pcn['eps_index'],
        )
        prev_top_phone_ids = list(pcn['top_phone_ids'])
        acoustic_post, acoustic_stats, visible_frame_count = build_acoustic_evidence(
            item,
            pcn['cn_post'],
            pcn['top_phone_ids'],
            phone_to_frame_id,
            phn_dict,
            audio_end,
        )
        top_word_count = len(hypotheses[0]['words']) if hypotheses else 0
        top_phone_count = sum(1 for phone_idx in pcn['top_phone_ids'] if phone_idx is not None)
        prosody = compute_prosody(audio_prefix, args.sample_rate, audio_end, top_word_count, top_phone_count)
        slot_prosody = None
        if args.include_slot_prosody:
            slot_prosody = compute_slot_prosody(
                audio_prefix,
                args.sample_rate,
                pcn.get('slot_times', []),
                pcn_word_id,
                pcn['top_phone_ids'],
                phn_dict,
            )
        cumulative_commit_mask, new_commit_mask, mapped_old_slot, commit_diagnostics, prev_commit_state = build_stateful_commit_masks(
            prev_commit_state,
            pcn,
            pcn_word_id,
            hypotheses[0].get('word_timestamps', []) if hypotheses else [],
            commit_time,
            audio_end,
            is_final,
        )
        targets = build_targets(
            item=item,
            cn_post=pcn['cn_post'],
            acoustic_post=acoustic_post,
            acoustic_stats=acoustic_stats,
            top_phone_ids=pcn['top_phone_ids'],
            hypotheses=hypotheses,
            phn_dict=phn_dict,
            cumulative_commit_mask=cumulative_commit_mask,
        )
        committed_word_ids = set(int(x) for x in pcn_word_id[cumulative_commit_mask > 0].tolist() if int(x) >= 0)
        new_word_ids = set(int(x) for x in pcn_word_id[new_commit_mask > 0].tolist() if int(x) >= 0)
        examples.append({
            'utt_id': utt_id,
            'wav_path': item['audio_path'],
            'chunk_id': int(chunk_id),
            'previous_chunk_id': int(previous_chunk_id),
            'utterance_index': int(utterance_index),
            'state_reset': int(previous_chunk_id < 0),
            'commit_time': float(commit_time),
            'audio_end': float(audio_end),
            'is_final': bool(is_final),
            'cn_post': pcn['cn_post'].astype(np.float32),
            'cn_stats': cn_stats_arr.astype(np.float32),
            'acoustic_post': acoustic_post.astype(np.float32),
            'acoustic_stats': acoustic_stats.astype(np.float32),
            'prosody': prosody.astype(np.float32),
            **({'slot_prosody': slot_prosody.astype(np.float32)} if slot_prosody is not None else {}),
            'pcn_word_id': pcn_word_id.astype(np.int32),
            'cumulative_commit_mask': cumulative_commit_mask.astype(np.float32),
            'new_commit_mask': new_commit_mask.astype(np.float32),
            'commit_mask': cumulative_commit_mask.astype(np.float32),
            'mapped_old_slot': mapped_old_slot.astype(np.int32),
            'new_committed_word_count': int(len(new_word_ids)),
            'cumulative_committed_word_count': int(len(committed_word_ids)),
            'commit_alignment_diagnostics': commit_diagnostics,
            **targets,
            'utt_target': np.array(
                [
                    item['utt_scores']['accuracy'],
                    item['utt_scores']['completeness'],
                    item['utt_scores']['fluency'],
                    item['utt_scores']['prosodic'],
                    item['utt_scores']['total'],
                ],
                dtype=np.float32,
            ),
            'hypotheses': hypotheses,
            'raw_hypothesis_count': int(len(raw_hypotheses)),
            'filtered_hypotheses': filtered_hypotheses,
            'hyp_weights': pcn['hyp_weights'],
            'slot_confidences': pcn.get('slot_confidences', []),
            'slot_times': pcn.get('slot_times', []),
            'prefix_stability': float(prefix_stability),
            'visible_frame_count': int(visible_frame_count),
            'top_phone_count': int(top_phone_count),
            'coverage_ratio': float(audio_end / max(final_time, EPS)),
        })
        previous_chunk_id = int(chunk_id)
    return examples, skipped_chunks


def build_split(split_name, split_items, scores, charsiu, asr_generator, phone_mapper, phn_dict, phone_to_frame_id, args):
    examples = []
    skipped = []
    skipped_chunks = []
    for utterance_index, (utt_id, audio_path) in enumerate(tqdm(split_items, desc=f'{split_name}-pcn')):
        aligned = align_gold_utterance(
            utt_id=utt_id,
            audio_path=audio_path,
            scores=scores,
            charsiu=charsiu,
            sample_rate=args.sample_rate,
            device=args.device,
            phone_to_frame_id=phone_to_frame_id,
            phn_dict=phn_dict,
        )
        if 'skip_reason' in aligned:
            skipped.append(aligned)
            continue
        try:
            cur_examples, cur_skipped_chunks = build_examples_for_utterance(
                item=aligned,
                asr_generator=asr_generator,
                phone_mapper=phone_mapper,
                phn_dict=phn_dict,
                phone_to_frame_id=phone_to_frame_id,
                args=args,
                utterance_index=utterance_index,
            )
            examples.extend(cur_examples)
            skipped_chunks.extend(cur_skipped_chunks)
        except Exception as exc:
            skipped.append({'utt_id': utt_id, 'skip_reason': f'pcn_build_failed:{exc}'})
    return examples, skipped, skipped_chunks


def build_split_incremental(
    split_name,
    split_items,
    scores,
    charsiu,
    asr_generator,
    phone_mapper,
    phn_dict,
    phone_to_frame_id,
    args,
    output_dir,
):
    indexed_items = [
        (utterance_index, utt_id, audio_path)
        for utterance_index, (utt_id, audio_path) in enumerate(split_items)
        if utterance_index % args.num_shards == args.shard_index
    ]
    processed = 0
    resumed = 0
    for utterance_index, utt_id, audio_path in tqdm(indexed_items, desc=f'{split_name}-pcn-shard{args.shard_index}', total=len(indexed_items)):
        record_path = progress_path(output_dir, split_name, utt_id)
        if args.resume and record_path.exists():
            try:
                record = read_progress_record(record_path)
                if record.get('schema') == PCN_SCHEMA and record.get('status') in {'ok', 'skipped'}:
                    resumed += 1
                    continue
            except Exception:
                pass

        aligned = align_gold_utterance(
            utt_id=utt_id,
            audio_path=audio_path,
            scores=scores,
            charsiu=charsiu,
            sample_rate=args.sample_rate,
            device=args.device,
            phone_to_frame_id=phone_to_frame_id,
            phn_dict=phn_dict,
        )
        if 'skip_reason' in aligned:
            record = make_progress_record(
                split_name=split_name,
                utterance_index=utterance_index,
                utt_id=utt_id,
                audio_path=audio_path,
                status='skipped',
                skip_record=aligned,
            )
            write_progress_record(record_path, record)
            processed += 1
            continue

        try:
            cur_examples, cur_skipped_chunks = build_examples_for_utterance(
                item=aligned,
                asr_generator=asr_generator,
                phone_mapper=phone_mapper,
                phn_dict=phn_dict,
                phone_to_frame_id=phone_to_frame_id,
                args=args,
                utterance_index=utterance_index,
            )
            record = make_progress_record(
                split_name=split_name,
                utterance_index=utterance_index,
                utt_id=utt_id,
                audio_path=audio_path,
                status='ok',
                examples=cur_examples,
                skipped_chunks=cur_skipped_chunks,
            )
        except Exception as exc:
            record = make_progress_record(
                split_name=split_name,
                utterance_index=utterance_index,
                utt_id=utt_id,
                audio_path=audio_path,
                status='skipped',
                skip_record={'utt_id': utt_id, 'skip_reason': f'pcn_build_failed:{exc}'},
            )
        write_progress_record(record_path, record)
        processed += 1
    return {
        'split': split_name,
        'shard_index': int(args.shard_index),
        'num_shards': int(args.num_shards),
        'assigned_utterances': int(len(indexed_items)),
        'processed_utterances': int(processed),
        'resumed_utterances': int(resumed),
    }


def collect_split_progress(split_name, split_items, output_dir):
    examples = []
    skipped = []
    skipped_chunks = []
    missing = []
    malformed = []
    status_counter = Counter()
    for utterance_index, (utt_id, audio_path) in enumerate(split_items):
        record_file = progress_path(output_dir, split_name, utt_id)
        if not record_file.exists():
            missing.append({'utt_id': utt_id, 'utterance_index': int(utterance_index), 'audio_path': str(audio_path)})
            continue
        try:
            record = read_progress_record(record_file)
        except Exception as exc:
            malformed.append({'utt_id': utt_id, 'utterance_index': int(utterance_index), 'error': str(exc)})
            continue
        if record.get('schema') != PCN_SCHEMA:
            malformed.append({
                'utt_id': utt_id,
                'utterance_index': int(utterance_index),
                'error': f"unexpected_schema:{record.get('schema')}",
            })
            continue
        status = record.get('status', 'unknown')
        status_counter[status] += 1
        if status == 'ok':
            examples.extend(record.get('examples', []) or [])
            skipped_chunks.extend(record.get('skipped_chunks', []) or [])
        else:
            skipped.append(record.get('skip_record') or {'utt_id': utt_id, 'skip_reason': status})
    summary = {
        'split': split_name,
        'expected_utterances': int(len(split_items)),
        'ok_utterances': int(status_counter.get('ok', 0)),
        'skipped_utterances': int(status_counter.get('skipped', 0)),
        'missing_utterances': int(len(missing)),
        'malformed_utterances': int(len(malformed)),
        'examples': int(len(examples)),
        'skipped_chunks': int(len(skipped_chunks)),
        'missing_examples': missing[:20],
        'malformed_examples': malformed[:20],
        'status_counter': dict(status_counter),
    }
    return examples, skipped, skipped_chunks, summary


def infer_seq_len(examples, max_seq_len):
    longest = max((example['cn_post'].shape[0] for example in examples), default=0)
    if longest <= 0:
        raise ValueError('No PCN examples generated.')
    if max_seq_len > 0:
        if longest > max_seq_len:
            raise ValueError(f'max_seq_len={max_seq_len} is smaller than longest PCN length={longest}.')
        return max_seq_len
    return longest


def pad_2d(src, length, fill_value=0.0):
    out = np.zeros((length,) + src.shape[1:], dtype=src.dtype)
    if fill_value != 0.0:
        out[...] = fill_value
    out[: src.shape[0]] = src
    return out


def build_arrays(examples, seq_len, phone_dim, prosody_dim):
    include_slot_prosody = any('slot_prosody' in example for example in examples)
    slot_prosody_dim = len(slot_prosody_feature_names()) if include_slot_prosody else 0
    arrays = {
        'cn_post': [],
        'cn_stats': [],
        'acoustic_post': [],
        'acoustic_stats': [],
        'prosody': [],
        'pcn_word_id': [],
        'phone_target': [],
        'word_target': [],
        'utt_target': [],
        'asr_correct_target': [],
        'uncertainty_target': [],
        'soft_label_weight': [],
        'commit_mask': [],
        'cumulative_commit_mask': [],
        'new_commit_mask': [],
        'mapped_old_slot': [],
        'confidence_target': [],
        'confidence_loss_mask': [],
        'abstention_target': [],
        'abstention_loss_mask': [],
        'teacher_prefix_utt_score': [],
        'teacher_final_utt_score': [],
        'teacher_utt_mask': [],
        'teacher_word_score': [],
        'teacher_word_mask': [],
        'coverage_ratio': [],
        'visible_len': [],
        'is_final': [],
        'chunk_id': [],
        'previous_chunk_id': [],
        'utterance_index': [],
        'state_reset': [],
        'new_committed_word_count': [],
        'cumulative_committed_word_count': [],
        'prefix_stability': [],
    }
    if include_slot_prosody:
        arrays['slot_prosody'] = []
        arrays['slot_is_vowel'] = []
        arrays['slot_voiced_ratio'] = []
    manifest = []
    for example in examples:
        visible_len = int(example['cn_post'].shape[0])
        arrays['cn_post'].append(pad_2d(example['cn_post'], seq_len))
        arrays['cn_stats'].append(pad_2d(example['cn_stats'], seq_len))
        arrays['acoustic_post'].append(pad_2d(example['acoustic_post'], seq_len))
        arrays['acoustic_stats'].append(pad_2d(example['acoustic_stats'], seq_len))
        arrays['prosody'].append(example['prosody'].reshape(prosody_dim))
        if include_slot_prosody:
            cur_slot_prosody = np.asarray(
                example.get('slot_prosody', np.zeros((visible_len, slot_prosody_dim), dtype=np.float32)),
                dtype=np.float32,
            )
            arrays['slot_prosody'].append(
                pad_2d(
                    cur_slot_prosody,
                    seq_len,
                )
            )
            arrays['slot_is_vowel'].append(pad_2d(cur_slot_prosody[:, 13:14], seq_len).reshape(seq_len))
            arrays['slot_voiced_ratio'].append(pad_2d(cur_slot_prosody[:, 7:8], seq_len).reshape(seq_len))
        arrays['pcn_word_id'].append(pad_2d(example['pcn_word_id'].reshape(-1, 1), seq_len, fill_value=-1).reshape(seq_len))
        arrays['phone_target'].append(pad_2d(example['phone_target'], seq_len, fill_value=-1.0))
        arrays['word_target'].append(pad_2d(example['word_target'], seq_len, fill_value=-1.0))
        arrays['utt_target'].append(example['utt_target'])
        arrays['asr_correct_target'].append(pad_2d(example['asr_correct_target'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['uncertainty_target'].append(pad_2d(example['uncertainty_target'].reshape(-1, 1), seq_len, fill_value=1.0).reshape(seq_len))
        arrays['soft_label_weight'].append(pad_2d(example['soft_label_weight'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['commit_mask'].append(pad_2d(example['commit_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['cumulative_commit_mask'].append(pad_2d(example['cumulative_commit_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['new_commit_mask'].append(pad_2d(example['new_commit_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['mapped_old_slot'].append(pad_2d(example['mapped_old_slot'].reshape(-1, 1), seq_len, fill_value=-1).reshape(seq_len))
        arrays['confidence_target'].append(pad_2d(example['confidence_target'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['confidence_loss_mask'].append(pad_2d(example['confidence_loss_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['abstention_target'].append(pad_2d(example['abstention_target'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['abstention_loss_mask'].append(pad_2d(example['abstention_loss_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['teacher_prefix_utt_score'].append(np.asarray(example.get('teacher_prefix_utt_score', np.zeros((5,), dtype=np.float32)), dtype=np.float32))
        arrays['teacher_final_utt_score'].append(np.asarray(example.get('teacher_final_utt_score', np.zeros((5,), dtype=np.float32)), dtype=np.float32))
        arrays['teacher_utt_mask'].append(float(example.get('teacher_utt_mask', 0.0)))
        arrays['teacher_word_score'].append(
            pad_2d(
                np.asarray(example.get('teacher_word_score', np.zeros((visible_len, 3), dtype=np.float32)), dtype=np.float32),
                seq_len,
            )
        )
        arrays['teacher_word_mask'].append(
            pad_2d(
                np.asarray(example.get('teacher_word_mask', np.zeros((visible_len,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1),
                seq_len,
            ).reshape(seq_len)
        )
        arrays['coverage_ratio'].append(float(example.get('coverage_ratio', 0.0)))
        arrays['visible_len'].append(visible_len)
        arrays['is_final'].append(int(example['is_final']))
        arrays['chunk_id'].append(int(example['chunk_id']))
        arrays['previous_chunk_id'].append(int(example['previous_chunk_id']))
        arrays['utterance_index'].append(int(example['utterance_index']))
        arrays['state_reset'].append(int(example['state_reset']))
        arrays['new_committed_word_count'].append(int(example['new_committed_word_count']))
        arrays['cumulative_committed_word_count'].append(int(example['cumulative_committed_word_count']))
        arrays['prefix_stability'].append(float(example['prefix_stability']))
        manifest.append({
            'utt_id': example['utt_id'],
            'wav_path': example.get('wav_path', ''),
            'chunk_id': int(example['chunk_id']),
            'previous_chunk_id': int(example['previous_chunk_id']),
            'utterance_index': int(example['utterance_index']),
            'state_reset': int(example['state_reset']),
            'commit_time': float(example['commit_time']),
            'audio_end': float(example['audio_end']),
            'is_final': bool(example['is_final']),
            'visible_len': visible_len,
            'top_phone_count': int(example['top_phone_count']),
            'visible_frame_count': int(example['visible_frame_count']),
            'prefix_stability': float(example['prefix_stability']),
            'coverage_ratio': float(example.get('coverage_ratio', 0.0)),
            'gt_word_seen_count': int(example['gt_word_seen_count']),
            'supervised_slot_count': int(np.sum(example['soft_label_weight'] > 0)),
            'new_committed_word_count': int(example['new_committed_word_count']),
            'cumulative_committed_word_count': int(example['cumulative_committed_word_count']),
            'commit_alignment_diagnostics': example.get('commit_alignment_diagnostics', {}),
            'raw_hypothesis_count': int(example.get('raw_hypothesis_count', len(example['hypotheses']))),
            'kept_hypothesis_count': int(len(example['hypotheses'])),
            'filtered_hypotheses': example.get('filtered_hypotheses', []),
            'hyp_text': [row['text'] for row in example['hypotheses']],
            'hyp_logprob': [float(row['logprob']) for row in example['hypotheses']],
            'hyp_sequence_score': [float(row.get('sequence_score', row.get('logprob', 0.0))) for row in example['hypotheses']],
            'hyp_length_normalized_sequence_score': [
                float(row.get('length_normalized_sequence_score', row.get('logprob', 0.0)))
                for row in example['hypotheses']
            ],
            'hyp_weight': [float(weight) for weight in example['hyp_weights']],
            'word_timestamps': [row['word_timestamps'] for row in example['hypotheses']],
            'timestamp_source': [row.get('timestamp_source', '') for row in example['hypotheses']],
            'token_ids': [row.get('token_ids', []) for row in example['hypotheses']],
            'token_logprobs': [row.get('token_logprobs', []) for row in example['hypotheses']],
            'token_confidences': [row.get('token_confidences', []) for row in example['hypotheses']],
            'word_token_ranges': [row.get('word_token_ranges', []) for row in example['hypotheses']],
            'word_logprobs': [row.get('word_logprobs', []) for row in example['hypotheses']],
            'word_confidences': [row.get('word_confidences', []) for row in example['hypotheses']],
            'slot_times': example.get('slot_times', []),
        })

    out = {
        'cn_post': np.stack(arrays['cn_post']).astype(np.float32),
        'cn_stats': np.stack(arrays['cn_stats']).astype(np.float32),
        'acoustic_post': np.stack(arrays['acoustic_post']).astype(np.float32),
        'acoustic_stats': np.stack(arrays['acoustic_stats']).astype(np.float32),
        'prosody': np.stack(arrays['prosody']).astype(np.float32),
        'pcn_word_id': np.stack(arrays['pcn_word_id']).astype(np.int32),
        'phone_target': np.stack(arrays['phone_target']).astype(np.float32),
        'word_target': np.stack(arrays['word_target']).astype(np.float32),
        'utt_target': np.stack(arrays['utt_target']).astype(np.float32),
        'asr_correct_target': np.stack(arrays['asr_correct_target']).astype(np.float32),
        'uncertainty_target': np.stack(arrays['uncertainty_target']).astype(np.float32),
        'soft_label_weight': np.stack(arrays['soft_label_weight']).astype(np.float32),
        'commit_mask': np.stack(arrays['commit_mask']).astype(np.float32),
        'cumulative_commit_mask': np.stack(arrays['cumulative_commit_mask']).astype(np.float32),
        'new_commit_mask': np.stack(arrays['new_commit_mask']).astype(np.float32),
        'mapped_old_slot': np.stack(arrays['mapped_old_slot']).astype(np.int32),
        'confidence_target': np.stack(arrays['confidence_target']).astype(np.float32),
        'confidence_loss_mask': np.stack(arrays['confidence_loss_mask']).astype(np.float32),
        'abstention_target': np.stack(arrays['abstention_target']).astype(np.float32),
        'abstention_loss_mask': np.stack(arrays['abstention_loss_mask']).astype(np.float32),
        'teacher_prefix_utt_score': np.stack(arrays['teacher_prefix_utt_score']).astype(np.float32),
        'teacher_final_utt_score': np.stack(arrays['teacher_final_utt_score']).astype(np.float32),
        'teacher_utt_mask': np.asarray(arrays['teacher_utt_mask'], dtype=np.float32),
        'teacher_word_score': np.stack(arrays['teacher_word_score']).astype(np.float32),
        'teacher_word_mask': np.stack(arrays['teacher_word_mask']).astype(np.float32),
        'coverage_ratio': np.asarray(arrays['coverage_ratio'], dtype=np.float32),
        'visible_len': np.asarray(arrays['visible_len'], dtype=np.int32),
        'is_final': np.asarray(arrays['is_final'], dtype=np.int8),
        'chunk_id': np.asarray(arrays['chunk_id'], dtype=np.int32),
        'previous_chunk_id': np.asarray(arrays['previous_chunk_id'], dtype=np.int32),
        'utterance_index': np.asarray(arrays['utterance_index'], dtype=np.int32),
        'state_reset': np.asarray(arrays['state_reset'], dtype=np.int8),
        'new_committed_word_count': np.asarray(arrays['new_committed_word_count'], dtype=np.int32),
        'cumulative_committed_word_count': np.asarray(arrays['cumulative_committed_word_count'], dtype=np.int32),
        'prefix_stability': np.asarray(arrays['prefix_stability'], dtype=np.float32),
        'manifest': manifest,
    }
    if include_slot_prosody:
        out['slot_prosody'] = np.stack(arrays['slot_prosody']).astype(np.float32)
        out['slot_is_vowel'] = np.stack(arrays['slot_is_vowel']).astype(np.float32)
        out['slot_voiced_ratio'] = np.stack(arrays['slot_voiced_ratio']).astype(np.float32)
    return out


def save_split(split_name, arrays, output_dir):
    payload = {
        'cn_post': arrays['cn_post'],
        'cn_stats': arrays['cn_stats'],
        'acoustic_post': arrays['acoustic_post'],
        'acoustic_stats': arrays['acoustic_stats'],
        'prosody': arrays['prosody'],
        'pcn_word_id': arrays['pcn_word_id'],
        'phone_target': arrays['phone_target'],
        'word_target': arrays['word_target'],
        'utt_target': arrays['utt_target'],
        'asr_correct_target': arrays['asr_correct_target'],
        'uncertainty_target': arrays['uncertainty_target'],
        'soft_label_weight': arrays['soft_label_weight'],
        'commit_mask': arrays['commit_mask'],
        'cumulative_commit_mask': arrays['cumulative_commit_mask'],
        'new_commit_mask': arrays['new_commit_mask'],
        'mapped_old_slot': arrays['mapped_old_slot'],
        'confidence_target': arrays['confidence_target'],
        'confidence_loss_mask': arrays['confidence_loss_mask'],
        'abstention_target': arrays['abstention_target'],
        'abstention_loss_mask': arrays['abstention_loss_mask'],
        'teacher_prefix_utt_score': arrays['teacher_prefix_utt_score'],
        'teacher_final_utt_score': arrays['teacher_final_utt_score'],
        'teacher_utt_mask': arrays['teacher_utt_mask'],
        'teacher_word_score': arrays['teacher_word_score'],
        'teacher_word_mask': arrays['teacher_word_mask'],
        'coverage_ratio': arrays['coverage_ratio'],
        'visible_len': arrays['visible_len'],
        'is_final': arrays['is_final'],
        'chunk_id': arrays['chunk_id'],
        'previous_chunk_id': arrays['previous_chunk_id'],
        'utterance_index': arrays['utterance_index'],
        'state_reset': arrays['state_reset'],
        'new_committed_word_count': arrays['new_committed_word_count'],
        'cumulative_committed_word_count': arrays['cumulative_committed_word_count'],
        'prefix_stability': arrays['prefix_stability'],
    }
    if 'slot_prosody' in arrays:
        payload['slot_prosody'] = arrays['slot_prosody']
    if 'slot_is_vowel' in arrays:
        payload['slot_is_vowel'] = arrays['slot_is_vowel']
    if 'slot_voiced_ratio' in arrays:
        payload['slot_voiced_ratio'] = arrays['slot_voiced_ratio']
    np.savez_compressed(output_dir / f'{split_name}_chunks.npz', **payload)
    with open(output_dir / f'{split_name}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for row in arrays['manifest']:
            handle.write(json.dumps(safe_json(row), ensure_ascii=False) + '\n')


def main():
    args = get_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    validate_shard_args(args)
    target_splits = parse_target_splits(args.target_splits)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume or args.finalize_only):
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild or --resume to continue.')
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = Path(args.scores_json)
    if not scores_path.is_absolute():
        scores_path = Path.cwd() / scores_path
    with open(scores_path, 'r', encoding='utf-8') as handle:
        scores = json.load(handle)

    dataset_root = Path(args.dataset_root)
    train_items, val_items, test_items, split_meta = resolve_dataset_splits(
        dataset_root=dataset_root,
        train_scp=args.train_scp,
        val_scp=args.val_scp,
        test_scp=args.test_scp,
        val_ratio=args.val_speaker_ratio,
        split_seed=args.split_seed,
    )
    split_items = {'train': train_items, 'val': val_items, 'test': test_items}
    utt_ids = [utt_id for split in target_splits for utt_id, _ in split_items[split] if utt_id in scores]
    phn_dict = build_phone_vocab(scores, utt_ids)
    lexicon = build_word_lexicon(scores, utt_ids)
    phone_mapper = PhoneMapper(lexicon, phn_dict)

    progress_generation_summary = {}
    if not args.finalize_only:
        charsiu = load_official_charsiu_aligner(
            model_name=args.aligner_model,
            device=args.device,
            sample_rate=args.sample_rate,
            sil_threshold=args.min_sil_frames,
            lang=args.charsiu_lang,
            charsiu_src_dir=args.charsiu_src_dir,
        )
        phone_to_frame_id, id2label, silence_ids = build_model_phone_map(charsiu)
        asr_generator = WhisperNBestGenerator(
            model_name=args.asr_model,
            language=args.language,
            device=args.device,
            nbest=args.nbest,
            beam_size=args.beam_size,
            max_new_tokens=args.asr_max_new_tokens,
            no_repeat_ngram_size=args.asr_no_repeat_ngram_size,
        )

        for split_name in target_splits:
            progress_generation_summary[split_name] = build_split_incremental(
                split_name=split_name,
                split_items=split_items[split_name],
                scores=scores,
                charsiu=charsiu,
                asr_generator=asr_generator,
                phone_mapper=phone_mapper,
                phn_dict=phn_dict,
                phone_to_frame_id=phone_to_frame_id,
                args=args,
                output_dir=output_dir,
            )

    if args.skip_finalize:
        progress_summary = {
            split_name: collect_split_progress(split_name, split_items[split_name], output_dir)[3]
            for split_name in target_splits
        }
        with open(output_dir / 'progress_summary.json', 'w', encoding='utf-8') as handle:
            json.dump(
                safe_json({
                    'schema': PCN_SCHEMA,
                    'pcn_type': PCN_TYPE,
                    'generation': progress_generation_summary,
                    'progress': progress_summary,
                }),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return

    split_examples = {}
    skipped = {}
    progress_summary = {}
    all_examples = []
    incomplete = []
    for split_name in target_splits:
        examples, cur_skipped, cur_skipped_chunks, cur_progress_summary = collect_split_progress(
            split_name=split_name,
            split_items=split_items[split_name],
            output_dir=output_dir,
        )
        split_examples[split_name] = examples
        skipped[split_name] = cur_skipped
        skipped[f'{split_name}_chunks'] = cur_skipped_chunks
        progress_summary[split_name] = cur_progress_summary
        all_examples.extend(examples)
        if cur_progress_summary['missing_utterances'] > 0 or cur_progress_summary['malformed_utterances'] > 0:
            incomplete.append(cur_progress_summary)
    if incomplete:
        with open(output_dir / 'progress_summary.json', 'w', encoding='utf-8') as handle:
            json.dump(
                safe_json({
                    'schema': PCN_SCHEMA,
                    'pcn_type': PCN_TYPE,
                    'generation': progress_generation_summary,
                    'progress': progress_summary,
                }),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        raise RuntimeError(
            'Progress is incomplete; not finalizing partial data. '
            f'See {output_dir / "progress_summary.json"}. '
            'Continue with --resume, or run all shards then --finalize-only.'
        )

    seq_len = infer_seq_len(all_examples, args.max_seq_len)
    phone_dim = len(phn_dict) + 1
    prosody_names = [
        'duration',
        'f0_mean',
        'f0_std',
        'f0_slope',
        'voiced_probability',
        'log_energy_mean',
        'log_energy_std',
        'log_energy_slope',
        'silence_ratio',
        'pause_count',
        'longest_pause',
        'word_rate',
        'phone_rate',
        'articulation_rate',
    ]
    slot_prosody_names = slot_prosody_feature_names() if args.include_slot_prosody else []
    for split_name in target_splits:
        arrays = build_arrays(split_examples[split_name], seq_len, phone_dim, len(prosody_names))
        save_split(split_name, arrays, output_dir)

    timestamp_counter = Counter()
    for example in all_examples:
        for row in example.get('hypotheses', []):
            timestamp_counter[row.get('timestamp_source', 'unknown')] += 1

    metadata = {
        'schema': PCN_SCHEMA,
        'pcn_type': PCN_TYPE,
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'split_meta': split_meta,
        'target_splits': target_splits,
        'progress': progress_summary,
        'progress_generation': progress_generation_summary,
        'resume_enabled': bool(args.resume),
        'finalize_only': bool(args.finalize_only),
        'num_shards': int(args.num_shards),
        'shard_index': int(args.shard_index),
        'aligner_model': args.aligner_model,
        'asr_model': args.asr_model,
        'nbest': int(args.nbest),
        'beam_size': int(args.beam_size),
        'asr_generation_filters': {
            'max_new_tokens': int(args.asr_max_new_tokens),
            'no_repeat_ngram_size': int(args.asr_no_repeat_ngram_size),
            'strict_repeat_word_trigger': int(args.asr_max_words),
            'strict_repeat_phone_trigger': int(args.asr_max_visible_phones),
            'max_phone_ratio': float(args.asr_max_phone_ratio),
            'repeat_ngram_min_repeats': int(args.asr_repeat_ngram_min_repeats),
            'repeat_max_ngram_size': int(args.asr_repeat_max_ngram_size),
            'repeat_ngram_coverage': float(args.asr_repeat_ngram_coverage),
            'repeat_token_ratio': float(args.asr_repeat_token_ratio),
            'length_policy': 'word_count and absolute phone_count only enable strict repetition checks; non-repetitive long hypotheses are retained',
            'hard_rejection_policy': 'drop individual N-best hypotheses with detected repetition or ASR/reference phone ratio above max_phone_ratio; drop chunk only if all hypotheses are filtered',
        },
        'chunk_sec': float(args.chunk_sec),
        'right_context_sec': float(args.right_context_sec),
        'seq_len': int(seq_len),
        'phone_dim': int(phone_dim),
        'epsilon_index': int(len(phn_dict)),
        'phn_dict': phn_dict,
        'cn_stats': ['epsilon_probability', 'entropy', 'top1_probability', 'top1_top2_margin', 'prefix_stability'],
        'acoustic_stats': ['entropy', 'top1_top2_margin', 'duration', 'pcn_charsiu_js_divergence'],
        'prosody': prosody_names,
        **(
            {
                'slot_prosody': slot_prosody_names,
                'slot_prosody_note': 'Lexical stress one-hot features are zero fallback because reliable lexical stress is not available in this builder.',
            }
            if args.include_slot_prosody
            else {}
        ),
        'targets': [
            'phone_target',
            'word_target',
            'utt_target',
            'asr_correct_target',
            'uncertainty_target',
            'soft_label_weight',
            'commit_mask',
            'cumulative_commit_mask',
            'new_commit_mask',
            'mapped_old_slot',
            'confidence_target',
            'confidence_loss_mask',
            'abstention_target',
            'abstention_loss_mask',
            'pcn_word_id',
            'visible_len',
            'is_final',
            'chunk_id',
            'previous_chunk_id',
            'utterance_index',
            'state_reset',
            'new_committed_word_count',
            'cumulative_committed_word_count',
            'prefix_stability',
            'teacher_prefix_utt_score',
            'teacher_final_utt_score',
            'teacher_word_score',
            'teacher_word_mask',
            'coverage_ratio',
        ] + (['slot_prosody', 'slot_is_vowel', 'slot_voiced_ratio'] if args.include_slot_prosody else []),
        'supervision_policy': {
            'gt_usage': 'GT text/phones/scores are used only for offline training supervision, not as inference input.',
            'exact_match': 'soft_label_weight=1 and asr_correct_target=1 when the committed PCN top phone matches GT and its GT word appears in N-best.',
            'phone_approx_match': 'soft_label_weight=PCN posterior of the GT phone when GT word appears in N-best but top phone differs.',
            'mismatch': 'soft_label_weight=0 and asr_correct_target=0; the chunk is retained.',
            'gt_not_in_nbest': 'word/phone scoring is not supervised for that slot; uncertainty_target=1.',
        },
        'distillation_fields': {
            'teacher_prefix_utt_score': 'Optional MultiPA/open-teacher utterance scores on the current prefix, scaled 0-5 before training normalization.',
            'teacher_final_utt_score': 'Optional MultiPA/open-teacher utterance scores on the full audio, used with coverage-ratio weighting.',
            'teacher_word_score': 'Optional teacher word scores aligned to PCN slots by time overlap.',
            'teacher_word_mask': '1 where teacher_word_score is valid.',
            'coverage_ratio': 'audio_end/full_duration, used by prefix-to-final distillation.',
        },
        'word_timestamp_policy': 'Primary path aligns each N-best G2P phone sequence to Charsiu frame posteriors and aggregates phone spans to words. duration_proportional_fallback is used only when hypothesis alignment fails.',
        'timestamp_source_counts': dict(timestamp_counter),
        'streaming_state_policy': {
            'cumulative_commit_mask': 'All ASR/PCN slots committed by the current chunk, including old slots mapped from previous chunks.',
            'new_commit_mask': 'Only newly committed complete top-hypothesis words after monotonic top-phone mapping from the previous chunk.',
            'gru_update': 'The sentence GRU must update only from new_commit_mask; commit_mask is a cumulative compatibility alias.',
            'commit_inputs': 'Commit logic uses ASR/PCN hypothesis timestamps, prefix stability, audio time, and Charsiu acoustic evidence; GT is only used for training labels and loss masks.',
        },
        'skipped': skipped,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(safe_json(metadata), handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
