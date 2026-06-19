import argparse
import gc
import inspect
import json
import math
import os
import pickle
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

from build_charsiu_seq_data import (
    EPS,
    align_reference_utterance,
    audio_logits,
    build_silence_keep_mask,
    build_model_phone_map,
    build_reference_word_records,
    load_official_charsiu_aligner,
    monotonic_align,
    normalize_phone,
    normalize_word,
    resolve_dataset_splits,
    segment_feature,
)
from build_streaming_charsiu_data import commit_schedule


def get_args():
    parser = argparse.ArgumentParser(description='Build streaming GOPT chunks from ASR hypotheses instead of gold text.')
    parser.add_argument('--dataset-root', type=str, required=True, help='SpeechOcean762 root that contains train/test wav.scp and WAVE/.')
    parser.add_argument('--scores-json', type=str, default='src/prep_data/scores.json')
    parser.add_argument('--train-scp', type=str, default=None)
    parser.add_argument('--val-scp', type=str, default=None)
    parser.add_argument('--test-scp', type=str, default=None)
    parser.add_argument('--val-speaker-ratio', type=float, default=0.5, help='When --val-scp is not set, hold out this fraction of original test speakers for validation.')
    parser.add_argument('--split-seed', type=int, default=1337)
    parser.add_argument('--output-dir', type=str, default='data/streaming_asr_gopt')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--charsiu-src-dir', type=str, default=os.environ.get('CHARSIU_SRC_DIR'))
    parser.add_argument('--charsiu-lang', type=str, default=os.environ.get('CHARSIU_LANG', 'en'))
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
    parser.add_argument('--asr-batch-size', type=int, default=4, help='Initial batch size used when decoding prefix chunks with the transformers Whisper pipeline.')
    parser.add_argument('--asr-min-batch-size', type=int, default=1, help='Minimum micro-batch size when retrying transformers Whisper decoding after CUDA OOM.')
    parser.add_argument('--asr-max-new-tokens', type=int, default=128, help='Upper bound for Whisper generation length during ASR-driven chunk building.')
    parser.add_argument('--asr-no-repeat-ngram-size', type=int, default=0, help='Transformers Whisper no-repeat ngram size. Disabled by default; post-decode repetition checks preserve valid repeated phrases.')
    parser.add_argument('--asr-max-words', type=int, default=64, help='Word-count threshold that enables strict repetition checks. Length alone never rejects a hypothesis. Set 0 to disable the trigger.')
    parser.add_argument('--asr-max-visible-phones', type=int, default=100, help='ASR-phone threshold that enables strict repetition checks. Length alone never rejects a hypothesis. Set 0 to disable the trigger.')
    parser.add_argument('--asr-max-phone-ratio', type=float, default=3.0, help='Reject when ASR phones exceed this multiple of reference phones. Set 0 to disable.')
    parser.add_argument('--asr-repeat-ngram-min-repeats', type=int, default=4, help='Reject a repeated word pattern seen at least this many times. Set 0 to disable.')
    parser.add_argument('--asr-repeat-max-ngram-size', type=int, default=12, help='Maximum word-pattern length checked for repetition.')
    parser.add_argument('--asr-repeat-ngram-coverage', type=float, default=0.6, help='Strict-mode minimum fraction covered by repeated non-overlapping ngrams.')
    parser.add_argument('--asr-repeat-token-ratio', type=float, default=0.5, help='Strict-mode minimum fraction occupied by one repeated token.')
    parser.add_argument('--asr-use-cache', action='store_true', help='Enable Whisper KV-cache during generation. Disabled by default to reduce peak GPU memory.')
    parser.add_argument('--asr-torch-dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'], help='Torch dtype for the transformers Whisper pipeline.')
    parser.add_argument('--asr-empty-cache', action='store_true', help='Call torch.cuda.empty_cache() between ASR micro-batches.')
    parser.add_argument('--target-splits', type=str, default='train,val,test', help='Comma-separated splits to process/finalize. Choices: train,val,test')
    parser.add_argument('--num-shards', type=int, default=1, help='Number of utterance shards for multi-GPU preprocessing.')
    parser.add_argument('--shard-index', type=int, default=0, help='0-based shard index for this worker.')
    parser.add_argument('--skip-finalize', action='store_true', help='Only build per-utterance progress records and skip NPZ/metadata finalization.')
    parser.add_argument('--finalize-only', action='store_true', help='Skip processing and only aggregate existing progress records into NPZ/metadata outputs.')
    parser.add_argument('--resume', action='store_true', help='Resume from per-utterance progress files in output-dir/progress.')
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


class AsrPhoneCounter:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.oov_cache = {}
        self.g2p = None
        try:
            from g2p_en import G2p

            self.g2p = G2p()
        except Exception:
            pass

    def count_word(self, word):
        phones = self.lexicon.get(word)
        if phones:
            return len(phones), 'lexicon'
        if word in self.oov_cache:
            return self.oov_cache[word]

        if self.g2p is not None:
            try:
                g2p_phones = [
                    normalize_phone(phone)
                    for phone in self.g2p(word.lower())
                    if re.fullmatch(r'[A-Za-z]+[0-2]?', str(phone))
                ]
                g2p_phones = [phone for phone in g2p_phones if phone]
                if g2p_phones:
                    result = (len(g2p_phones), 'g2p_en')
                    self.oov_cache[word] = result
                    return result
            except Exception:
                self.g2p = None

        letter_count = len(re.sub(r'[^A-Z]', '', normalize_word(word)))
        result = (max(1, int(math.ceil(letter_count / 3.0))), 'character_estimate')
        self.oov_cache[word] = result
        return result

    def count_words(self, words):
        total = 0
        source_counts = Counter()
        for word in words:
            phone_count, source = self.count_word(str(word['text']))
            total += phone_count
            source_counts[source] += 1
        return total, dict(source_counts)


def parse_target_splits(raw_value):
    valid = ['train', 'val', 'test']
    selected = [item.strip() for item in raw_value.split(',') if item.strip()]
    if not selected:
        raise ValueError('target_splits cannot be empty.')
    invalid = [item for item in selected if item not in valid]
    if invalid:
        raise ValueError(f'Unknown split(s) in target_splits: {invalid}')
    deduped = []
    seen = set()
    for item in selected:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def summarize_reason_counts(rows, key, top_k=10):
    counter = Counter()
    for row in rows:
        if not row:
            continue
        counter[str(row.get(key, 'unknown'))] += 1
    return [{'reason': reason, 'count': int(count)} for reason, count in counter.most_common(top_k)]


def preview_phone_vocab(phone_to_frame_id, limit=16):
    preview = []
    for phone in sorted(phone_to_frame_id.keys())[:limit]:
        preview.append({'phone': phone, 'id': int(phone_to_frame_id[phone])})
    return preview


def allow_existing_output_dir(args):
    return bool(args.skip_finalize and args.num_shards > 1 and not args.finalize_only)


def shard_items(split_items, num_shards, shard_index):
    if num_shards <= 1:
        return split_items
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards}), got {shard_index}')
    return [item for idx, item in enumerate(split_items) if idx % num_shards == shard_index]


def longest_common_prefix_len(prev_words, curr_words):
    limit = min(len(prev_words), len(curr_words))
    idx = 0
    while idx < limit and prev_words[idx]['text'] == curr_words[idx]['text']:
        idx += 1
    return idx


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

    # Exact adjacent cycles catch the common Whisper failure mode without
    # treating normal repeated words elsewhere in a long sentence as a loop.
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

    # Long outputs receive an additional periodicity check. Occurrences must
    # be non-overlapping and cover most of the hypothesis, which avoids
    # rejecting legitimate long sentences that reuse common short phrases.
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


def resolve_torch_dtype(device, dtype_name):
    if not isinstance(device, str) or not device.startswith('cuda'):
        return torch.float32
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    if dtype_name == 'float32':
        return torch.float32
    return torch.float16


def build_transformers_asr(asr_model, language, device, torch_dtype_name, max_new_tokens, no_repeat_ngram_size, use_cache):
    if device.startswith('cuda'):
        pipe_device = int(device.split(':', 1)[1]) if ':' in device else 0
    else:
        pipe_device = -1
    torch_dtype = resolve_torch_dtype(device, torch_dtype_name)
    pipe = pipeline(
        'automatic-speech-recognition',
        model=asr_model,
        tokenizer=asr_model,
        feature_extractor=asr_model,
        device=pipe_device,
        torch_dtype=torch_dtype,
    )
    if hasattr(pipe.model, 'generation_config'):
        pipe.model.generation_config.use_cache = bool(use_cache)
    if hasattr(pipe.feature_extractor, 'return_attention_mask'):
        pipe.feature_extractor.return_attention_mask = True
    generate_kwargs = {
        'language': language,
        'task': 'transcribe',
        'max_new_tokens': max_new_tokens,
        'use_cache': bool(use_cache),
    }
    if no_repeat_ngram_size > 0:
        generate_kwargs['no_repeat_ngram_size'] = int(no_repeat_ngram_size)
    return pipe, {
        'return_timestamps': 'word',
        'generate_kwargs': generate_kwargs,
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


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def is_cuda_oom(exc):
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return 'cuda out of memory' in str(exc).lower()


def transcribe_audio_prefixes(audio_prefixes, sample_rate, backend_name, backend_model, backend_kwargs, language, beam_size, best_of, asr_batch_size, asr_min_batch_size, asr_empty_cache):
    if not audio_prefixes:
        return []
    if backend_name == 'transformers':
        items = [
            {
                'raw': np.asarray(audio_prefix, dtype=np.float32),
                'sampling_rate': sample_rate,
            }
            for audio_prefix in audio_prefixes
        ]
        outputs = []
        next_index = 0
        batch_size = max(int(asr_batch_size), 1)
        min_batch_size = max(int(asr_min_batch_size), 1)
        while next_index < len(items):
            cur_batch_size = min(batch_size, len(items) - next_index)
            while True:
                try:
                    results = backend_model(
                        items[next_index : next_index + cur_batch_size],
                        batch_size=cur_batch_size,
                        **backend_kwargs,
                    )
                    if isinstance(results, dict):
                        results = [results]
                    outputs.extend(extract_words_from_transformers(result) for result in results)
                    next_index += cur_batch_size
                    if asr_empty_cache:
                        clear_cuda_cache()
                    break
                except RuntimeError as exc:
                    if not is_cuda_oom(exc):
                        raise
                    clear_cuda_cache()
                    if cur_batch_size <= min_batch_size:
                        raise
                    cur_batch_size = max(min_batch_size, cur_batch_size // 2)
        return outputs

    whisper = backend_kwargs['module']
    outputs = []
    transcribe_parameters = inspect.signature(whisper.transcribe).parameters
    supports_previous_text_control = (
        'condition_on_previous_text' in transcribe_parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in transcribe_parameters.values()
        )
    )
    for audio_prefix in audio_prefixes:
        transcribe_kwargs = {
            'language': language,
            'beam_size': beam_size,
            'best_of': best_of,
        }
        if supports_previous_text_control:
            transcribe_kwargs['condition_on_previous_text'] = False
        result = whisper.transcribe(backend_model, audio_prefix, **transcribe_kwargs)
        outputs.append(extract_words_from_whisper_timestamped(result))
        if asr_empty_cache:
            clear_cuda_cache()
    return outputs


def select_visible_frames(probs, keep_mask, audio_end, frame_step):
    frame_limit = max(int(math.ceil(audio_end / max(frame_step, EPS))), 1)
    frame_mask = np.zeros(len(probs), dtype=bool)
    frame_mask[: min(frame_limit, len(probs))] = True
    visible_mask = keep_mask & frame_mask
    return np.flatnonzero(visible_mask), probs[visible_mask]


def align_gold_utterance(utt_id, audio_path, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict):
    aligned = align_reference_utterance(
        utt_id=utt_id,
        audio_path=audio_path,
        scores=scores,
        charsiu=charsiu,
        sample_rate=sample_rate,
        device=device,
        phone_to_frame_id=phone_to_frame_id,
        phn_dict=phn_dict,
    )
    if 'skip_reason' in aligned:
        return aligned

    gold_words = build_reference_word_records(scores[utt_id])
    probs, audio_duration = audio_logits(audio_path, charsiu.charsiu_processor, charsiu.aligner, sample_rate, device)
    frame_step = audio_duration / max(len(probs), 1)
    keep_mask = build_silence_keep_mask(charsiu, probs)
    word_start_times = {}
    word_end_times = {}
    for phone in aligned['phones']:
        word_id = int(phone['word_id'])
        word_start_times[word_id] = min(word_start_times.get(word_id, phone['start_time']), phone['start_time'])
        word_end_times[word_id] = max(word_end_times.get(word_id, 0.0), phone['end_time'])

    return {
        'utt_id': utt_id,
        'audio_path': str(audio_path),
        'audio_duration': float(audio_duration),
        'frame_step': float(frame_step),
        'probs': probs,
        'keep_mask': keep_mask,
        'gold_phone_segments': aligned['phones'],
        'word_start_times': word_start_times,
        'word_end_times': word_end_times,
        'gold_words': gold_words,
        'utt_scores': aligned['utt_scores'],
    }


def align_split(split_items, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict):
    aligned = []
    skipped = []
    for utt_id, audio_path in tqdm(split_items, desc='gold-align'):
        result = align_gold_utterance(utt_id, audio_path, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict)
        if 'skip_reason' in result:
            skipped.append(result)
        else:
            aligned.append(result)
    return aligned, skipped


def build_chunk_examples_for_utterance(item, asr_backend_name, asr_backend_model, asr_backend_kwargs, args, lexicon, phn_dict, phone_to_frame_id, asr_phone_counter):
    examples = []
    skipped_chunks = []
    chunk_records = []
    utt_id = item['utt_id']
    audio, _ = librosa.load(item['audio_path'], sr=args.sample_rate, mono=True)
    gold_words = item['gold_words']
    final_time = max(item['word_end_times'].values()) if item['word_end_times'] else item['audio_duration']
    prev_visible_words = []
    chunk_specs = []
    audio_prefixes = []

    for chunk_id, commit_time in enumerate(commit_schedule(final_time, args.chunk_sec)):
        is_final = abs(commit_time - final_time) < 1e-5
        audio_end = final_time if is_final else min(final_time, commit_time + args.right_context_sec)
        chunk_specs.append({
            'chunk_id': int(chunk_id),
            'commit_time': float(commit_time),
            'is_final': bool(is_final),
            'audio_end': float(audio_end),
        })
        audio_prefixes.append(audio[: int(max(audio_end, 1e-4) * args.sample_rate)])

    asr_outputs = transcribe_audio_prefixes(
        audio_prefixes=audio_prefixes,
        sample_rate=args.sample_rate,
        backend_name=asr_backend_name,
        backend_model=asr_backend_model,
        backend_kwargs=asr_backend_kwargs,
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        asr_batch_size=args.asr_batch_size,
        asr_min_batch_size=args.asr_min_batch_size,
        asr_empty_cache=args.asr_empty_cache,
    )

    for spec, asr_words in zip(chunk_specs, asr_outputs):
        chunk_id = spec['chunk_id']
        commit_time = spec['commit_time']
        is_final = spec['is_final']
        audio_end = spec['audio_end']
        chunk_base = {
            'utt_id': utt_id,
            'chunk_id': int(chunk_id),
            'audio_end': float(audio_end),
            'commit_time': float(commit_time),
            'is_final': bool(is_final),
            'utt_scores': item['utt_scores'],
            'asr_words': asr_words,
            'segments': [],
            'matched_ratio': 0.0,
            'matched_committed_words': 0,
            'committed_lexicon_words': 0,
            'utt_loss_mask': 0.0,
        }
        if not asr_words:
            reason = 'empty_asr_hypothesis'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({**chunk_base, 'status': 'skipped', 'skip_reason': reason})
            prev_visible_words = []
            continue

        alignable_phone_count = sum(
            len(lexicon.get(word['text'], []))
            for word in asr_words
        )
        asr_phone_count, asr_phone_source_counts = asr_phone_counter.count_words(asr_words)
        reference_phone_count = sum(len(word['phones']) for word in gold_words)
        phone_ratio = float(
            asr_phone_count / max(reference_phone_count, 1)
        )
        long_word_trigger = (
            args.asr_max_words > 0
            and len(asr_words) > args.asr_max_words
        )
        long_phone_trigger = (
            args.asr_max_visible_phones > 0
            and asr_phone_count > args.asr_max_visible_phones
        )
        strict_repeat_check = bool(long_word_trigger or long_phone_trigger)
        length_diagnostics = {
            'asr_word_count': int(len(asr_words)),
            'asr_phone_count': int(asr_phone_count),
            'alignable_phone_count': int(alignable_phone_count),
            'asr_phone_source_counts': asr_phone_source_counts,
            'reference_phone_count': int(reference_phone_count),
            'phone_ratio': phone_ratio,
            'strict_repeat_check': strict_repeat_check,
            'long_word_trigger': bool(long_word_trigger),
            'long_phone_trigger': bool(long_phone_trigger),
        }
        chunk_base.update(length_diagnostics)

        repeat_info = find_repeated_ngram(
            asr_words,
            min_repeats=args.asr_repeat_ngram_min_repeats,
            max_ngram_size=args.asr_repeat_max_ngram_size,
            strict=strict_repeat_check,
            min_coverage=args.asr_repeat_ngram_coverage,
            dominant_token_ratio=args.asr_repeat_token_ratio,
        )
        if repeat_info is not None:
            reason = 'repetitive_asr_hypothesis'
            skipped_chunks.append({
                'utt_id': utt_id,
                'chunk_id': int(chunk_id),
                'reason': reason,
                **length_diagnostics,
                **repeat_info,
            })
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
                'repeat_info': repeat_info,
            })
            prev_visible_words = []
            continue

        exceeds_ratio_limit = (
            args.asr_max_phone_ratio > 0
            and phone_ratio > args.asr_max_phone_ratio
        )
        if exceeds_ratio_limit:
            reason = 'asr_phone_ratio_outlier'
            skipped_chunks.append({
                'utt_id': utt_id,
                'chunk_id': int(chunk_id),
                'reason': reason,
                **length_diagnostics,
            })
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
            })
            prev_visible_words = []
            continue

        timestamp_commit_len = sum(1 for word in asr_words if word['end'] <= commit_time + 1e-6)
        stable_prefix_len = longest_common_prefix_len(prev_visible_words, asr_words) if prev_visible_words else timestamp_commit_len
        committed_len = timestamp_commit_len if is_final else min(timestamp_commit_len, stable_prefix_len)
        prev_visible_words = asr_words
        chunk_base.update({
            'timestamp_commit_len': int(timestamp_commit_len),
            'stable_prefix_len': int(stable_prefix_len),
            'committed_len': int(committed_len),
        })

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
                word_asr_accuracy = 0.0
                loss_ok = False
                if word_match and gold_word is not None and phone_idx < len(gold_word['phone_scores']):
                    phone_score = float(gold_word['phone_scores'][phone_idx])
                    word_accuracy = float(gold_word['accuracy'])
                    word_stress = float(gold_word['stress'])
                    word_total = float(gold_word['total'])
                    loss_ok = committed
                if committed and word_match:
                    word_asr_accuracy = 1.0
                pseudo_records.append({
                    'phone': phone,
                    'phone_id': int(phn_dict[phone]),
                    'phone_score': phone_score,
                    'word_confidence': None if asr_word.get('confidence') is None else float(asr_word['confidence']),
                    'word_local_id': int(visible_word_local_id),
                    'word_accuracy': word_accuracy,
                    'word_stress': word_stress,
                    'word_total': word_total,
                    'word_asr_accuracy': float(word_asr_accuracy),
                    'word_match': bool(word_match),
                    'committed': bool(committed),
                    'word_end': float(asr_word['end']),
                    'phone_loss_ok': bool(loss_ok),
                    'display_word': asr_word['display_text'],
                })
            visible_word_local_id += 1

        if not pseudo_records:
            reason = 'no_alignable_asr_phones'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
                'pseudo_phone_count': 0,
            })
            continue

        kept_indices, kept_probs = select_visible_frames(item['probs'], item['keep_mask'], audio_end, item['frame_step'])
        if kept_probs.shape[0] < len(pseudo_records):
            reason = 'not_enough_frames_for_asr_phones'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
                'pseudo_phone_count': int(len(pseudo_records)),
                'visible_frame_count': int(kept_probs.shape[0]),
            })
            continue

        try:
            phone_ids = [phone_to_frame_id[record['phone']] for record in pseudo_records]
            path = monotonic_align(-np.log(np.clip(kept_probs, EPS, None)), phone_ids)
        except Exception as exc:
            reason = f'asr_alignment_failed:{exc}'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
                'pseudo_phone_count': int(len(pseudo_records)),
                'visible_frame_count': int(kept_probs.shape[0]),
            })
            continue

        segments = []
        for tok_idx, record in enumerate(pseudo_records):
            tok_frames = kept_indices[path == tok_idx]
            if tok_frames.size == 0:
                segments = []
                break
            target_id = phone_to_frame_id[record['phone']]
            base_feature = segment_feature(item['probs'][tok_frames], target_id, item['frame_step']).astype(np.float32)
            asr_confidence = 0.0 if record.get('word_confidence') is None else float(record['word_confidence'])
            feature = np.concatenate([base_feature, np.array([asr_confidence], dtype=np.float32)], axis=0)
            end_time = float((int(tok_frames[-1]) + 1) * item['frame_step'])
            segments.append({
                'feature': feature.astype(np.float32),
                'phone_id': int(record['phone_id']),
                'phone_score': float(record['phone_score']),
                'word_confidence': float(asr_confidence),
                'word_id': int(record['word_local_id']),
                'word_accuracy': float(record['word_accuracy']),
                'word_stress': float(record['word_stress']),
                'word_total': float(record['word_total']),
                'word_asr_accuracy': float(record['word_asr_accuracy']),
                'phone_loss_mask': float(record['phone_loss_ok'] and end_time <= commit_time + 1e-6),
                'word_loss_mask': float(record['phone_loss_ok'] and record['word_end'] <= commit_time + 1e-6),
                'word_asr_loss_mask': 1.0,
                'end_time': end_time,
            })
        if not segments:
            reason = 'empty_asr_phone_segment'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({
                **chunk_base,
                'status': 'skipped',
                'skip_reason': reason,
                'pseudo_phone_count': int(len(pseudo_records)),
                'visible_frame_count': int(kept_probs.shape[0]),
            })
            continue

        matched_ratio = float(matched_committed_words) / float(max(committed_lexicon_words, 1))
        utt_loss_mask = float(is_final and matched_ratio >= args.min_utt_match_ratio)
        phone_loss_count = sum(segment['phone_loss_mask'] for segment in segments)
        word_loss_count = sum(segment['word_loss_mask'] for segment in segments)
        example_record = {
            **chunk_base,
            'segments': segments,
            'matched_ratio': matched_ratio,
            'matched_committed_words': int(matched_committed_words),
            'committed_lexicon_words': int(committed_lexicon_words),
            'utt_loss_mask': utt_loss_mask,
        }
        if phone_loss_count == 0 and word_loss_count == 0 and utt_loss_mask == 0:
            reason = 'no_supervised_tokens_after_matching'
            skipped_chunks.append({'utt_id': utt_id, 'chunk_id': int(chunk_id), 'reason': reason})
            chunk_records.append({**example_record, 'status': 'skipped', 'skip_reason': reason})
            continue
        examples.append(example_record)
        chunk_records.append({**example_record, 'status': 'ok', 'skip_reason': None})

    return examples, skipped_chunks, chunk_records


def build_chunk_examples(aligned_items, asr_backend_name, asr_backend_model, asr_backend_kwargs, args, lexicon, phn_dict, phone_to_frame_id, asr_phone_counter):
    examples = []
    skipped_chunks = []
    chunk_records = []
    for item in tqdm(aligned_items, desc='asr-chunks'):
        cur_examples, cur_skipped_chunks, cur_chunk_records = build_chunk_examples_for_utterance(
            item=item,
            asr_backend_name=asr_backend_name,
            asr_backend_model=asr_backend_model,
            asr_backend_kwargs=asr_backend_kwargs,
            args=args,
            lexicon=lexicon,
            phn_dict=phn_dict,
            phone_to_frame_id=phone_to_frame_id,
            asr_phone_counter=asr_phone_counter,
        )
        examples.extend(cur_examples)
        skipped_chunks.extend(cur_skipped_chunks)
        chunk_records.extend(cur_chunk_records)
    return examples, skipped_chunks, chunk_records


def infer_seq_len(examples):
    return max(len(example['segments']) for example in examples)


def infer_feat_dim(examples):
    feat_dims = {
        int(np.asarray(segment['feature']).shape[-1])
        for example in examples
        for segment in example['segments']
    }
    if not feat_dims:
        raise ValueError('No segment features found while inferring feat_dim.')
    if len(feat_dims) != 1:
        raise ValueError(f'Inconsistent feature dimensions found in examples: {sorted(feat_dims)}')
    return feat_dims.pop()


def train_norm_from_examples(examples):
    feats = []
    for example in examples:
        for segment in example['segments']:
            feats.append(segment['feature'])
    feat = np.stack(feats).astype(np.float32)
    return float(feat.mean()), float(feat.std() + EPS)


def attach_word_weights(examples):
    visible_word_chunk_counts = Counter()
    chunk_word_phone_counts = Counter()

    for example in examples:
        seen_words = set()
        for segment in example['segments']:
            word_id = int(segment['word_id'])
            if word_id < 0:
                continue
            word_key = (str(example['utt_id']), word_id)
            chunk_word_key = (str(example['utt_id']), int(example['chunk_id']), word_id)
            chunk_word_phone_counts[chunk_word_key] += 1
            if word_key not in seen_words:
                visible_word_chunk_counts[word_key] += 1
                seen_words.add(word_key)

    positive_word_weights = []
    positive_word_asr_weights = []
    for example in examples:
        utt_id = str(example['utt_id'])
        chunk_id = int(example['chunk_id'])
        for segment in example['segments']:
            word_id = int(segment['word_id'])
            if word_id < 0:
                segment['word_weight'] = 0.0
                segment['word_asr_weight'] = 0.0
                continue
            word_key = (utt_id, word_id)
            chunk_word_key = (utt_id, chunk_id, word_id)
            chunk_occurrences = max(int(visible_word_chunk_counts[word_key]), 1)
            phone_repetitions = max(int(chunk_word_phone_counts[chunk_word_key]), 1)
            repeat_weight = 1.0 / float(chunk_occurrences * phone_repetitions)
            total_score = float(segment['word_total'])
            low_score_boost = 1.0 if total_score < 0 else 1.0 + max(0.0, 9.0 - total_score) / 2.0
            segment['word_weight'] = float(repeat_weight * low_score_boost)
            segment['word_asr_weight'] = float(repeat_weight)
            if float(segment['word_loss_mask']) > 0:
                positive_word_weights.append(segment['word_weight'])
            if float(segment['word_asr_loss_mask']) > 0:
                positive_word_asr_weights.append(segment['word_asr_weight'])

    word_scale = float(np.mean(positive_word_weights)) if positive_word_weights else 1.0
    word_asr_scale = float(np.mean(positive_word_asr_weights)) if positive_word_asr_weights else 1.0
    word_scale = max(word_scale, EPS)
    word_asr_scale = max(word_asr_scale, EPS)

    for example in examples:
        for segment in example['segments']:
            segment['word_weight'] = float(segment.get('word_weight', 0.0) / word_scale)
            segment['word_asr_weight'] = float(segment.get('word_asr_weight', 0.0) / word_asr_scale)

    return {
        'visible_word_chunk_keys': int(len(visible_word_chunk_counts)),
        'chunk_word_phone_keys': int(len(chunk_word_phone_counts)),
        'word_weight_mean_before_norm': float(word_scale),
        'word_asr_weight_mean_before_norm': float(word_asr_scale),
    }


def build_arrays(examples, seq_len, feat_dim):
    feat_rows = []
    phn_id_rows = []
    phn_score_rows = []
    word_rows = []
    utt_rows = []
    phone_loss_rows = []
    word_loss_rows = []
    word_asr_loss_rows = []
    word_weight_rows = []
    word_asr_weight_rows = []
    utt_loss_rows = []
    is_final_rows = []
    visible_len_rows = []
    manifest = []

    for example in examples:
        cur_feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
        cur_phn_id = np.zeros((seq_len,), dtype=np.int64) - 1
        cur_phn_score = np.zeros((seq_len,), dtype=np.float32) - 1
        cur_word = np.zeros((seq_len, 5), dtype=np.float32) - 1
        cur_phone_loss = np.zeros((seq_len,), dtype=np.float32)
        cur_word_loss = np.zeros((seq_len,), dtype=np.float32)
        cur_word_asr_loss = np.zeros((seq_len,), dtype=np.float32)
        cur_word_weight = np.zeros((seq_len,), dtype=np.float32)
        cur_word_asr_weight = np.zeros((seq_len,), dtype=np.float32)
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
            cur_word[tok_idx, 3] = segment['word_asr_accuracy']
            cur_word[tok_idx, 4] = segment['word_id']
            cur_phone_loss[tok_idx] = segment['phone_loss_mask']
            cur_word_loss[tok_idx] = segment['word_loss_mask']
            cur_word_asr_loss[tok_idx] = segment['word_asr_loss_mask']
            cur_word_weight[tok_idx] = float(segment.get('word_weight', 1.0))
            cur_word_asr_weight[tok_idx] = float(segment.get('word_asr_weight', 1.0))

        feat_rows.append(cur_feat)
        phn_id_rows.append(cur_phn_id)
        phn_score_rows.append(cur_phn_score)
        word_rows.append(cur_word)
        utt_rows.append(cur_utt)
        phone_loss_rows.append(cur_phone_loss)
        word_loss_rows.append(cur_word_loss)
        word_asr_loss_rows.append(cur_word_asr_loss)
        word_weight_rows.append(cur_word_weight)
        word_asr_weight_rows.append(cur_word_asr_weight)
        utt_loss_rows.append(example['utt_loss_mask'])
        is_final_rows.append(int(example['is_final']))
        visible_len_rows.append(len(example['segments']))
        manifest.append({
            'utt_id': example['utt_id'],
            'chunk_id': example['chunk_id'],
            'commit_time': example['commit_time'],
            'audio_end': example['audio_end'],
            'mean_asr_confidence': float(np.mean([segment.get('word_confidence', 0.0) for segment in example['segments']])) if example['segments'] else 0.0,
            'visible_phone_count': int(len(example['segments'])),
            'committed_phone_count': int(cur_phone_loss.sum()),
            'committed_word_phone_count': int(cur_word_loss.sum()),
            'visible_word_phone_count': int(cur_word_asr_loss.sum()),
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
        'word_asr_loss_mask': np.stack(word_asr_loss_rows).astype(np.float32),
        'word_weight': np.stack(word_weight_rows).astype(np.float32),
        'word_asr_weight': np.stack(word_asr_weight_rows).astype(np.float32),
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
        word_asr_loss_mask=arrays['word_asr_loss_mask'],
        word_weight=arrays['word_weight'],
        word_asr_weight=arrays['word_asr_weight'],
        utt_loss_mask=arrays['utt_loss_mask'],
        is_final=arrays['is_final'],
        visible_len=arrays['visible_len'],
    )
    with open(output_dir / f'{prefix}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for row in arrays['manifest']:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def get_progress_split_dir(output_dir, split_name):
    return output_dir / 'progress' / split_name


def get_progress_record_path(output_dir, split_name, utt_id):
    return get_progress_split_dir(output_dir, split_name) / f'{utt_id}.pkl'


def save_progress_record(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_progress_record(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def collect_split_progress(split_name, split_items, output_dir):
    examples = []
    skipped = []
    skipped_chunks = []
    chunk_records = []
    missing_records = []
    for utt_id, _ in split_items:
        record_path = get_progress_record_path(output_dir, split_name, utt_id)
        if not record_path.exists():
            missing_records.append(utt_id)
            continue
        payload = load_progress_record(record_path)
        if payload['status'] == 'skipped':
            skipped.append(payload['skip_record'])
        else:
            payload_chunk_records = payload.get('chunk_records') or []
            if payload_chunk_records:
                chunk_records.extend(payload_chunk_records)
                examples.extend(row for row in payload_chunk_records if row.get('status') == 'ok')
                if payload.get('skipped_chunks'):
                    skipped_chunks.extend(payload['skipped_chunks'])
                else:
                    skipped_chunks.extend(
                        {
                            'utt_id': row['utt_id'],
                            'chunk_id': row['chunk_id'],
                            'reason': row.get('skip_reason', 'unknown'),
                        }
                        for row in payload_chunk_records
                        if row.get('status') != 'ok'
                    )
            else:
                examples.extend(payload['examples'])
                skipped_chunks.extend(payload['skipped_chunks'])

    if missing_records:
        raise RuntimeError(
            f'Missing progress records for split={split_name}: missing={len(missing_records)} sample={missing_records[:5]}'
        )

    return examples, skipped, skipped_chunks, {
        'existing_records': len(split_items) - len(missing_records),
        'total_utterances': len(split_items),
        'cached_chunk_records': len(chunk_records),
    }


def process_split_with_resume(split_name, split_items, scores, charsiu, sample_rate, device, phone_to_frame_id, phn_dict, asr_backend_name, asr_backend_model, asr_backend_kwargs, args, lexicon, asr_phone_counter, output_dir):
    split_dir = get_progress_split_dir(output_dir, split_name)
    split_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    resumed = 0
    for utt_id, audio_path in tqdm(split_items, desc=f'{split_name}-resume'):
        record_path = get_progress_record_path(output_dir, split_name, utt_id)
        if args.resume and record_path.exists():
            resumed += 1
            continue

        aligned = align_gold_utterance(
            utt_id=utt_id,
            audio_path=audio_path,
            scores=scores,
            charsiu=charsiu,
            sample_rate=sample_rate,
            device=device,
            phone_to_frame_id=phone_to_frame_id,
            phn_dict=phn_dict,
        )
        if 'skip_reason' in aligned:
            save_progress_record(record_path, {
                'status': 'skipped',
                'utt_id': utt_id,
                'skip_record': aligned,
                'examples': [],
                'skipped_chunks': [],
                'chunk_records': [],
            })
            processed += 1
            continue

        examples, skipped_chunks, chunk_records = build_chunk_examples_for_utterance(
            item=aligned,
            asr_backend_name=asr_backend_name,
            asr_backend_model=asr_backend_model,
            asr_backend_kwargs=asr_backend_kwargs,
            args=args,
            lexicon=lexicon,
            phn_dict=phn_dict,
            phone_to_frame_id=phone_to_frame_id,
            asr_phone_counter=asr_phone_counter,
        )
        save_progress_record(record_path, {
            'status': 'ok',
            'utt_id': utt_id,
            'skip_record': None,
            'examples': examples,
            'skipped_chunks': skipped_chunks,
            'chunk_records': chunk_records,
        })
        processed += 1

    examples, skipped, skipped_chunks, _ = collect_split_progress(split_name, split_items, output_dir)
    return examples, skipped, skipped_chunks, {
        'processed_now': processed,
        'resumed_existing': resumed,
        'total_utterances': len(split_items),
    }


def main():
    args = get_args()
    target_splits = parse_target_splits(args.target_splits)
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError('--shard-index must satisfy 0 <= shard-index < num-shards')
    if args.finalize_only and args.overwrite:
        raise ValueError('--finalize-only and --overwrite are mutually exclusive.')
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite are mutually exclusive.')
    existing_entries = []
    if output_dir.exists():
        existing_entries = sorted(path.name for path in output_dir.iterdir())
    print(
        '[gopt_data-start] '
        f'skip_finalize={args.skip_finalize} '
        f'finalize_only={args.finalize_only} '
        f'num_shards={args.num_shards} '
        f'shard_index={args.shard_index} '
        f'resume={args.resume} '
        f'overwrite={args.overwrite} '
        f'allow_existing={allow_existing_output_dir(args)} '
        f'output_dir_exists={output_dir.exists()} '
        f'output_dir_entries={existing_entries[:8]}',
        flush=True,
    )
    if output_dir.exists():
        if args.overwrite and not args.finalize_only:
            shutil.rmtree(output_dir)
        elif (
            not args.resume
            and not args.finalize_only
            and not allow_existing_output_dir(args)
            and any(output_dir.iterdir())
        ):
            raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild or --resume to continue.')
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
    split_items_map = {
        'train': train_items,
        'val': val_items,
        'test': test_items,
    }
    utt_ids = [utt_id for split in target_splits for utt_id, _ in split_items_map[split] if utt_id in scores]
    phn_dict = build_phone_vocab(scores, utt_ids)
    lexicon = build_word_lexicon(scores, utt_ids)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    charsiu = None
    phone_to_frame_id = {}
    id2label = {}
    silence_ids = []
    phone_vocab_preview = []
    if not args.finalize_only:
        asr_phone_counter = AsrPhoneCounter(lexicon)
        charsiu = load_official_charsiu_aligner(
            model_name=args.aligner_model,
            device=device,
            sample_rate=args.sample_rate,
            sil_threshold=args.min_sil_frames,
            lang=args.charsiu_lang,
            charsiu_src_dir=args.charsiu_src_dir,
        )
        aligner_model = charsiu.aligner
        phone_to_frame_id, id2label, silence_ids = build_model_phone_map(charsiu)
        phone_vocab_preview = preview_phone_vocab(phone_to_frame_id)
        print(
            '[gopt_data] '
            f'build_charsiu_seq_data_file={build_model_phone_map.__code__.co_filename} '
            f'charsiu_src_dir={getattr(charsiu, "_custom_charsiu_src_dir", None)} '
            f'aligner_model={args.aligner_model} '
            f'phone_vocab_size={len(phone_to_frame_id)} '
            f'phone_vocab_preview={phone_vocab_preview}',
            flush=True,
        )

    split_progress = {}
    if not args.finalize_only:
        if args.timestamp_backend == 'transformers':
            asr_backend_model, asr_backend_kwargs = build_transformers_asr(
                args.asr_model,
                args.language,
                device,
                args.asr_torch_dtype,
                args.asr_max_new_tokens,
                args.asr_no_repeat_ngram_size,
                args.asr_use_cache,
            )
            asr_backend_name = 'transformers'
        else:
            asr_model, whisper_module = build_whisper_timestamped_asr(args.asr_model, device)
            asr_backend_model = asr_model
            asr_backend_kwargs = {'module': whisper_module}
            asr_backend_name = 'whisper_timestamped'

        for split_name in target_splits:
            shard_split = shard_items(split_items_map[split_name], args.num_shards, args.shard_index)
            _, _, _, split_progress[split_name] = process_split_with_resume(
                split_name=split_name,
                split_items=shard_split,
                scores=scores,
                charsiu=charsiu,
                sample_rate=args.sample_rate,
                device=device,
                phone_to_frame_id=phone_to_frame_id,
                phn_dict=phn_dict,
                asr_backend_name=asr_backend_name,
                asr_backend_model=asr_backend_model,
                asr_backend_kwargs=asr_backend_kwargs,
                args=args,
                lexicon=lexicon,
                asr_phone_counter=asr_phone_counter,
                output_dir=output_dir,
            )

        if args.skip_finalize:
            progress_summary = {
                split_name: {
                    **split_progress[split_name],
                    'num_shards': int(args.num_shards),
                    'shard_index': int(args.shard_index),
                    'shard_utterances': int(len(shard_items(split_items_map[split_name], args.num_shards, args.shard_index))),
                }
                for split_name in target_splits
            }
            with open(output_dir / f'worker_progress_shard{args.shard_index:02d}.json', 'w', encoding='utf-8') as handle:
                json.dump(progress_summary, handle, ensure_ascii=False, indent=2)
            return

    collected = {}
    for split_name in ['train', 'val', 'test']:
        examples, skipped, skipped_chunks, progress_meta = collect_split_progress(
            split_name,
            split_items_map[split_name],
            output_dir,
        )
        collected[split_name] = {
            'examples': examples,
            'skipped': skipped,
            'skipped_chunks': skipped_chunks,
            'progress': progress_meta,
        }

    train_examples = collected['train']['examples']
    skipped_train = collected['train']['skipped']
    skipped_train_chunks = collected['train']['skipped_chunks']
    train_progress = collected['train']['progress']
    val_examples = collected['val']['examples']
    skipped_val = collected['val']['skipped']
    skipped_val_chunks = collected['val']['skipped_chunks']
    val_progress = collected['val']['progress']
    test_examples = collected['test']['examples']
    skipped_test = collected['test']['skipped']
    skipped_test_chunks = collected['test']['skipped_chunks']
    test_progress = collected['test']['progress']

    if not train_examples or not val_examples or not test_examples:
        failure_diag = {
            'train_examples': int(len(train_examples)),
            'val_examples': int(len(val_examples)),
            'test_examples': int(len(test_examples)),
            'build_charsiu_seq_data_file': str(build_model_phone_map.__code__.co_filename),
            'charsiu_src_dir': str(getattr(charsiu, '_custom_charsiu_src_dir', None)),
            'aligner_model': str(args.aligner_model),
            'phone_vocab_size': int(len(phone_to_frame_id)),
            'phone_vocab_preview': phone_vocab_preview,
            'skipped_train_utterances': int(len(skipped_train)),
            'skipped_val_utterances': int(len(skipped_val)),
            'skipped_test_utterances': int(len(skipped_test)),
            'skipped_train_chunks': int(len(skipped_train_chunks)),
            'skipped_val_chunks': int(len(skipped_val_chunks)),
            'skipped_test_chunks': int(len(skipped_test_chunks)),
            'train_skip_summary': summarize_reason_counts(skipped_train, 'skip_reason'),
            'val_skip_summary': summarize_reason_counts(skipped_val, 'skip_reason'),
            'test_skip_summary': summarize_reason_counts(skipped_test, 'skip_reason'),
            'train_chunk_skip_summary': summarize_reason_counts(skipped_train_chunks, 'reason'),
            'val_chunk_skip_summary': summarize_reason_counts(skipped_val_chunks, 'reason'),
            'test_chunk_skip_summary': summarize_reason_counts(skipped_test_chunks, 'reason'),
            'progress': {
                'train': train_progress,
                'val': val_progress,
                'test': test_progress,
            },
        }
        with open(output_dir / 'failure_diagnostics.json', 'w', encoding='utf-8') as handle:
            json.dump(failure_diag, handle, ensure_ascii=False, indent=2)
        raise ValueError(
            'No ASR-driven streaming chunks were generated. '
            f'train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}. '
            f'See {output_dir / "failure_diagnostics.json"} for skip statistics.'
        )

    train_weighting = attach_word_weights(train_examples)
    val_weighting = attach_word_weights(val_examples)
    test_weighting = attach_word_weights(test_examples)

    seq_len = infer_seq_len(train_examples + val_examples + test_examples)
    feat_dim = infer_feat_dim(train_examples + val_examples + test_examples)
    train_norm_mean, train_norm_std = train_norm_from_examples(train_examples)
    train_arrays = build_arrays(train_examples, seq_len, feat_dim)
    val_arrays = build_arrays(val_examples, seq_len, feat_dim)
    test_arrays = build_arrays(test_examples, seq_len, feat_dim)

    save_chunk_split('train', train_arrays, output_dir)
    save_chunk_split('val', val_arrays, output_dir)
    save_chunk_split('test', test_arrays, output_dir)

    metadata = {
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'split_meta': split_meta,
        'resume_enabled': bool(args.resume),
        'aligner_model': args.aligner_model,
        'charsiu_src_dir': args.charsiu_src_dir,
        'charsiu_lang': args.charsiu_lang,
        'asr_model': args.asr_model,
        'timestamp_backend': args.timestamp_backend,
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
            'condition_on_previous_text': False,
            'length_policy': 'word_count and absolute phone_count only enable strict repetition checks; non-repetitive long hypotheses are retained',
            'hard_rejection_policy': 'empty hypothesis, detected repetition, or ASR/reference phone ratio above max_phone_ratio',
            'phone_count_policy': 'reference lexicon first, g2p_en for OOV words, then character-length estimate if G2P is unavailable',
        },
        'chunk_sec': float(args.chunk_sec),
        'right_context_sec': float(args.right_context_sec),
        'seq_len': int(seq_len),
        'feat_dim': int(feat_dim),
        'asr_confidence_feat_dim': 1,
        'phn_dict': phn_dict,
        'phn_num': int(len(phn_dict) + 1),
        'train_norm_mean': float(train_norm_mean),
        'train_norm_std': float(train_norm_std),
        'num_frame_labels': int(getattr(getattr(charsiu, 'aligner', None), 'config', None).num_labels) if charsiu is not None else int(max(feat_dim - 5, 0)),
        'frame_id2label': {str(key): str(value) for key, value in id2label.items()} if id2label else {},
        'silence_ids': [int(x) for x in silence_ids] if silence_ids else [],
        'lexicon_size': int(len(lexicon)),
        'train_chunks': int(train_arrays['feat'].shape[0]),
        'val_chunks': int(val_arrays['feat'].shape[0]),
        'test_chunks': int(test_arrays['feat'].shape[0]),
        'train_final_chunks': int(train_arrays['is_final'].sum()),
        'val_final_chunks': int(val_arrays['is_final'].sum()),
        'test_final_chunks': int(test_arrays['is_final'].sum()),
        'word_weighting': {
            'formula': 'word_weight=(1/max(chunk_occurrences*phone_repetitions,1))*(1+max(0,9-word_total)/2), word_asr_weight=1/max(chunk_occurrences*phone_repetitions,1), both normalized to positive-mean=1 within each split',
            'train': train_weighting,
            'val': val_weighting,
            'test': test_weighting,
        },
        'skipped_train': skipped_train,
        'skipped_val': skipped_val,
        'skipped_test': skipped_test,
        'skipped_train_chunks': skipped_train_chunks,
        'skipped_val_chunks': skipped_val_chunks,
        'skipped_test_chunks': skipped_test_chunks,
        'progress': {
            'train': train_progress,
            'val': val_progress,
            'test': test_progress,
        },
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
