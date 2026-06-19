import argparse
import gc
import json
import math
import os
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
    find_repeated_ngram,
    lcs_align_words,
    select_visible_frames,
)
from build_streaming_charsiu_data import commit_schedule


EPS_TOKEN = '<eps>'


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


def softmax_1d(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32)
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
        texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        if getattr(output, 'sequences_scores', None) is not None:
            logprobs = output.sequences_scores.detach().float().cpu().numpy().tolist()
        else:
            logprobs = [0.0 for _ in texts]

        rows = []
        for rank, (text, logprob) in enumerate(zip(texts, logprobs)):
            words = normalize_text_to_words(text)
            rows.append({
                'rank': int(rank),
                'text': ' '.join(words).lower(),
                'words': words,
                'logprob': float(logprob),
                'word_timestamps': estimate_word_timestamps(words, audio_end),
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return rows


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


def align_token_sequences(ref_tokens, hyp_tokens, mismatch_cost=1.0):
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


def insert_slot(slots, insert_at, eps_index, prior_mass):
    slot = {'counts': Counter()}
    if prior_mass > 0:
        slot['counts'][eps_index] += float(prior_mass)
    slots.insert(insert_at, slot)
    return slot


def build_pcn_from_hypotheses(hypotheses, phone_mapper, phn_dict):
    eps_index = len(phn_dict)
    phone_dim = len(phn_dict) + 1
    hyp_logprobs = [float(row['logprob']) for row in hypotheses]
    hyp_weights = softmax_1d(hyp_logprobs)
    id_to_phone = {idx: phone for phone, idx in phn_dict.items()}

    hyp_phone_rows = []
    for row, weight in zip(hypotheses, hyp_weights):
        phones, phone_to_word, source_counts = phone_mapper.words_to_phone_sequence(row['words'])
        phone_ids = [phn_dict[phone] for phone in phones if phone in phn_dict]
        hyp_phone_rows.append({
            'phones': phones,
            'phone_ids': phone_ids,
            'phone_to_word': phone_to_word,
            'weight': float(weight),
            'source_counts': source_counts,
        })

    slots = []
    total_mass = 0.0
    for hyp_idx, hyp in enumerate(hyp_phone_rows):
        phone_ids = hyp['phone_ids']
        weight = float(hyp['weight'])
        if hyp_idx == 0:
            for phone_idx in phone_ids:
                slot = {'counts': Counter()}
                slot['counts'][phone_idx] += weight
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
        pairs = align_token_sequences(consensus_phone_ids, phone_ids)
        cursor = 0
        matched_slots = set()
        for ref_pos, hyp_pos in pairs:
            if ref_pos is None:
                insert_at = consensus_slot_ids[cursor] if cursor < len(consensus_slot_ids) else len(slots)
                slot = insert_slot(slots, insert_at, eps_index, total_mass)
                slot['counts'][phone_ids[hyp_pos]] += weight
                cursor += 1
            else:
                slot_idx = consensus_slot_ids[ref_pos]
                matched_slots.add(slot_idx)
                cursor = ref_pos + 1
                if hyp_pos is None:
                    slots[slot_idx]['counts'][eps_index] += weight
                else:
                    slots[slot_idx]['counts'][phone_ids[hyp_pos]] += weight
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
    for slot_idx, slot in enumerate(slots):
        slot_mass = sum(slot['counts'].values())
        if slot_mass < total_mass:
            slot['counts'][eps_index] += total_mass - slot_mass
        for phone_idx, mass in slot['counts'].items():
            cn_post[slot_idx, int(phone_idx)] = float(mass) / max(float(total_mass), EPS)
        cn_post[slot_idx] /= np.clip(cn_post[slot_idx].sum(), EPS, None)
        top_phone_ids.append(slot_top_phone(slot['counts'], eps_index))

    return {
        'cn_post': cn_post,
        'top_phone_ids': top_phone_ids,
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


def build_targets(item, cn_post, top_phone_ids, hypotheses, phn_dict, commit_time):
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
    commit_mask = np.zeros((len(top_phone_ids),), dtype=np.float32)

    for slot_idx, gt_idx in slot_to_gt.items():
        gt = gt_rows[gt_idx]
        gt_phone_id = phn_dict.get(gt['phone'])
        if gt_phone_id is None:
            continue
        committed = gt['end_time'] <= commit_time + 1e-6
        gt_word_seen = gt['word_id'] in seen_word_ids
        top_phone = slot_tokens[slot_idx]
        exact = top_phone == gt['phone']
        gt_posterior = float(cn_post[slot_idx, gt_phone_id])
        entropy = entropy_np(cn_post[slot_idx])
        entropy_norm = entropy / math.log(max(cn_post.shape[-1], 2))
        commit_mask[slot_idx] = float(committed)
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

    return {
        'phone_target': phone_target,
        'word_target': word_target,
        'asr_correct_target': asr_correct_target,
        'uncertainty_target': uncertainty_target,
        'soft_label_weight': soft_label_weight,
        'commit_mask': commit_mask,
        'gt_word_seen_count': int(len(seen_word_ids)),
    }


def build_examples_for_utterance(item, asr_generator, phone_mapper, phn_dict, phone_to_frame_id, args):
    examples = []
    skipped_chunks = []
    utt_id = item['utt_id']
    audio, _ = librosa.load(item['audio_path'], sr=args.sample_rate, mono=True)
    final_time = max(item['word_end_times'].values()) if item['word_end_times'] else item['audio_duration']
    prev_top_phone_ids = []

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
        pcn = build_pcn_from_hypotheses(hypotheses, phone_mapper, phn_dict)
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
        targets = build_targets(
            item=item,
            cn_post=pcn['cn_post'],
            top_phone_ids=pcn['top_phone_ids'],
            hypotheses=hypotheses,
            phn_dict=phn_dict,
            commit_time=commit_time,
        )
        examples.append({
            'utt_id': utt_id,
            'wav_path': item['audio_path'],
            'chunk_id': int(chunk_id),
            'commit_time': float(commit_time),
            'audio_end': float(audio_end),
            'is_final': bool(is_final),
            'cn_post': pcn['cn_post'].astype(np.float32),
            'cn_stats': cn_stats_arr.astype(np.float32),
            'acoustic_post': acoustic_post.astype(np.float32),
            'acoustic_stats': acoustic_stats.astype(np.float32),
            'prosody': prosody.astype(np.float32),
            'pcn_word_id': pcn_word_id.astype(np.int32),
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
            'prefix_stability': float(prefix_stability),
            'visible_frame_count': int(visible_frame_count),
            'top_phone_count': int(top_phone_count),
            'coverage_ratio': float(audio_end / max(final_time, EPS)),
        })
    return examples, skipped_chunks


def build_split(split_name, split_items, scores, charsiu, asr_generator, phone_mapper, phn_dict, phone_to_frame_id, args):
    examples = []
    skipped = []
    skipped_chunks = []
    for utt_id, audio_path in tqdm(split_items, desc=f'{split_name}-pcn'):
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
            )
            examples.extend(cur_examples)
            skipped_chunks.extend(cur_skipped_chunks)
        except Exception as exc:
            skipped.append({'utt_id': utt_id, 'skip_reason': f'pcn_build_failed:{exc}'})
    return examples, skipped, skipped_chunks


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
        'teacher_prefix_utt_score': [],
        'teacher_final_utt_score': [],
        'teacher_utt_mask': [],
        'teacher_word_score': [],
        'teacher_word_mask': [],
        'coverage_ratio': [],
        'visible_len': [],
        'is_final': [],
    }
    manifest = []
    for example in examples:
        visible_len = int(example['cn_post'].shape[0])
        arrays['cn_post'].append(pad_2d(example['cn_post'], seq_len))
        arrays['cn_stats'].append(pad_2d(example['cn_stats'], seq_len))
        arrays['acoustic_post'].append(pad_2d(example['acoustic_post'], seq_len))
        arrays['acoustic_stats'].append(pad_2d(example['acoustic_stats'], seq_len))
        arrays['prosody'].append(example['prosody'].reshape(prosody_dim))
        arrays['pcn_word_id'].append(pad_2d(example['pcn_word_id'].reshape(-1, 1), seq_len, fill_value=-1).reshape(seq_len))
        arrays['phone_target'].append(pad_2d(example['phone_target'], seq_len, fill_value=-1.0))
        arrays['word_target'].append(pad_2d(example['word_target'], seq_len, fill_value=-1.0))
        arrays['utt_target'].append(example['utt_target'])
        arrays['asr_correct_target'].append(pad_2d(example['asr_correct_target'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['uncertainty_target'].append(pad_2d(example['uncertainty_target'].reshape(-1, 1), seq_len, fill_value=1.0).reshape(seq_len))
        arrays['soft_label_weight'].append(pad_2d(example['soft_label_weight'].reshape(-1, 1), seq_len).reshape(seq_len))
        arrays['commit_mask'].append(pad_2d(example['commit_mask'].reshape(-1, 1), seq_len).reshape(seq_len))
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
        manifest.append({
            'utt_id': example['utt_id'],
            'wav_path': example.get('wav_path', ''),
            'chunk_id': int(example['chunk_id']),
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
            'raw_hypothesis_count': int(example.get('raw_hypothesis_count', len(example['hypotheses']))),
            'kept_hypothesis_count': int(len(example['hypotheses'])),
            'filtered_hypotheses': example.get('filtered_hypotheses', []),
            'hyp_text': [row['text'] for row in example['hypotheses']],
            'hyp_logprob': [float(row['logprob']) for row in example['hypotheses']],
            'hyp_weight': [float(weight) for weight in example['hyp_weights']],
            'word_timestamps': [row['word_timestamps'] for row in example['hypotheses']],
        })

    return {
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
        'teacher_prefix_utt_score': np.stack(arrays['teacher_prefix_utt_score']).astype(np.float32),
        'teacher_final_utt_score': np.stack(arrays['teacher_final_utt_score']).astype(np.float32),
        'teacher_utt_mask': np.asarray(arrays['teacher_utt_mask'], dtype=np.float32),
        'teacher_word_score': np.stack(arrays['teacher_word_score']).astype(np.float32),
        'teacher_word_mask': np.stack(arrays['teacher_word_mask']).astype(np.float32),
        'coverage_ratio': np.asarray(arrays['coverage_ratio'], dtype=np.float32),
        'visible_len': np.asarray(arrays['visible_len'], dtype=np.int32),
        'is_final': np.asarray(arrays['is_final'], dtype=np.int8),
        'manifest': manifest,
    }


def save_split(split_name, arrays, output_dir):
    np.savez_compressed(
        output_dir / f'{split_name}_chunks.npz',
        cn_post=arrays['cn_post'],
        cn_stats=arrays['cn_stats'],
        acoustic_post=arrays['acoustic_post'],
        acoustic_stats=arrays['acoustic_stats'],
        prosody=arrays['prosody'],
        pcn_word_id=arrays['pcn_word_id'],
        phone_target=arrays['phone_target'],
        word_target=arrays['word_target'],
        utt_target=arrays['utt_target'],
        asr_correct_target=arrays['asr_correct_target'],
        uncertainty_target=arrays['uncertainty_target'],
        soft_label_weight=arrays['soft_label_weight'],
        commit_mask=arrays['commit_mask'],
        teacher_prefix_utt_score=arrays['teacher_prefix_utt_score'],
        teacher_final_utt_score=arrays['teacher_final_utt_score'],
        teacher_utt_mask=arrays['teacher_utt_mask'],
        teacher_word_score=arrays['teacher_word_score'],
        teacher_word_mask=arrays['teacher_word_mask'],
        coverage_ratio=arrays['coverage_ratio'],
        visible_len=arrays['visible_len'],
        is_final=arrays['is_final'],
    )
    with open(output_dir / f'{split_name}_manifest.jsonl', 'w', encoding='utf-8') as handle:
        for row in arrays['manifest']:
            handle.write(json.dumps(safe_json(row), ensure_ascii=False) + '\n')


def main():
    args = get_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    target_splits = parse_target_splits(args.target_splits)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'{output_dir} already exists. Use --overwrite to rebuild.')
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

    split_examples = {}
    skipped = {}
    all_examples = []
    for split_name in target_splits:
        examples, cur_skipped, cur_skipped_chunks = build_split(
            split_name=split_name,
            split_items=split_items[split_name],
            scores=scores,
            charsiu=charsiu,
            asr_generator=asr_generator,
            phone_mapper=phone_mapper,
            phn_dict=phn_dict,
            phone_to_frame_id=phone_to_frame_id,
            args=args,
        )
        split_examples[split_name] = examples
        skipped[split_name] = cur_skipped
        skipped[f'{split_name}_chunks'] = cur_skipped_chunks
        all_examples.extend(examples)

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
    for split_name in target_splits:
        arrays = build_arrays(split_examples[split_name], seq_len, phone_dim, len(prosody_names))
        save_split(split_name, arrays, output_dir)

    metadata = {
        'schema': 'streaming_pcn_gopt_v1',
        'dataset_root': str(dataset_root),
        'scores_json': str(scores_path),
        'split_meta': split_meta,
        'target_splits': target_splits,
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
        'targets': [
            'phone_target',
            'word_target',
            'utt_target',
            'asr_correct_target',
            'uncertainty_target',
            'soft_label_weight',
            'commit_mask',
            'pcn_word_id',
            'visible_len',
            'is_final',
            'teacher_prefix_utt_score',
            'teacher_final_utt_score',
            'teacher_word_score',
            'teacher_word_mask',
            'coverage_ratio',
        ],
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
        'word_timestamp_policy': 'N-best generation returns sequence scores. Word timestamps are stored as duration-proportional estimates; PCN/acoustic alignment uses Charsiu frames.',
        'skipped': skipped,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(safe_json(metadata), handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
