import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'prep_data'))

from models import PCNStreamingScorer
from build_charsiu_seq_data import audio_logits, build_model_phone_map, build_silence_keep_mask, load_official_charsiu_aligner
from build_streaming_charsiu_data import commit_schedule
from build_streaming_pcn_gopt_data import (
    PhoneMapper,
    WhisperNBestGenerator,
    align_hypothesis_with_charsiu,
    build_acoustic_evidence,
    build_pcn_from_hypotheses,
    build_stateful_commit_masks,
    compute_prosody,
    pcn_stats,
    top_hyp_word_ids_for_slots,
    validate_pcn,
)


def get_args():
    parser = argparse.ArgumentParser(description='Run no-GT stateful streaming PCN inference on one WAV.')
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--whisper-model', type=str, default='openai/whisper-base')
    parser.add_argument('--aligner-model', type=str, default='charsiu/en_w2v2_tiny_fc_10ms')
    parser.add_argument('--charsiu-src-dir', type=str, default=None)
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--chunk-sec', type=float, default=0.64)
    parser.add_argument('--right-context-sec', type=float, default=0.16)
    parser.add_argument('--nbest', type=int, default=5)
    parser.add_argument('--beam-size', type=int, default=8)
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--mock-frontends', action='store_true', help='For smoke tests only; does not read GT.')
    parser.add_argument('--limit-chunks', type=int, default=0)
    return parser.parse_args()


def pad(src, seq_len, fill_value=0.0):
    src = np.asarray(src)
    out = np.zeros((seq_len,) + src.shape[1:], dtype=src.dtype)
    if fill_value != 0.0:
        out[...] = fill_value
    n = min(seq_len, src.shape[0])
    out[:n] = src[:n]
    return out


def load_metadata(config_path):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    data_dir = config.get('data_dir')
    if data_dir and (Path(data_dir) / 'metadata.json').exists():
        metadata = json.loads((Path(data_dir) / 'metadata.json').read_text(encoding='utf-8'))
    else:
        metadata = config
    return config, metadata


def load_model(config, metadata, checkpoint, device):
    model_args = config.get('args', {})
    model = PCNStreamingScorer(
        phone_dim=int(metadata['phone_dim']),
        seq_len=int(metadata['seq_len']),
        prosody_dim=int(config.get('prosody_dim', len(metadata.get('prosody', [])) or 14)),
        embed_dim=int(model_args.get('embed_dim', 40)),
        num_heads=int(model_args.get('heads', 2)),
        depth=int(model_args.get('depth', 2)),
        gru_dim=int(model_args.get('gru_dim', 32)),
        main_context_tokens=int(model_args.get('main_context_tokens', 16)),
        utt_pooling_head=str(config.get('utt_pooling_head', model_args.get('utt_pooling_head', 'gru'))),
        fusion_mode=str(config.get('fusion_mode', model_args.get('fusion_mode', 'scalar_gate'))),
        slot_prosody_dim=int(config.get('slot_prosody_dim', 0)),
        stress_branch=str(config.get('stress_branch', model_args.get('stress_branch', 'none'))),
        stress_grad_scale=float(config.get('stress_grad_scale', model_args.get('stress_grad_scale', 0.2))),
    )
    weights = torch.load(checkpoint, map_location=device)
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def mock_hypotheses(audio_end, phn_dict):
    words = ['toy', 'prefix']
    step = float(audio_end) / len(words)
    return [
        {
            'rank': 0,
            'text': 'toy prefix',
            'words': words,
            'logprob': -0.1,
            'sequence_score': -0.2,
            'length_normalized_sequence_score': -0.1,
            'token_ids': [1, 2],
            'token_logprobs': [-0.1, -0.1],
            'token_confidences': [0.9, 0.9],
            'word_token_ranges': [[0, 1], [1, 2]],
            'word_logprobs': [-0.1, -0.1],
            'word_confidences': [0.9, 0.9],
            'word_timestamps': [
                {'word': words[0], 'start': 0.0, 'end': step, 'source': 'mock'},
                {'word': words[1], 'start': step, 'end': float(audio_end), 'source': 'mock'},
            ],
            'timestamp_source': 'mock',
        }
    ]


def make_item(args, audio, metadata, charsiu=None):
    if args.mock_frontends:
        frame_count = max(2, int(np.ceil(len(audio) / args.sample_rate / 0.02)))
        phone_dim = int(metadata['phone_dim'])
        probs = np.ones((frame_count, phone_dim), dtype=np.float32) / float(phone_dim)
        return {
            'audio_path': str(args.wav),
            'probs': probs,
            'keep_mask': np.ones((frame_count,), dtype=bool),
            'frame_step': 0.02,
            'audio_duration': len(audio) / args.sample_rate,
        }, {f'P{i}': i for i in range(phone_dim - 1)}
    probs, duration = audio_logits(args.wav, charsiu.charsiu_processor, charsiu.aligner, args.sample_rate, args.device)
    keep_mask = build_silence_keep_mask(charsiu, probs)
    return {
        'audio_path': str(args.wav),
        'probs': probs,
        'keep_mask': keep_mask,
        'frame_step': duration / max(probs.shape[0], 1),
        'audio_duration': duration,
    }, None


def tensor_batch(example, metadata, config, device):
    seq_len = int(metadata['seq_len'])
    prosody_mean = np.asarray(config.get('prosody_norm_mean', np.zeros((example['prosody'].shape[0],), dtype=np.float32)), dtype=np.float32)
    prosody_std = np.asarray(config.get('prosody_norm_std', np.ones((example['prosody'].shape[0],), dtype=np.float32)), dtype=np.float32)
    prosody = (example['prosody'] - prosody_mean) / np.clip(prosody_std, 1e-6, None)
    batch = {
        'cn_post': pad(example['cn_post'], seq_len)[None],
        'cn_stats': pad(example['cn_stats'], seq_len)[None],
        'acoustic_post': pad(example['acoustic_post'], seq_len)[None],
        'acoustic_stats': pad(example['acoustic_stats'], seq_len)[None],
        'prosody': prosody.reshape(1, -1),
        'pcn_word_id': pad(example['pcn_word_id'].reshape(-1, 1), seq_len, fill_value=-1).reshape(1, seq_len),
        'cumulative_commit_mask': pad(example['cumulative_commit_mask'].reshape(-1, 1), seq_len).reshape(1, seq_len),
        'new_commit_mask': pad(example['new_commit_mask'].reshape(-1, 1), seq_len).reshape(1, seq_len),
        'visible_len': np.array([min(example['cn_post'].shape[0], seq_len)], dtype=np.int64),
    }
    slot_prosody_dim = int(config.get('slot_prosody_dim', 0))
    if slot_prosody_dim > 0:
        if 'slot_prosody' in example:
            slot_mean = np.asarray(config.get('slot_prosody_norm_mean', np.zeros((slot_prosody_dim,), dtype=np.float32)), dtype=np.float32)
            slot_std = np.asarray(config.get('slot_prosody_norm_std', np.ones((slot_prosody_dim,), dtype=np.float32)), dtype=np.float32)
            slot_prosody = (example['slot_prosody'] - slot_mean.reshape(1, -1)) / np.clip(slot_std.reshape(1, -1), 1e-6, None)
            batch['slot_prosody'] = pad(slot_prosody, seq_len)[None]
        else:
            batch['slot_prosody'] = np.zeros((1, seq_len, slot_prosody_dim), dtype=np.float32)
    return {
        key: torch.tensor(value, dtype=torch.long if key in {'pcn_word_id', 'visible_len'} else torch.float32, device=device)
        for key, value in batch.items()
    }


def main():
    args = get_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    config, metadata = load_metadata(args.config)
    phn_dict = metadata.get('phn_dict') or {f'P{i}': i for i in range(int(metadata['phone_dim']) - 1)}
    model = load_model(config, metadata, args.checkpoint, device)
    audio, _ = librosa.load(args.wav, sr=args.sample_rate, mono=True)
    duration = len(audio) / args.sample_rate

    charsiu = None
    phone_to_frame_id = {phone: idx for phone, idx in phn_dict.items()}
    if not args.mock_frontends:
        charsiu = load_official_charsiu_aligner(
            model_name=args.aligner_model,
            device=args.device,
            sample_rate=args.sample_rate,
            sil_threshold=4,
            lang='en',
            charsiu_src_dir=args.charsiu_src_dir,
        )
        phone_to_frame_id, _, _ = build_model_phone_map(charsiu)
    item, mock_phn_dict = make_item(args, audio, metadata, charsiu=charsiu)
    if mock_phn_dict is not None:
        phn_dict = mock_phn_dict
    phone_mapper = PhoneMapper({}, phn_dict)
    asr_generator = None if args.mock_frontends else WhisperNBestGenerator(
        args.whisper_model,
        args.language,
        args.device,
        args.nbest,
        args.beam_size,
        max_new_tokens=128,
        no_repeat_ngram_size=0,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state = None
    commit_state = None
    prev_top = []
    with args.output_jsonl.open('w', encoding='utf-8') as handle:
        for chunk_id, commit_time in enumerate(commit_schedule(duration, args.chunk_sec)):
            if args.limit_chunks > 0 and chunk_id >= args.limit_chunks:
                break
            started = time.perf_counter()
            is_final = abs(commit_time - duration) < 1e-5
            audio_end = duration if is_final else min(duration, commit_time + args.right_context_sec)
            audio_prefix = audio[: int(max(audio_end, 1e-4) * args.sample_rate)]
            if args.mock_frontends:
                hypotheses = mock_hypotheses(audio_end, phn_dict)
            else:
                hypotheses = asr_generator.generate(audio_prefix, args.sample_rate, audio_end)
            hypotheses = [
                align_hypothesis_with_charsiu(item, dict(row), phone_mapper, phn_dict, phone_to_frame_id, audio_end)
                for row in hypotheses
            ]
            pcn = build_pcn_from_hypotheses(hypotheses, phone_mapper, phn_dict)
            validate_pcn(pcn)
            pcn_word_id = top_hyp_word_ids_for_slots(pcn)
            cn_stats, prefix_stability = pcn_stats(pcn['cn_post'], pcn['top_phone_ids'], prev_top, pcn['eps_index'])
            prev_top = list(pcn['top_phone_ids'])
            acoustic_post, acoustic_stats, visible_frames = build_acoustic_evidence(
                item,
                pcn['cn_post'],
                pcn['top_phone_ids'],
                phone_to_frame_id,
                phn_dict,
                audio_end,
            )
            cumulative_mask, new_mask, mapped_old_slot, diagnostics, commit_state = build_stateful_commit_masks(
                commit_state,
                pcn,
                pcn_word_id,
                hypotheses[0].get('word_timestamps', []) if hypotheses else [],
                commit_time,
                audio_end,
                is_final,
            )
            prosody = compute_prosody(
                audio_prefix,
                args.sample_rate,
                audio_end,
                len(hypotheses[0].get('words', [])) if hypotheses else 0,
                sum(1 for item_phone in pcn['top_phone_ids'] if item_phone is not None),
            )
            example = {
                'cn_post': pcn['cn_post'].astype(np.float32),
                'cn_stats': cn_stats.astype(np.float32),
                'acoustic_post': acoustic_post.astype(np.float32),
                'acoustic_stats': acoustic_stats.astype(np.float32),
                'prosody': prosody.astype(np.float32),
                'pcn_word_id': pcn_word_id.astype(np.int32),
                'cumulative_commit_mask': cumulative_mask.astype(np.float32),
                'new_commit_mask': new_mask.astype(np.float32),
            }
            batch = tensor_batch(example, metadata, config, device)
            out = model.stream_step(batch, prev_state=state)
            state = out['next_state']
            new_word_ids = sorted(set(int(x) for x in pcn_word_id[new_mask > 0].tolist() if int(x) >= 0))
            row = {
                'chunk_id': int(chunk_id),
                'audio_end': float(audio_end),
                'committed_hypotheses': hypotheses[0].get('words', []) if hypotheses else [],
                'new_committed_words': [
                    hypotheses[0]['words'][word_idx]
                    for word_idx in new_word_ids
                    if hypotheses and word_idx < len(hypotheses[0].get('words', []))
                ],
                'pcn_summary': {
                    'slot_count': int(pcn['cn_post'].shape[0]),
                    'epsilon_mean': float(np.mean(pcn['cn_post'][:, pcn['eps_index']])),
                    'prefix_stability': float(prefix_stability),
                    'visible_frame_count': int(visible_frames),
                    'timestamp_source': [hyp.get('timestamp_source', '') for hyp in hypotheses],
                },
                'phone_scores': out['phone_score'].squeeze(0).squeeze(-1).detach().cpu().tolist(),
                'word_scores': out['word_scores'].squeeze(0).detach().cpu().tolist(),
                'utterance_scores': out['utt_scores'].squeeze(0).detach().cpu().tolist(),
                'confidence': out['confidence'].squeeze(0).squeeze(-1).detach().cpu().tolist(),
                'abstention_probability': out['abstention_probability'].squeeze(0).squeeze(-1).detach().cpu().tolist(),
                'sentence_state_updated': bool(np.sum(new_mask) > 0),
                'commit_alignment_diagnostics': diagnostics,
                'mapped_old_slot': mapped_old_slot.tolist(),
                'process_time_sec': time.perf_counter() - started,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
