import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

try:
    import torch
    from models import PCNStreamingScorer
    from train_streaming_pcn import (
        PCNUtteranceDataset,
        move_batch,
        reset_state_where_needed,
        restore_invalid_state,
        slice_chunk,
        valid_slot_mask,
    )
except Exception:  # pragma: no cover - metric-only unit tests do not require torch.
    torch = None
    PCNStreamingScorer = None
    PCNUtteranceDataset = None


SENTENCE_DIMS = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
WORD_DIMS = ['accuracy', 'stress', 'total']
COVERAGE_NODES = [0.25, 0.50, 0.75, 0.90, 1.00]
DEFAULT_EXP_ROOT = REPO_ROOT / 'exp' / 'pcn_extra_correct_multipa_20260710'


def parse_args():
    parser = argparse.ArgumentParser(description='Unified paper evaluation for PCN and baseline streaming scoring models.')
    parser.add_argument('--exp-root', type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'paper_experiments')
    parser.add_argument('--models', type=str, default='MultiPA,GOPT-open-base,GOPT-open-medium,GOPT-closed-oracle,Experiment-H')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--test-list', type=Path, default=None, help='Optional JSON file with a shared utt_id list.')
    parser.add_argument('--limit-utterances', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=1, help='Reserved; PCN paper eval streams utterances one at a time.')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--bootstrap-samples', type=int, default=500)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-missing-baselines', action='store_true')
    return parser.parse_args()


def finite_array(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def pcc(pred, target):
    pred = finite_array(pred)
    target = finite_array(target)
    n = min(pred.size, target.size)
    if n < 2:
        return 0.0
    pred = pred[:n]
    target = target[:n]
    if np.std(pred) <= 1e-12 or np.std(target) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def mse(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return 0.0
    return float(np.mean((pred[mask] - target[mask]) ** 2))


def mae(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def ece(conf, target, bins=10):
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(conf) & np.isfinite(target)
    conf = np.clip(conf[mask], 0.0, 1.0)
    target = np.clip(target[mask], 0.0, 1.0)
    if conf.size == 0:
        return 0.0
    total = float(conf.size)
    out = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf >= lo) & (conf <= hi if hi >= 1.0 else conf < hi)
        if np.any(in_bin):
            out += float(np.sum(in_bin)) / total * abs(float(np.mean(conf[in_bin])) - float(np.mean(target[in_bin])))
    return float(out)


def brier(conf, target):
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(conf) & np.isfinite(target)
    if not np.any(mask):
        return 0.0
    return float(np.mean((np.clip(conf[mask], 0.0, 1.0) - np.clip(target[mask], 0.0, 1.0)) ** 2))


def aurc(conf, error):
    conf = np.asarray(conf, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    mask = np.isfinite(conf) & np.isfinite(error)
    conf = conf[mask]
    error = error[mask]
    if conf.size == 0:
        return 0.0
    order = np.argsort(-conf)
    risk = np.cumsum(error[order]) / np.arange(1, error.size + 1)
    coverage = np.arange(1, error.size + 1) / error.size
    return float(np.trapz(risk, coverage))


def bootstrap_ci_by_speaker(rows, value_fn, seed=1337, samples=500):
    grouped = defaultdict(list)
    for row in rows:
        speaker = str(row.get('speaker_id') or 'UNKNOWN')
        grouped[speaker].append(row)
    speakers = sorted(grouped)
    if len(speakers) < 2:
        val = value_fn(rows)
        return [val, val]
    rng = random.Random(seed)
    vals = []
    for _ in range(max(int(samples), 1)):
        sampled = []
        for _ in speakers:
            sampled.extend(grouped[rng.choice(speakers)])
        vals.append(value_fn(sampled))
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        val = value_fn(rows)
        return [val, val]
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return [float(lo), float(hi)]


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_manifest(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def speaker_from_manifest(row):
    if row.get('speaker_id'):
        return str(row['speaker_id']).upper()
    wav_path = str(row.get('wav_path') or row.get('audio_path') or '')
    parts = Path(wav_path).parts
    for part in reversed(parts):
        if part.upper().startswith('SPEAKER'):
            return part.upper()
    return 'UNKNOWN'


def unique_test_list(data_dir, split):
    rows = read_manifest(Path(data_dir) / f'{split}_manifest.jsonl')
    seen = []
    used = set()
    for row in rows:
        utt_id = str(row.get('utt_id'))
        if utt_id and utt_id not in used:
            used.add(utt_id)
            seen.append(utt_id)
    return seen


def no_overwrite_path(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 1000):
        candidate = path.with_name(f'{stem}.{idx}{suffix}')
        if not candidate.exists():
            return candidate
    raise FileExistsError(f'Could not find unused output path for {path}')


def load_pcn_model(exp_dir, config, metadata, device):
    if torch is None or PCNStreamingScorer is None:
        raise RuntimeError('PyTorch/model imports are unavailable; cannot evaluate PCN checkpoints.')
    model_args = config.get('args', {})
    model = PCNStreamingScorer(
        phone_dim=int(config.get('phone_dim', metadata['phone_dim'])),
        seq_len=int(config.get('seq_len', metadata['seq_len'])),
        prosody_dim=int(config.get('prosody_dim', len(metadata.get('prosody', [])) or 14)),
        embed_dim=int(model_args.get('embed_dim', 40)),
        num_heads=int(model_args.get('heads', 2)),
        depth=int(model_args.get('depth', 2)),
        gru_dim=int(model_args.get('gru_dim', 32)),
        main_context_tokens=int(model_args.get('main_context_tokens', 16)),
        use_state_projection=bool(config.get('uses_state_projection', False)),
        utt_pooling_head=str(config.get('utt_pooling_head', model_args.get('utt_pooling_head', 'gru'))),
        fusion_mode=str(config.get('fusion_mode', model_args.get('fusion_mode', 'scalar_gate'))),
        slot_prosody_dim=int(config.get('slot_prosody_dim', 0)),
        stress_branch=str(config.get('stress_branch', model_args.get('stress_branch', 'none'))),
        stress_grad_scale=float(config.get('stress_grad_scale', model_args.get('stress_grad_scale', 0.2))),
        use_acoustic=bool(config.get('use_acoustic', not model_args.get('disable_acoustic', False))),
        use_prosody=bool(config.get('use_prosody', not model_args.get('disable_prosody', False))),
        use_uncertainty_stats=bool(config.get('use_uncertainty_stats', not model_args.get('disable_uncertainty_stats', False))),
    )
    checkpoint = Path(exp_dir) / 'models' / 'best_audio_model.pth'
    if not checkpoint.exists():
        checkpoint = Path(exp_dir) / 'last_checkpoint.pt'
    weights = torch.load(checkpoint, map_location=device)
    if isinstance(weights, dict) and 'model' in weights:
        weights = weights['model']
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def build_pcn_dataset(config, split):
    data_dir = Path(config['data_dir'])
    with np.load(data_dir / 'train_chunks.npz', allow_pickle=True) as train_raw:
        prosody_mean = train_raw['prosody'].mean(axis=0).astype(np.float32)
        prosody_std = train_raw['prosody'].std(axis=0).astype(np.float32)
        if 'slot_prosody' in train_raw.files:
            slot_mean = train_raw['slot_prosody'].mean(axis=(0, 1)).astype(np.float32)
            slot_std = train_raw['slot_prosody'].std(axis=(0, 1)).astype(np.float32)
        else:
            slot_mean = None
            slot_std = None
    return PCNUtteranceDataset(split, data_dir, prosody_mean, prosody_std, slot_mean, slot_std)


def sample_to_batch(sample, device):
    out = {'chunk_valid_mask': torch.ones((1, sample['cn_post'].shape[0]), dtype=torch.float32, device=device)}
    for key, value in sample.items():
        dtype = torch.long if np.asarray(value).dtype.kind in {'i', 'u'} else torch.float32
        out[key] = torch.tensor(value[None], dtype=dtype, device=device)
    return out


def apply_input_ablation(chunk, config):
    model_args = config.get('args', {})
    mode = str(config.get('pcn_input_mode', model_args.get('pcn_input_mode', 'posterior')))
    if mode == 'top1_onehot':
        cn_post = chunk['cn_post']
        top_idx = cn_post.argmax(dim=-1, keepdim=True)
        chunk = dict(chunk)
        chunk['cn_post'] = torch.zeros_like(cn_post).scatter_(-1, top_idx, 1.0)
    return chunk


def mean_confidence(conf, valid_mask):
    denom = valid_mask.sum().clamp_min(1.0)
    return float((conf * valid_mask).sum().detach().cpu().item() / denom.detach().cpu().item())


def list_visible(values, visible_len, scale=1.0):
    arr = np.asarray(values)
    return (arr[: int(visible_len)] * scale).astype(float).tolist()


def evaluate_pcn_experiment(model_name, exp_dir, split, shared_utts, device, output_jsonl, limit_utterances=0, state_update_mode='incremental'):
    config = read_json(Path(exp_dir) / 'config.json', {})
    data_dir = Path(config['data_dir'])
    metadata = read_json(data_dir / 'metadata.json', {})
    manifest = read_manifest(data_dir / f'{split}_manifest.jsonl')
    dataset = build_pcn_dataset(config, split)
    model = load_pcn_model(exp_dir, config, metadata, device)

    utt_to_group = {}
    for group_idx, indices in enumerate(dataset.groups):
        if not indices:
            continue
        utt_id = str(manifest[indices[0]].get('utt_id'))
        utt_to_group[utt_id] = group_idx
    selected_utts = [utt for utt in shared_utts if utt in utt_to_group]
    if limit_utterances > 0:
        selected_utts = selected_utts[:limit_utterances]

    records = []
    with torch.no_grad(), output_jsonl.open('w', encoding='utf-8') as handle:
        for utt_id in selected_utts:
            group_idx = utt_to_group[utt_id]
            indices = dataset.groups[group_idx]
            sample = dataset[group_idx]
            batch = sample_to_batch(sample, device)
            state = None
            prev_phone = None
            prev_word = None
            for local_chunk_idx, source_index in enumerate(indices):
                row_meta = manifest[source_index]
                chunk = apply_input_ablation(slice_chunk(batch, local_chunk_idx), config)
                cur_valid = batch['chunk_valid_mask'][:, local_chunk_idx]
                state = reset_state_where_needed(state, chunk['state_reset'])
                if state_update_mode == 'replay_all_committed':
                    model_prev_state = None
                    model_new_commit_mask = chunk['cumulative_commit_mask']
                else:
                    model_prev_state = state
                    model_new_commit_mask = chunk['new_commit_mask']
                out = model(
                    cn_post=chunk['cn_post'],
                    cn_stats=chunk['cn_stats'],
                    acoustic_post=chunk['acoustic_post'],
                    acoustic_stats=chunk['acoustic_stats'],
                    prosody=chunk['prosody'],
                    visible_len=chunk['visible_len'],
                    cumulative_commit_mask=chunk['cumulative_commit_mask'],
                    new_commit_mask=model_new_commit_mask,
                    word_ids=chunk['pcn_word_id'],
                    slot_prosody=chunk.get('slot_prosody'),
                    prev_state=model_prev_state,
                    detach_next_state=True,
                )
                if state_update_mode == 'replay_all_committed':
                    state = None
                else:
                    state = restore_invalid_state(out['next_state'], state, cur_valid)
                valid_mask = valid_slot_mask(chunk)
                visible_len = int(chunk['visible_len'][0].detach().cpu().item())
                slot_conf = out['confidence'][0, :visible_len, 0].detach().cpu().numpy()
                abst = out['abstention_probability'][0, :visible_len, 0].detach().cpu().numpy()
                gate_tensor = out.get('reliability_gate')
                if gate_tensor is not None:
                    gate_arr = gate_tensor[0, :visible_len].detach().cpu().numpy()
                    gate_slot = gate_arr.mean(axis=-1) if gate_arr.ndim == 2 else gate_arr
                else:
                    gate_slot = np.zeros((visible_len,), dtype=np.float32)
                phone_pred = out['phone_score'][0, :visible_len, 0].detach().cpu().numpy() * 5.0
                word_pred = out['word_scores'][0, :visible_len].detach().cpu().numpy() * 5.0
                utt_pred = out['utt_scores'][0].detach().cpu().numpy() * 5.0
                phone_target = chunk['phone_score_target'][0, :visible_len].detach().cpu().numpy() * 5.0
                word_target = chunk['word_score_target'][0, :visible_len].detach().cpu().numpy() * 5.0
                utt_target = chunk['utt_target'][0].detach().cpu().numpy() * 5.0
                committed = chunk['cumulative_commit_mask'][0, :visible_len].detach().cpu().numpy() > 0
                valid_slots = valid_mask[0, :visible_len].detach().cpu().numpy() > 0
                phone_revision = 0.0
                word_revision = 0.0
                if prev_phone is not None:
                    n = min(prev_phone.shape[0], phone_pred.shape[0])
                    if n:
                        phone_revision = float(np.mean(np.abs(phone_pred[:n] - prev_phone[:n]) > 0.25))
                        word_revision = float(np.mean(np.abs(word_pred[:n] - prev_word[:n]).mean(axis=-1) > 0.25))
                prev_phone = phone_pred.copy()
                prev_word = word_pred.copy()
                rec = {
                    'utt_id': utt_id,
                    'speaker_id': speaker_from_manifest(row_meta),
                    'duration_sec': float(max(row_meta.get('audio_end', 0.0), row_meta.get('commit_time', 0.0))),
                    'model': model_name,
                    'seed': config.get('args', {}).get('seed'),
                    'chunk_id': int(row_meta.get('chunk_id', local_chunk_idx)),
                    'commit_time': float(row_meta.get('commit_time', 0.0)),
                    'audio_end': float(row_meta.get('audio_end', 0.0)),
                    'coverage': float(row_meta.get('coverage_ratio', 0.0)),
                    'is_final': bool(row_meta.get('is_final', False)),
                    'condition': 'GT-free inference; GT labels used only for metrics',
                    'state_update_mode': state_update_mode,
                    'targets': {
                        'sentence': dict(zip(SENTENCE_DIMS, utt_target.astype(float).tolist())),
                        'phone': phone_target.astype(float).tolist(),
                        'word': [dict(zip(WORD_DIMS, values.astype(float).tolist())) for values in word_target],
                    },
                    'predictions': {
                        'sentence': dict(zip(SENTENCE_DIMS, utt_pred.astype(float).tolist())),
                        'phone': phone_pred.astype(float).tolist(),
                        'word': [dict(zip(WORD_DIMS, values.astype(float).tolist())) for values in word_pred],
                    },
                    'confidence': {'mean': float(np.mean(slot_conf)) if slot_conf.size else 0.0, 'slot': slot_conf.astype(float).tolist()},
                    'abstention': {'mean': float(np.mean(abst)) if abst.size else 0.0, 'slot': abst.astype(float).tolist()},
                    'gate': {
                        'mean': float(np.mean(gate_slot)) if gate_slot.size else 0.0,
                        'std': float(np.std(gate_slot)) if gate_slot.size else 0.0,
                        'slot': np.asarray(gate_slot, dtype=np.float64).astype(float).tolist(),
                    },
                    'input_stats': {
                        'pcn_entropy': chunk['cn_stats'][0, :visible_len, 1].detach().cpu().numpy().astype(float).tolist(),
                        'acoustic_entropy': chunk['acoustic_stats'][0, :visible_len, 0].detach().cpu().numpy().astype(float).tolist(),
                        'pcn_acoustic_js': chunk['acoustic_stats'][0, :visible_len, 3].detach().cpu().numpy().astype(float).tolist(),
                    },
                    'visible_len': visible_len,
                    'new_committed_word_count': int(chunk['new_committed_word_count'][0].detach().cpu().item()),
                    'cumulative_committed_word_count': int(chunk['cumulative_committed_word_count'][0].detach().cpu().item()),
                    'valid_slot_mask': valid_slots.astype(int).tolist(),
                    'committed_slot_mask': committed.astype(int).tolist(),
                    'confidence_target': chunk['confidence_target'][0, :visible_len].detach().cpu().numpy().astype(float).tolist(),
                    'phone_revision_rate_from_prev': phone_revision,
                    'word_revision_rate_from_prev': word_revision,
                    'raw_manifest': {
                        'wav_path': row_meta.get('wav_path', ''),
                        'hyp_text': row_meta.get('hyp_text', [])[:3],
                    },
                }
                records.append(rec)
                handle.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return records


def candidate_external_files(exp_root, model_name):
    repo_eval = REPO_ROOT / 'downloads' / 'custom-gopt-252' / 'eval'
    globs = {
        'MultiPA': ['**/*multipa*/*.jsonl', '**/*multipa*.jsonl'],
        'GOPT-open-base': ['**/*open*base*/*.jsonl', '**/*gopt*base*/*.jsonl'],
        'GOPT-open-medium': ['**/*open*medium*/*.jsonl', '**/*gopt*medium*/*.jsonl'],
        'GOPT-closed-oracle': ['**/*closed*oracle*/*.jsonl', '**/*oracle*gt*time*/*.jsonl'],
    }
    out = []
    for base in [repo_eval, exp_root]:
        if not base.exists():
            continue
        for pattern in globs.get(model_name, []):
            out.extend(sorted(path for path in base.glob(pattern) if path.is_file()))
    return out


def normalize_external_predictions(model_name, source_jsonl, shared_utts, output_jsonl, limit_utterances=0):
    selected = set(shared_utts[:limit_utterances] if limit_utterances > 0 else shared_utts)
    rows = []
    with Path(source_jsonl).open('r', encoding='utf-8') as src, output_jsonl.open('w', encoding='utf-8') as dst:
        for line in src:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw_utt_id = str(raw.get('utt_id') or raw.get('utt') or raw.get('id') or '')
            source_utt_id = str(raw.get('source_utt_id') or raw.get('base_utt_id') or raw_utt_id)
            if source_utt_id == raw_utt_id and '_c' in source_utt_id:
                source_utt_id = source_utt_id.rsplit('_c', 1)[0]
            if selected and source_utt_id not in selected:
                continue
            scores = raw.get('predictions') or raw.get('prediction') or {}
            if not scores and isinstance(raw.get('scores'), dict):
                scores = {'sentence': raw.get('scores')}
            targets = raw.get('targets') or raw.get('labels') or {}
            if not targets and isinstance(raw.get('target_scores'), dict):
                targets = {'sentence': raw.get('target_scores')}
            if raw.get('word_scores') is not None:
                scores = dict(scores)
                scores['word_raw'] = raw.get('word_scores')
            row = {
                'utt_id': source_utt_id,
                'raw_utt_id': raw_utt_id,
                'speaker_id': raw.get('speaker_id') or speaker_from_manifest(raw),
                'duration_sec': raw.get('duration_sec') or raw.get('duration') or raw.get('audio_end'),
                'model': model_name,
                'seed': raw.get('seed'),
                'chunk_id': int(raw.get('chunk_id', 0) or 0),
                'commit_time': raw.get('commit_time'),
                'audio_end': raw.get('audio_end'),
                'coverage': raw.get('coverage') or raw.get('coverage_ratio') or (1.0 if raw.get('is_final', True) else None),
                'is_final': bool(raw.get('is_final', True)),
                'condition': 'GT-oracle' if model_name == 'GOPT-closed-oracle' else 'GT-free',
                'targets': targets,
                'predictions': scores,
                'confidence': raw.get('confidence', {}),
                'abstention': raw.get('abstention', raw.get('abstention_probability', {})),
                'visible_len': raw.get('visible_len'),
                'new_committed_word_count': raw.get('new_committed_word_count'),
                'cumulative_committed_word_count': raw.get('cumulative_committed_word_count'),
                'source_file': str(source_jsonl),
            }
            rows.append(row)
            dst.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows


def sentence_value(row, source, dim):
    obj = row.get(source, {})
    sent = obj.get('sentence', obj) if isinstance(obj, dict) else {}
    if isinstance(sent, dict):
        return sent.get(dim)
    if isinstance(sent, list) and dim in SENTENCE_DIMS:
        idx = SENTENCE_DIMS.index(dim)
        if idx < len(sent):
            return sent[idx]
    return None


def collect_metric_rows(records, seed=1337, bootstrap_samples=500):
    rows = []
    final_records = [row for row in records if row.get('is_final')]
    for dim in SENTENCE_DIMS:
        if dim == 'completeness':
            continue
        scoped = []
        pred = []
        target = []
        for row in final_records:
            p = sentence_value(row, 'predictions', dim)
            t = sentence_value(row, 'targets', dim)
            if p is not None and t is not None:
                scoped.append(row)
                pred.append(float(p))
                target.append(float(t))
        if pred:
            value_fn = lambda rs, d=dim: pcc(
                [sentence_value(r, 'predictions', d) for r in rs],
                [sentence_value(r, 'targets', d) for r in rs],
            )
            ci = bootstrap_ci_by_speaker(scoped, value_fn, seed=seed, samples=bootstrap_samples)
            rows.append({
                'level': 'sentence',
                'metric': dim,
                'coverage': 'final',
                'n': len(pred),
                'pcc': pcc(pred, target),
                'mse': mse(pred, target),
                'mae': mae(pred, target),
                'pcc_ci_low': ci[0],
                'pcc_ci_high': ci[1],
            })

    slot_pred = {'phone': [], 'word_accuracy': [], 'word_stress': [], 'word_total': []}
    slot_target = {'phone': [], 'word_accuracy': [], 'word_stress': [], 'word_total': []}
    slot_conf = []
    slot_conf_target = []
    slot_error = []
    for row in records:
        valid = np.asarray(row.get('valid_slot_mask') or [], dtype=bool)
        committed = np.asarray(row.get('committed_slot_mask') or valid, dtype=bool)
        keep = valid & committed if valid.size else committed
        conf = row.get('confidence', {})
        conf_slots = conf.get('slot', []) if isinstance(conf, dict) else []
        targets = row.get('targets', {})
        preds = row.get('predictions', {})
        phone_t = np.asarray(targets.get('phone', []), dtype=np.float64)
        phone_p = np.asarray(preds.get('phone', []), dtype=np.float64)
        if keep.size and phone_t.size and phone_p.size:
            n = min(keep.size, phone_t.size, phone_p.size)
            mask = keep[:n] & np.isfinite(phone_t[:n]) & (phone_t[:n] >= 0)
            slot_pred['phone'].extend(phone_p[:n][mask].tolist())
            slot_target['phone'].extend(phone_t[:n][mask].tolist())
            if conf_slots:
                conf_arr = np.asarray(conf_slots[:n], dtype=np.float64)
                slot_conf.extend(conf_arr[mask].tolist())
                err = np.clip(np.abs(phone_p[:n][mask] - phone_t[:n][mask]) / 5.0, 0.0, 1.0)
                slot_error.extend(err.tolist())
        word_t = targets.get('word', [])
        word_p = preds.get('word', [])
        for idx, name in enumerate(WORD_DIMS):
            key = f'word_{name}'
            cur_p = []
            cur_t = []
            for pos, (p_row, t_row) in enumerate(zip(word_p, word_t)):
                if keep.size and pos < keep.size and not keep[pos]:
                    continue
                if isinstance(p_row, dict):
                    p_val = p_row.get(name)
                else:
                    p_val = p_row[idx] if idx < len(p_row) else None
                if isinstance(t_row, dict):
                    t_val = t_row.get(name)
                else:
                    t_val = t_row[idx] if idx < len(t_row) else None
                if p_val is not None and t_val is not None and float(t_val) >= 0:
                    cur_p.append(float(p_val))
                    cur_t.append(float(t_val))
            slot_pred[key].extend(cur_p)
            slot_target[key].extend(cur_t)
        slot_conf_target.extend(row.get('confidence_target', []))
    for key, pred in slot_pred.items():
        target = slot_target[key]
        if pred:
            level, metric = ('phone', 'phone') if key == 'phone' else ('word', key.replace('word_', ''))
            rows.append({
                'level': level,
                'metric': metric,
                'coverage': 'slots',
                'n': len(pred),
                'pcc': pcc(pred, target),
                'mse': mse(pred, target),
                'mae': mae(pred, target),
                'pcc_ci_low': '',
                'pcc_ci_high': '',
            })
    if slot_conf:
        target = np.asarray(slot_conf_target[:len(slot_conf)], dtype=np.float64)
        rows.append({
            'level': 'calibration',
            'metric': 'confidence',
            'coverage': 'slots',
            'n': len(slot_conf),
            'pcc': '',
            'mse': '',
            'mae': '',
            'pcc_ci_low': '',
            'pcc_ci_high': '',
            'ece': ece(slot_conf, target),
            'brier': brier(slot_conf, target),
            'aurc': aurc(slot_conf, slot_error[:len(slot_conf)]),
        })
    return rows


def coverage_metrics(records):
    rows = []
    by_utt = defaultdict(list)
    for row in records:
        by_utt[row.get('utt_id')].append(row)
    for utt_rows in by_utt.values():
        utt_rows.sort(key=lambda row: (float(row.get('coverage') or 0.0), int(row.get('chunk_id') or 0)))
    for dim in SENTENCE_DIMS:
        if dim == 'completeness':
            continue
        for node in COVERAGE_NODES:
            prefix_pred = []
            prefix_target = []
            prefix_final_pred = []
            for utt_id, utt_rows in by_utt.items():
                if not utt_rows:
                    continue
                final = next((row for row in reversed(utt_rows) if row.get('is_final')), utt_rows[-1])
                prefix = min(utt_rows, key=lambda row: abs(float(row.get('coverage') or 0.0) - node))
                p = sentence_value(prefix, 'predictions', dim)
                t = sentence_value(prefix, 'targets', dim)
                f = sentence_value(final, 'predictions', dim)
                if p is not None and t is not None:
                    prefix_pred.append(float(p))
                    prefix_target.append(float(t))
                if p is not None and f is not None:
                    prefix_final_pred.append((float(p), float(f)))
            final_p = [x[0] for x in prefix_final_pred]
            final_t = [x[1] for x in prefix_final_pred]
            rows.append({
                'level': 'sentence_prefix',
                'metric': dim,
                'coverage': node,
                'n': len(prefix_pred),
                'prefix_human_pcc': pcc(prefix_pred, prefix_target),
                'prefix_human_mae': mae(prefix_pred, prefix_target),
                'prefix_final_mae': mae(final_p, final_t),
            })
    revision_phone = []
    revision_word = []
    first_stable = []
    adjacent_delta = []
    for utt_rows in by_utt.values():
        final = next((row for row in reversed(utt_rows) if row.get('is_final')), utt_rows[-1])
        final_total = sentence_value(final, 'predictions', 'total')
        stable_idx = None
        prev_total = None
        for idx, row in enumerate(utt_rows):
            total = sentence_value(row, 'predictions', 'total')
            if total is not None and prev_total is not None:
                adjacent_delta.append(abs(float(total) - float(prev_total)))
            prev_total = total
            if final_total is not None and total is not None and abs(float(total) - float(final_total)) <= 0.25 and stable_idx is None:
                stable_idx = idx
            if row.get('phone_revision_rate_from_prev') is not None:
                revision_phone.append(float(row['phone_revision_rate_from_prev']))
            if row.get('word_revision_rate_from_prev') is not None:
                revision_word.append(float(row['word_revision_rate_from_prev']))
        if stable_idx is not None:
            first_stable.append(stable_idx)
    rows.append({
        'level': 'streaming',
        'metric': 'revision_stability',
        'coverage': 'all',
        'n': len(records),
        'phone_revision_rate': float(np.mean(revision_phone)) if revision_phone else 0.0,
        'word_revision_rate': float(np.mean(revision_word)) if revision_word else 0.0,
        'adjacent_score_delta': float(np.mean(adjacent_delta)) if adjacent_delta else 0.0,
        'first_stable_chunk': float(np.mean(first_stable)) if first_stable else 0.0,
    })
    return rows


def write_metrics_csv(path, rows, overwrite=False):
    path = no_overwrite_path(path, overwrite=overwrite)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ['model']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    run_id = args.run_id or time.strftime('%Y%m%d_%H%M%S')
    pred_dir = args.output_root / 'predictions' / run_id
    metric_dir = args.output_root / 'metrics' / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    h_exp = args.exp_root / 'stress_runs' / 'H_stress_weighted_G'
    h_config = read_json(h_exp / 'config.json', {})
    if not h_config:
        h_exp = args.exp_root / 'stress_runs' / 'H_stress_weighted_G_corrected'
        h_config = read_json(h_exp / 'config.json', {})
    if args.test_list and args.test_list.exists():
        shared_utts = json.loads(args.test_list.read_text(encoding='utf-8'))
    elif h_config.get('data_dir'):
        shared_utts = unique_test_list(h_config['data_dir'], args.split)
    else:
        shared_utts = []
    if args.limit_utterances > 0:
        shared_utts = shared_utts[:args.limit_utterances]
    (metric_dir / 'shared_test_utterances.json').write_text(json.dumps(shared_utts, ensure_ascii=False, indent=2), encoding='utf-8')

    requested = [item.strip() for item in args.models.split(',') if item.strip()]
    all_metric_rows = []
    summary = {}
    for model_name in requested:
        out_jsonl = no_overwrite_path(pred_dir / f'{model_name}.predictions.jsonl', overwrite=args.overwrite)
        records = []
        if model_name in {'Experiment-H', 'H_stress_weighted_G'}:
            if not h_config:
                if not args.allow_missing_baselines:
                    raise FileNotFoundError(f'Missing Experiment H config under {args.exp_root}/stress_runs/H_stress_weighted_G')
                summary[model_name] = {'status': 'missing'}
                continue
            if torch is None:
                raise RuntimeError('PyTorch is required to evaluate Experiment H.')
            device = torch.device(args.device)
            records = evaluate_pcn_experiment('Experiment-H', h_exp, args.split, shared_utts, device, out_jsonl, args.limit_utterances)
        else:
            candidates = candidate_external_files(args.exp_root, model_name)
            if not candidates:
                if not args.allow_missing_baselines:
                    raise FileNotFoundError(f'No existing prediction JSONL found for {model_name}. Use --allow-missing-baselines to record missing.')
                summary[model_name] = {'status': 'missing'}
                continue
            records = normalize_external_predictions(model_name, candidates[0], shared_utts, out_jsonl, args.limit_utterances)
        metric_rows = collect_metric_rows(records, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
        metric_rows.extend(coverage_metrics(records))
        for row in metric_rows:
            row['model'] = model_name
        all_metric_rows.extend(metric_rows)
        summary[model_name] = {'status': 'evaluated', 'predictions': str(out_jsonl), 'records': len(records)}

    metrics_csv = write_metrics_csv(metric_dir / 'paper_metrics.csv', all_metric_rows, overwrite=args.overwrite)
    summary_path = metric_dir / 'summary.json'
    summary_path.write_text(json.dumps({'models': summary, 'metrics_csv': str(metrics_csv)}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'predictions_dir': str(pred_dir), 'metrics_dir': str(metric_dir), 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
