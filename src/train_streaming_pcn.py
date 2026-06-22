import argparse
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import PCNStreamingScorer


PCN_SCHEMA = 'streaming_pcn_gopt_v2_stateful'

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, default='exp/streaming-pcn-gopt')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=40)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--gru-dim', type=int, default=32)
    parser.add_argument('--main-context-tokens', type=int, default=16)
    parser.add_argument('--tbptt-steps', type=int, default=0)
    parser.add_argument('--loss-w-phone', type=float, default=1.0)
    parser.add_argument('--loss-w-word', type=float, default=1.0)
    parser.add_argument('--loss-w-utt', type=float, default=1.0)
    parser.add_argument('--loss-w-asr', type=float, default=0.5)
    parser.add_argument('--loss-w-uncertainty', type=float, default=0.2)
    parser.add_argument('--loss-w-confidence', type=float, default=0.2)
    parser.add_argument('--loss-w-abstention', type=float, default=0.2)
    parser.add_argument('--loss-w-calibration', type=float, default=0.1)
    parser.add_argument('--loss-w-teacher-score', type=float, default=0.5)
    parser.add_argument('--loss-w-prefix-kd', type=float, default=0.5)
    parser.add_argument('--loss-w-rank', type=float, default=0.1)
    parser.add_argument('--loss-w-phone-stability', type=float, default=0.02)
    parser.add_argument('--loss-w-word-stability', type=float, default=0.02)
    parser.add_argument('--loss-w-utt-stability', type=float, default=0.02)
    parser.add_argument('--loss-w-commit-consistency', type=float, default=0.0, help='Deprecated compatibility wrapper; use stability losses.')
    parser.add_argument('--loss-w-state-projection', type=float, default=0.0)
    parser.add_argument('--rank-margin', type=float, default=0.02)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def read_manifest(data_dir, split):
    manifest_path = Path(data_dir) / f'{split}_manifest.jsonl'
    rows = []
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


class PCNUtteranceDataset(Dataset):
    def __init__(self, split, data_dir, prosody_mean=None, prosody_std=None):
        self.split = split
        self.data_dir = Path(data_dir)
        metadata = json.loads((self.data_dir / 'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('schema') != PCN_SCHEMA:
            raise ValueError(
                f'PCN stateful trainer requires metadata.schema={PCN_SCHEMA}, '
                f'got {metadata.get("schema")!r}. Rebuild data or migrate v1 explicitly.'
            )
        archive = np.load(self.data_dir / f'{split}_chunks.npz')
        self.arrays = {name: archive[name] for name in archive.files}
        required = [
            'cumulative_commit_mask',
            'new_commit_mask',
            'mapped_old_slot',
            'utterance_index',
            'state_reset',
            'new_committed_word_count',
            'cumulative_committed_word_count',
            'confidence_target',
            'confidence_loss_mask',
            'abstention_target',
            'abstention_loss_mask',
        ]
        missing = [name for name in required if name not in self.arrays]
        if missing:
            raise KeyError(f'{split}_chunks.npz is missing v2 stateful arrays: {missing}')

        prosody = self.arrays['prosody'].astype(np.float32)
        if prosody_mean is not None and prosody_std is not None:
            prosody = (prosody - prosody_mean) / np.clip(prosody_std, 1e-6, None)
        self.arrays['prosody'] = prosody.astype(np.float32)
        self.arrays['utt_target'] = (self.arrays['utt_target'].astype(np.float32) / 5.0)
        self.arrays['phone_score_target'] = (self.arrays['phone_target'][:, :, 1].astype(np.float32) / 5.0)
        self.arrays['word_score_target'] = (self.arrays['word_target'][:, :, 0:3].astype(np.float32) / 5.0)
        for name in ['teacher_prefix_utt_score', 'teacher_final_utt_score']:
            if name in self.arrays:
                self.arrays[name] = self.arrays[name].astype(np.float32) / 5.0
        if 'teacher_word_score' in self.arrays:
            self.arrays['teacher_word_score'] = self.arrays['teacher_word_score'].astype(np.float32) / 5.0

        if 'teacher_utt_dim_mask' not in self.arrays:
            mask = self.arrays.get('teacher_utt_mask', np.zeros((len(self.arrays['cn_post']),), dtype=np.float32))
            self.arrays['teacher_utt_dim_mask'] = np.repeat(mask[:, None], 5, axis=1).astype(np.float32)

        groups = defaultdict(list)
        for idx, utt_idx in enumerate(self.arrays['utterance_index'].astype(np.int64).tolist()):
            groups[int(utt_idx)].append(idx)
        self.groups = []
        for utt_idx in sorted(groups):
            indices = sorted(groups[utt_idx], key=lambda item: int(self.arrays['chunk_id'][item]) if 'chunk_id' in self.arrays else item)
            if 'chunk_id' not in self.arrays:
                manifest = read_manifest(self.data_dir, split)
                if len(manifest) == len(self.arrays['cn_post']):
                    indices = sorted(indices, key=lambda item: int(manifest[item].get('chunk_id', 0)))
            self.groups.append(indices)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = np.asarray(self.groups[idx], dtype=np.int64)
        sample = {}
        for name, arr in self.arrays.items():
            sample[name] = arr[indices]
        if 'chunk_id' not in sample:
            sample['chunk_id'] = np.arange(indices.shape[0], dtype=np.int32)
        return sample


def pcn_utterance_collate(samples):
    batch_size = len(samples)
    max_chunks = max(sample['cn_post'].shape[0] for sample in samples)
    out = {'chunk_valid_mask': torch.zeros((batch_size, max_chunks), dtype=torch.float32)}
    keys = sorted(set().union(*(sample.keys() for sample in samples)))
    for key in keys:
        if key in out:
            continue
        exemplar = samples[0][key]
        shape = (batch_size, max_chunks) + tuple(exemplar.shape[1:])
        dtype = torch.long if exemplar.dtype.kind in {'i', 'u'} else torch.float32
        fill_value = -1 if key in {'pcn_word_id', 'mapped_old_slot'} else 0
        tensor = torch.full(shape, fill_value=fill_value, dtype=dtype)
        for row, sample in enumerate(samples):
            value = torch.tensor(sample[key], dtype=dtype)
            tensor[row, : value.shape[0]] = value
            out['chunk_valid_mask'][row, : value.shape[0]] = 1.0
        out[key] = tensor
    return out


def masked_mse(pred, target, mask, weight=None):
    effective = mask if weight is None else mask * weight
    while effective.dim() < pred.dim():
        effective = effective.unsqueeze(-1)
    denom = effective.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * effective).sum() / denom


def masked_huber(pred, target, mask, delta=0.05):
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    loss = F.smooth_l1_loss(pred, target, reduction='none', beta=delta) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_bce_with_logits(logits, target, mask):
    effective = mask
    while effective.dim() < logits.dim():
        effective = effective.unsqueeze(-1)
    target = target.unsqueeze(-1) if target.dim() + 1 == logits.dim() else target
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none') * effective
    return loss.sum() / effective.sum().clamp_min(1.0)


def pairwise_rank_loss(student_score, teacher_score, mask, margin):
    valid = torch.nonzero(mask > 0, as_tuple=False).squeeze(1)
    if valid.numel() < 2:
        return student_score.new_tensor(0.0)
    s = student_score[valid, -1]
    t = teacher_score[valid, -1]
    diff_t = t.unsqueeze(0) - t.unsqueeze(1)
    diff_s = s.unsqueeze(0) - s.unsqueeze(1)
    sign = torch.sign(diff_t)
    pair_mask = sign.abs() > 0
    if pair_mask.sum().item() <= 0:
        return student_score.new_tensor(0.0)
    loss = torch.relu(margin - sign * diff_s)
    return loss[pair_mask].mean()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def valid_slot_mask(chunk):
    seq_len = chunk['cn_post'].shape[1]
    idx = torch.arange(seq_len, device=chunk['cn_post'].device).unsqueeze(0)
    return (idx < chunk['visible_len'].unsqueeze(1)).float()


def slice_chunk(batch, chunk_idx):
    skip = {'chunk_valid_mask'}
    return {key: value[:, chunk_idx] for key, value in batch.items() if key not in skip}


def reset_state_where_needed(state, state_reset):
    if state is None:
        return None
    reset = (state_reset > 0).view(1, -1, 1).to(dtype=state.dtype, device=state.device)
    return state * (1.0 - reset)


def restore_invalid_state(next_state, prev_state, valid):
    if prev_state is None:
        return next_state
    keep_prev = (valid <= 0).view(1, -1, 1).to(dtype=next_state.dtype, device=next_state.device)
    return next_state * (1.0 - keep_prev) + prev_state * keep_prev


def stability_losses(prev_out, out, chunk, prev_valid, cur_valid):
    device = chunk['cn_post'].device
    zero = chunk['cn_post'].new_tensor(0.0)
    phone_losses = []
    word_losses = []
    utt_losses = []
    revision_phone = 0
    revision_word = 0
    total_pairs = 0
    batch_size, seq_len = chunk['mapped_old_slot'].shape
    for row in range(batch_size):
        if prev_valid[row].item() <= 0 or cur_valid[row].item() <= 0:
            continue
        mapped = chunk['mapped_old_slot'][row]
        old_mask = (mapped >= 0) & (chunk['cumulative_commit_mask'][row] > 0)
        if old_mask.any():
            cur_pos = torch.nonzero(old_mask, as_tuple=False).squeeze(1)
            prev_pos = mapped[cur_pos].long().clamp_min(0)
            weight = chunk['prefix_stability'][row] * (
                1.0
                - chunk['new_committed_word_count'][row].float()
                / torch.clamp(chunk['cumulative_committed_word_count'][row].float(), min=1.0)
            )
            weight = torch.clamp(weight, min=0.0, max=1.0)
            phone_delta = out['phone_score'][row, cur_pos].squeeze(-1) - prev_out['phone_score'][row, prev_pos].squeeze(-1)
            word_delta = out['word_scores'][row, cur_pos] - prev_out['word_scores'][row, prev_pos]
            phone_losses.append(F.smooth_l1_loss(phone_delta, torch.zeros_like(phone_delta), reduction='mean', beta=0.05) * weight)
            word_losses.append(F.smooth_l1_loss(word_delta, torch.zeros_like(word_delta), reduction='mean', beta=0.05) * weight)
            revision_phone += int((phone_delta.detach().abs() > 0.05).sum().item())
            revision_word += int((word_delta.detach().abs().mean(dim=-1) > 0.05).sum().item())
            total_pairs += int(cur_pos.numel())
        weight = chunk['prefix_stability'][row] * (
            1.0
            - chunk['new_committed_word_count'][row].float()
            / torch.clamp(chunk['cumulative_committed_word_count'][row].float(), min=1.0)
        )
        weight = torch.clamp(weight, min=0.0, max=1.0)
        utt_delta = out['utt_scores'][row] - prev_out['utt_scores'][row]
        utt_losses.append(F.smooth_l1_loss(utt_delta, torch.zeros_like(utt_delta), reduction='mean', beta=0.05) * weight)
    return {
        'phone': torch.stack(phone_losses).mean() if phone_losses else zero,
        'word': torch.stack(word_losses).mean() if word_losses else zero,
        'utt': torch.stack(utt_losses).mean() if utt_losses else zero,
        'revision_phone': revision_phone,
        'revision_word': revision_word,
        'revision_pairs': total_pairs,
    }


def compute_sequential_losses(model, batch, args):
    max_chunks = batch['chunk_valid_mask'].shape[1]
    state = None
    prev_out = None
    prev_valid = None
    totals = defaultdict(lambda: batch['cn_post'].new_tensor(0.0))
    stats = defaultdict(float)
    valid_chunk_count = batch['chunk_valid_mask'].sum().clamp_min(1.0)

    for chunk_idx in range(max_chunks):
        cur_valid = batch['chunk_valid_mask'][:, chunk_idx]
        if cur_valid.sum().item() <= 0:
            continue
        chunk = slice_chunk(batch, chunk_idx)
        state = reset_state_where_needed(state, chunk['state_reset'])
        out = model(
            cn_post=chunk['cn_post'],
            cn_stats=chunk['cn_stats'],
            acoustic_post=chunk['acoustic_post'],
            acoustic_stats=chunk['acoustic_stats'],
            prosody=chunk['prosody'],
            visible_len=chunk['visible_len'],
            cumulative_commit_mask=chunk['cumulative_commit_mask'],
            new_commit_mask=chunk['new_commit_mask'],
            word_ids=chunk['pcn_word_id'],
            prev_state=state,
            detach_next_state=False,
        )
        state = restore_invalid_state(out['next_state'], state, cur_valid)
        valid_mask = valid_slot_mask(chunk) * cur_valid.unsqueeze(-1)
        supervise_weight = chunk['soft_label_weight'] * chunk['cumulative_commit_mask'] * valid_mask
        teacher_mask = chunk.get('teacher_utt_mask', torch.zeros_like(cur_valid)) * cur_valid
        teacher_dim_mask = chunk.get('teacher_utt_dim_mask', torch.zeros((cur_valid.shape[0], 5), device=cur_valid.device)) * cur_valid.unsqueeze(-1)
        teacher_word_mask = chunk.get('teacher_word_dim_mask')
        if teacher_word_mask is None:
            teacher_word_mask = chunk.get(
                'teacher_word_mask',
                torch.zeros_like(valid_mask),
            ).unsqueeze(-1)
        teacher_word_mask = teacher_word_mask * valid_mask.unsqueeze(-1)
        beta_prefix = (chunk['coverage_ratio'].clamp(0.0, 1.0) ** 2) * teacher_mask
        beta_prefix_dim = beta_prefix.unsqueeze(-1) * teacher_dim_mask

        losses = {
            'phone': masked_mse(out['phone_score'].squeeze(-1), chunk['phone_score_target'], supervise_weight),
            'word': masked_mse(out['word_scores'], chunk['word_score_target'], supervise_weight),
            'utt': masked_mse(out['utt_scores'], chunk['utt_target'], chunk['is_final'] * cur_valid),
            'asr': masked_bce_with_logits(out['asr_correct_logits'], chunk['asr_correct_target'], valid_mask),
            'uncertainty': masked_bce_with_logits(out['uncertainty_logits'], chunk['uncertainty_target'], valid_mask),
            'confidence': masked_bce_with_logits(out['confidence_logit'], chunk['confidence_target'], chunk['confidence_loss_mask'] * valid_mask),
            'abstention': masked_bce_with_logits(out['abstention_logit'], chunk['abstention_target'], chunk['abstention_loss_mask'] * valid_mask),
            'calibration': masked_mse(out['confidence'].squeeze(-1), chunk['confidence_target'], chunk['confidence_loss_mask'] * valid_mask),
            'teacher_score': masked_mse(out['utt_scores'], chunk.get('teacher_prefix_utt_score', torch.zeros_like(out['utt_scores'])), teacher_dim_mask)
            + masked_mse(
                out['word_scores'],
                chunk.get('teacher_word_score', torch.zeros_like(out['word_scores'])),
                teacher_word_mask,
            ),
            'prefix_kd': masked_mse(out['utt_scores'], chunk.get('teacher_final_utt_score', torch.zeros_like(out['utt_scores'])), beta_prefix_dim),
            'rank': pairwise_rank_loss(out['utt_scores'], chunk.get('teacher_final_utt_score', torch.zeros_like(out['utt_scores'])), teacher_mask, args.rank_margin),
        }
        if 'teacher_state_embedding' in chunk and 'state_projection' in out and args.loss_w_state_projection > 0:
            losses['state_projection'] = masked_mse(
                out['state_projection'],
                chunk['teacher_state_embedding'],
                chunk.get('teacher_state_mask', torch.zeros_like(cur_valid)) * cur_valid,
            )
        else:
            losses['state_projection'] = out['utt_scores'].new_tensor(0.0)

        if prev_out is not None:
            stability = stability_losses(prev_out, out, chunk, prev_valid, cur_valid)
            losses['phone_stability'] = stability['phone']
            losses['word_stability'] = stability['word']
            losses['utt_stability'] = stability['utt']
            stats['revision_phone'] += stability['revision_phone']
            stats['revision_word'] += stability['revision_word']
            stats['revision_pairs'] += stability['revision_pairs']
        else:
            losses['phone_stability'] = out['utt_scores'].new_tensor(0.0)
            losses['word_stability'] = out['utt_scores'].new_tensor(0.0)
            losses['utt_stability'] = out['utt_scores'].new_tensor(0.0)

        for key, value in losses.items():
            totals[key] = totals[key] + value * cur_valid.sum()
        stats['state_updates'] += int(((chunk['new_committed_word_count'] > 0) & (cur_valid > 0)).sum().item())
        stats['new_committed_words'] += float((chunk['new_committed_word_count'].float() * cur_valid).sum().item())
        stats['valid_chunks'] += float(cur_valid.sum().item())
        prev_out = out
        prev_valid = cur_valid
        if args.tbptt_steps > 0 and (chunk_idx + 1) % args.tbptt_steps == 0 and state is not None:
            state = state.detach()

    mean_losses = {key: value / valid_chunk_count for key, value in totals.items()}
    total = (
        args.loss_w_phone * mean_losses['phone']
        + args.loss_w_word * mean_losses['word']
        + args.loss_w_utt * mean_losses['utt']
        + args.loss_w_asr * mean_losses['asr']
        + args.loss_w_uncertainty * mean_losses['uncertainty']
        + args.loss_w_confidence * mean_losses['confidence']
        + args.loss_w_abstention * mean_losses['abstention']
        + args.loss_w_calibration * mean_losses['calibration']
        + args.loss_w_teacher_score * mean_losses['teacher_score']
        + args.loss_w_prefix_kd * mean_losses['prefix_kd']
        + args.loss_w_rank * mean_losses['rank']
        + args.loss_w_phone_stability * mean_losses['phone_stability']
        + args.loss_w_word_stability * mean_losses['word_stability']
        + args.loss_w_utt_stability * mean_losses['utt_stability']
        + args.loss_w_state_projection * mean_losses['state_projection']
    )
    mean_losses['loss'] = total
    return mean_losses, stats


class CounterFloat:
    def __init__(self):
        self.values = defaultdict(float)

    def add(self, key, value):
        self.values[key] += float(value)

    def mean(self, denom):
        return {key: value / float(max(denom, 1.0)) for key, value in self.values.items()}


def pcc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auprc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / max(labels.sum(), 1)
    return float(np.trapz(precision, recall))


def ece(conf, target, bins=10):
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    out = 0.0
    for lo in np.linspace(0.0, 1.0, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            out += float(mask.mean()) * abs(float(conf[mask].mean()) - float(target[mask].mean()))
    return out


def risk_coverage(conf, error):
    conf = np.asarray(conf, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if conf.size == 0:
        return {'aurc': 0.0}
    order = np.argsort(-conf)
    sorted_error = error[order]
    risk = np.cumsum(sorted_error) / np.arange(1, sorted_error.size + 1)
    coverage = np.arange(1, sorted_error.size + 1) / sorted_error.size
    points = []
    for target_coverage in np.linspace(0.1, 1.0, 10):
        idx = min(max(int(math.ceil(target_coverage * sorted_error.size)) - 1, 0), sorted_error.size - 1)
        points.append({'coverage': float(coverage[idx]), 'risk': float(risk[idx])})
    return {'aurc': float(np.trapz(risk, coverage)), 'risk_coverage_curve': points}


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()
    totals = CounterFloat()
    stats = defaultdict(float)
    seen_chunks = 0.0
    conf_values, conf_targets = [], []
    abst_values, abst_targets = [], []
    slot_conf_for_risk, slot_errors = [], []
    utt_pred, utt_target, utt_conf = [], [], []
    adjacent_deltas = []
    convergence_by_utt = defaultdict(list)

    for batch in loader:
        batch = move_batch(batch, device)
        losses, cur_stats = compute_sequential_losses(model, batch, args)
        valid_chunks = float(batch['chunk_valid_mask'].sum().item())
        for key, value in losses.items():
            totals.add(key, float(value.detach().cpu()) * valid_chunks)
        for key, value in cur_stats.items():
            stats[key] += float(value)
        seen_chunks += valid_chunks

        state = None
        prev_utt_scores = None
        max_chunks = batch['chunk_valid_mask'].shape[1]
        for chunk_idx in range(max_chunks):
            cur_valid = batch['chunk_valid_mask'][:, chunk_idx]
            if cur_valid.sum().item() <= 0:
                continue
            chunk = slice_chunk(batch, chunk_idx)
            state = reset_state_where_needed(state, chunk['state_reset'])
            out = model(
                cn_post=chunk['cn_post'],
                cn_stats=chunk['cn_stats'],
                acoustic_post=chunk['acoustic_post'],
                acoustic_stats=chunk['acoustic_stats'],
                prosody=chunk['prosody'],
                visible_len=chunk['visible_len'],
                cumulative_commit_mask=chunk['cumulative_commit_mask'],
                new_commit_mask=chunk['new_commit_mask'],
                word_ids=chunk['pcn_word_id'],
                prev_state=state,
                detach_next_state=True,
            )
            state = restore_invalid_state(out['next_state'], state, cur_valid)
            valid_mask = valid_slot_mask(chunk) * cur_valid.unsqueeze(-1)
            conf_mask = chunk['confidence_loss_mask'] * valid_mask
            abst_mask = chunk['abstention_loss_mask'] * valid_mask
            if conf_mask.sum().item() > 0:
                conf_values.extend(out['confidence'].squeeze(-1)[conf_mask > 0].detach().cpu().tolist())
                conf_targets.extend(chunk['confidence_target'][conf_mask > 0].detach().cpu().tolist())
            if abst_mask.sum().item() > 0:
                abst_values.extend(out['abstention_probability'].squeeze(-1)[abst_mask > 0].detach().cpu().tolist())
                abst_targets.extend(chunk['abstention_target'][abst_mask > 0].detach().cpu().tolist())
            supervise = chunk['soft_label_weight'] * chunk['cumulative_commit_mask'] * valid_mask
            if supervise.sum().item() > 0:
                slot_conf_for_risk.extend(out['confidence'].squeeze(-1)[supervise > 0].detach().cpu().tolist())
                slot_errors.extend(
                    (out['phone_score'].squeeze(-1) - chunk['phone_score_target']).abs()[supervise > 0].detach().cpu().tolist()
                )
            final_mask = (chunk['is_final'] > 0) & (cur_valid > 0)
            if final_mask.any():
                utt_pred.extend(out['utt_scores'][final_mask, -1].detach().cpu().tolist())
                utt_target.extend(chunk['utt_target'][final_mask, -1].detach().cpu().tolist())
                conf_mean = (out['confidence'].squeeze(-1) * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                utt_conf.extend(conf_mean[final_mask].detach().cpu().tolist())
            if prev_utt_scores is not None:
                valid_pair = (cur_valid > 0)
                deltas = (out['utt_scores'][:, -1] - prev_utt_scores[:, -1]).abs()[valid_pair]
                adjacent_deltas.extend(deltas.detach().cpu().tolist())
            for row in range(chunk['cn_post'].shape[0]):
                if cur_valid[row].item() > 0:
                    utt_key = int(chunk['utterance_index'][row].item())
                    convergence_by_utt[utt_key].append(float(out['utt_scores'][row, -1].detach().cpu()))
            prev_utt_scores = out['utt_scores'].detach()

    metrics = totals.mean(seen_chunks)
    conf_values = np.asarray(conf_values, dtype=np.float64)
    conf_targets = np.asarray(conf_targets, dtype=np.float64)
    abst_values = np.asarray(abst_values, dtype=np.float64)
    abst_targets = np.asarray(abst_targets, dtype=np.float64)
    if conf_values.size:
        metrics['confidence_brier'] = float(np.mean((conf_values - conf_targets) ** 2))
        metrics['confidence_ece'] = ece(conf_values, conf_targets)
    if abst_values.size:
        metrics['abstention_auroc'] = auroc(abst_values, (abst_targets >= 0.5).astype(np.int32))
        metrics['abstention_auprc'] = auprc(abst_values, (abst_targets >= 0.5).astype(np.int32))
    metrics.update(risk_coverage(slot_conf_for_risk, slot_errors))
    utt_pred = np.asarray(utt_pred, dtype=np.float64)
    utt_target = np.asarray(utt_target, dtype=np.float64)
    utt_conf = np.asarray(utt_conf, dtype=np.float64)
    for coverage in [1.0, 0.9, 0.8, 0.7]:
        if utt_pred.size:
            keep = max(1, int(math.ceil(utt_pred.size * coverage)))
            order = np.argsort(-utt_conf)[:keep]
            metrics[f'coverage_{int(coverage * 100)}_mae'] = float(np.mean(np.abs(utt_pred[order] - utt_target[order])))
            metrics[f'coverage_{int(coverage * 100)}_pcc'] = pcc(utt_pred[order], utt_target[order])
    if adjacent_deltas:
        metrics['mean_adjacent_utt_delta'] = float(np.mean(adjacent_deltas))
        metrics['p90_adjacent_utt_delta'] = float(np.percentile(adjacent_deltas, 90))
    metrics['phone_revision_rate'] = float(stats['revision_phone'] / max(stats['revision_pairs'], 1.0))
    metrics['word_revision_rate'] = float(stats['revision_word'] / max(stats['revision_pairs'], 1.0))
    metrics['state_update_count'] = float(stats['state_updates'])
    metrics['mean_new_committed_word_count'] = float(stats['new_committed_words'] / max(stats['valid_chunks'], 1.0))
    stable_times = []
    convergence = []
    for values in convergence_by_utt.values():
        if len(values) < 2:
            continue
        final = values[-1]
        convergence.extend([abs(value - final) for value in values])
        for idx, value in enumerate(values):
            if abs(value - final) <= 0.05:
                stable_times.append(idx)
                break
    if convergence:
        metrics['convergence_progress'] = float(np.mean(convergence))
    if stable_times:
        metrics['first_stable_chunk'] = float(np.mean(stable_times))
    return metrics


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': pcn_utterance_collate,
    }
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return DataLoader(**kwargs)


def state_dict(model):
    model = getattr(model, '_orig_mod', model)
    return model.state_dict()


def load_state(model, weights):
    model = getattr(model, '_orig_mod', model)
    model.load_state_dict(weights)


def save_checkpoint(exp_dir, model, optimizer, epoch, best_val, history):
    torch.save(
        {
            'model_state': state_dict(model),
            'optimizer_state': optimizer.state_dict(),
            'epoch': int(epoch),
            'best_val': float(best_val),
            'history': history,
        },
        exp_dir / 'last_checkpoint.pt',
    )


def train(model, train_loader, val_loader, test_loader, args, device):
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0
    best_val = math.inf
    history = []
    ckpt = exp_dir / 'last_checkpoint.pt'
    if args.resume and ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        load_state(model, payload['model_state'])
        optimizer.load_state_dict(payload['optimizer_state'])
        start_epoch = int(payload.get('epoch', 0))
        best_val = float(payload.get('best_val', math.inf))
        history = payload.get('history', [])

    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        totals = CounterFloat()
        seen_chunks = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            losses, _ = compute_sequential_losses(model, batch, args)
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            optimizer.step()
            valid_chunks = float(batch['chunk_valid_mask'].sum().item())
            for key, value in losses.items():
                totals.add(key, float(value.detach().cpu()) * valid_chunks)
            seen_chunks += valid_chunks
        train_metrics = totals.mean(seen_chunks)
        val_metrics = evaluate(model, val_loader, args, device)
        row = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
        history.append(row)
        (exp_dir / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        val_loss = float(val_metrics.get('loss', math.inf))
        if val_loss < best_val:
            best_val = val_loss
            models_dir = exp_dir / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict(model), models_dir / 'best_audio_model.pth')
        save_checkpoint(exp_dir, model, optimizer, epoch + 1, best_val, history)

    best_path = exp_dir / 'models' / 'best_audio_model.pth'
    if best_path.exists():
        load_state(model, torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, args, device)
    (exp_dir / 'test_metrics.json').write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.tf32 and device.type == 'cuda':
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('schema') != PCN_SCHEMA:
        raise ValueError(
            f'train_streaming_pcn.py now requires {PCN_SCHEMA}. '
            f'Found {metadata.get("schema")!r}; rebuild with build_streaming_pcn_gopt_data.py.'
        )
    train_raw = np.load(data_dir / 'train_chunks.npz')
    prosody_mean = train_raw['prosody'].mean(axis=0).astype(np.float32)
    prosody_std = train_raw['prosody'].std(axis=0).astype(np.float32)

    train_set = PCNUtteranceDataset('train', data_dir, prosody_mean, prosody_std)
    val_set = PCNUtteranceDataset('val', data_dir, prosody_mean, prosody_std)
    test_set = PCNUtteranceDataset('test', data_dir, prosody_mean, prosody_std)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    phone_dim = int(metadata['phone_dim'])
    seq_len = int(metadata['seq_len'])
    prosody_dim = len(metadata.get('prosody', [])) or int(train_raw['prosody'].shape[-1])
    has_teacher_state = 'teacher_state_embedding' in train_raw.files
    model = PCNStreamingScorer(
        phone_dim=phone_dim,
        seq_len=seq_len,
        prosody_dim=prosody_dim,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        depth=args.depth,
        gru_dim=args.gru_dim,
        main_context_tokens=args.main_context_tokens,
        use_state_projection=bool(has_teacher_state and args.loss_w_state_projection > 0),
    )
    config = {
        'data_dir': str(data_dir),
        'metadata_schema': metadata.get('schema'),
        'pcn_type': metadata.get('pcn_type'),
        'phone_dim': phone_dim,
        'seq_len': seq_len,
        'prosody_dim': prosody_dim,
        'prosody_norm_mean': prosody_mean.tolist(),
        'prosody_norm_std': prosody_std.tolist(),
        'uses_state_projection': bool(has_teacher_state and args.loss_w_state_projection > 0),
        'args': vars(args),
    }
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    train(model, train_loader, val_loader, test_loader, args, device)


if __name__ == '__main__':
    main()
