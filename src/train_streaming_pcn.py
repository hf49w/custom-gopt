import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import PCNStreamingScorer


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--exp-dir', type=str, default='exp/streaming-pcn-gopt')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=40)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--gru-dim', type=int, default=32)
    parser.add_argument('--main-context-tokens', type=int, default=16)
    parser.add_argument('--loss-w-phone', type=float, default=1.0)
    parser.add_argument('--loss-w-word', type=float, default=1.0)
    parser.add_argument('--loss-w-utt', type=float, default=1.0)
    parser.add_argument('--loss-w-asr', type=float, default=0.5)
    parser.add_argument('--loss-w-uncertainty', type=float, default=0.2)
    parser.add_argument('--loss-w-teacher-score', type=float, default=0.5)
    parser.add_argument('--loss-w-prefix-kd', type=float, default=0.5)
    parser.add_argument('--loss-w-rank', type=float, default=0.1)
    parser.add_argument('--loss-w-commit-consistency', type=float, default=0.02)
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


class PCNChunkDataset(Dataset):
    def __init__(self, split, data_dir, prosody_mean=None, prosody_std=None, final_only=False):
        self.split = split
        self.data_dir = Path(data_dir)
        archive = np.load(self.data_dir / f'{split}_chunks.npz')
        is_final = archive['is_final'].astype(bool)
        keep = is_final if final_only else np.ones_like(is_final, dtype=bool)
        self.keep_indices = np.flatnonzero(keep)

        def take(name, default=None, dtype=torch.float32):
            if name in archive.files:
                arr = archive[name]
            elif default is not None:
                arr = default()
            else:
                raise KeyError(f'Missing array {name} in {self.data_dir / f"{split}_chunks.npz"}')
            return torch.tensor(arr[keep], dtype=dtype)

        n_all = archive['cn_post'].shape[0]
        seq_len = archive['cn_post'].shape[1]
        self.cn_post = take('cn_post')
        self.cn_stats = take('cn_stats')
        self.acoustic_post = take('acoustic_post')
        self.acoustic_stats = take('acoustic_stats')
        prosody = archive['prosody'][keep].astype(np.float32)
        if prosody_mean is not None and prosody_std is not None:
            prosody = (prosody - prosody_mean) / np.clip(prosody_std, 1e-6, None)
        self.prosody = torch.tensor(prosody, dtype=torch.float32)
        self.pcn_word_id = take(
            'pcn_word_id',
            default=lambda: np.zeros((n_all, seq_len), dtype=np.int32) - 1,
            dtype=torch.long,
        )
        self.phone_target = take('phone_target')
        self.word_target = take('word_target')
        self.utt_target = take('utt_target') / 5.0
        self.phone_score_target = self.phone_target[:, :, 1] / 5.0
        self.word_score_target = self.word_target[:, :, 0:3] / 5.0
        self.asr_correct_target = take('asr_correct_target')
        self.uncertainty_target = take('uncertainty_target')
        self.soft_label_weight = take('soft_label_weight')
        self.commit_mask = take('commit_mask')
        self.visible_len = take('visible_len', dtype=torch.long)
        self.is_final = take('is_final', dtype=torch.float32)
        self.coverage_ratio = take(
            'coverage_ratio',
            default=lambda: np.ones((n_all,), dtype=np.float32),
        )
        self.teacher_prefix_utt_score = take(
            'teacher_prefix_utt_score',
            default=lambda: np.zeros((n_all, 5), dtype=np.float32),
        ) / 5.0
        self.teacher_final_utt_score = take(
            'teacher_final_utt_score',
            default=lambda: np.zeros((n_all, 5), dtype=np.float32),
        ) / 5.0
        self.teacher_utt_mask = take(
            'teacher_utt_mask',
            default=lambda: np.zeros((n_all,), dtype=np.float32),
        )
        self.teacher_utt_dim_mask = take(
            'teacher_utt_dim_mask',
            default=lambda: np.repeat(archive['teacher_utt_mask'][:, None], 5, axis=1).astype(np.float32)
            if 'teacher_utt_mask' in archive.files
            else np.zeros((n_all, 5), dtype=np.float32),
        )
        self.teacher_word_score = take(
            'teacher_word_score',
            default=lambda: np.zeros((n_all, seq_len, 3), dtype=np.float32),
        ) / 5.0
        self.teacher_word_mask = take(
            'teacher_word_mask',
            default=lambda: np.zeros((n_all, seq_len), dtype=np.float32),
        )

        manifest = read_manifest(self.data_dir, split)
        if len(manifest) == n_all:
            utt_ids = [manifest[idx].get('utt_id', '') for idx in self.keep_indices]
            utt_map = {utt_id: index for index, utt_id in enumerate(sorted(set(utt_ids)))}
            self.utt_index = torch.tensor([utt_map[utt_id] for utt_id in utt_ids], dtype=torch.long)
            self.chunk_id = torch.tensor([int(manifest[idx].get('chunk_id', 0)) for idx in self.keep_indices], dtype=torch.long)
        else:
            self.utt_index = torch.arange(len(self.keep_indices), dtype=torch.long)
            self.chunk_id = torch.zeros(len(self.keep_indices), dtype=torch.long)

    def __len__(self):
        return self.cn_post.shape[0]

    def __getitem__(self, idx):
        return {
            'cn_post': self.cn_post[idx],
            'cn_stats': self.cn_stats[idx],
            'acoustic_post': self.acoustic_post[idx],
            'acoustic_stats': self.acoustic_stats[idx],
            'prosody': self.prosody[idx],
            'pcn_word_id': self.pcn_word_id[idx],
            'phone_score_target': self.phone_score_target[idx],
            'word_score_target': self.word_score_target[idx],
            'utt_target': self.utt_target[idx],
            'asr_correct_target': self.asr_correct_target[idx],
            'uncertainty_target': self.uncertainty_target[idx],
            'soft_label_weight': self.soft_label_weight[idx],
            'commit_mask': self.commit_mask[idx],
            'visible_len': self.visible_len[idx],
            'is_final': self.is_final[idx],
            'coverage_ratio': self.coverage_ratio[idx],
            'teacher_prefix_utt_score': self.teacher_prefix_utt_score[idx],
            'teacher_final_utt_score': self.teacher_final_utt_score[idx],
            'teacher_utt_mask': self.teacher_utt_mask[idx],
            'teacher_utt_dim_mask': self.teacher_utt_dim_mask[idx],
            'teacher_word_score': self.teacher_word_score[idx],
            'teacher_word_mask': self.teacher_word_mask[idx],
            'utt_index': self.utt_index[idx],
            'chunk_id': self.chunk_id[idx],
        }


def masked_mse(pred, target, mask, weight=None):
    effective = mask if weight is None else mask * weight
    while effective.dim() < pred.dim():
        effective = effective.unsqueeze(-1)
    denom = effective.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * effective).sum() / denom


def masked_bce_with_logits(logits, target, mask):
    effective = mask
    while effective.dim() < logits.dim():
        effective = effective.unsqueeze(-1)
    target = target.unsqueeze(-1) if target.dim() + 1 == logits.dim() else target
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none') * effective
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


def commit_consistency_loss(utt_scores, utt_index, chunk_id):
    losses = []
    for utt in torch.unique(utt_index):
        indices = torch.nonzero(utt_index == utt, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            continue
        order = torch.argsort(chunk_id[indices])
        ordered = utt_scores[indices[order]]
        losses.append(((ordered[1:] - ordered[:-1]) ** 2).mean())
    if not losses:
        return utt_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def valid_slot_mask(batch):
    seq_len = batch['cn_post'].shape[1]
    idx = torch.arange(seq_len, device=batch['cn_post'].device).unsqueeze(0)
    return (idx < batch['visible_len'].unsqueeze(1)).float()


def compute_losses(model, batch, args):
    out = model(
        cn_post=batch['cn_post'],
        cn_stats=batch['cn_stats'],
        acoustic_post=batch['acoustic_post'],
        acoustic_stats=batch['acoustic_stats'],
        prosody=batch['prosody'],
        visible_len=batch['visible_len'],
        commit_mask=batch['commit_mask'],
        word_ids=batch['pcn_word_id'],
    )
    valid_mask = valid_slot_mask(batch)
    supervise_weight = batch['soft_label_weight'] * batch['commit_mask'] * valid_mask
    word_teacher_mask = batch['teacher_word_mask'] * valid_mask
    teacher_mask = batch['teacher_utt_mask']
    teacher_dim_mask = batch['teacher_utt_dim_mask']
    beta_prefix = (batch['coverage_ratio'].clamp(0.0, 1.0) ** 2) * teacher_mask
    beta_prefix_dim = beta_prefix.unsqueeze(-1) * teacher_dim_mask

    loss_phone = masked_mse(
        out['phone_score'].squeeze(-1),
        batch['phone_score_target'],
        supervise_weight,
    )
    loss_word = masked_mse(
        out['word_scores'],
        batch['word_score_target'],
        supervise_weight,
    )
    loss_utt = masked_mse(
        out['utt_scores'],
        batch['utt_target'],
        batch['is_final'],
    )
    loss_asr = masked_bce_with_logits(
        out['asr_correct_logits'],
        batch['asr_correct_target'],
        valid_mask,
    )
    loss_uncertainty = masked_bce_with_logits(
        out['uncertainty_logits'],
        batch['uncertainty_target'],
        valid_mask,
    )
    loss_teacher_prefix = masked_mse(out['utt_scores'], batch['teacher_prefix_utt_score'], teacher_dim_mask)
    loss_teacher_final = masked_mse(out['utt_scores'], batch['teacher_final_utt_score'], beta_prefix_dim)
    loss_teacher_word = masked_mse(out['word_scores'], batch['teacher_word_score'], word_teacher_mask)
    loss_rank = pairwise_rank_loss(out['utt_scores'], batch['teacher_final_utt_score'], teacher_mask, args.rank_margin)
    loss_commit = commit_consistency_loss(out['utt_scores'], batch['utt_index'], batch['chunk_id'])

    loss_teacher = loss_teacher_prefix + loss_teacher_word
    loss_prefix_kd = loss_teacher_final
    total = (
        args.loss_w_phone * loss_phone
        + args.loss_w_word * loss_word
        + args.loss_w_utt * loss_utt
        + args.loss_w_asr * loss_asr
        + args.loss_w_uncertainty * loss_uncertainty
        + args.loss_w_teacher_score * loss_teacher
        + args.loss_w_prefix_kd * loss_prefix_kd
        + args.loss_w_rank * loss_rank
        + args.loss_w_commit_consistency * loss_commit
    )
    losses = {
        'loss': total,
        'phone': loss_phone,
        'word': loss_word,
        'utt': loss_utt,
        'asr': loss_asr,
        'uncertainty': loss_uncertainty,
        'teacher_score': loss_teacher,
        'prefix_kd': loss_prefix_kd,
        'rank': loss_rank,
        'commit': loss_commit,
    }
    return losses, out


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()
    totals = CounterFloat()
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        losses, out = compute_losses(model, batch, args)
        batch_size = batch['cn_post'].shape[0]
        for key, value in losses.items():
            totals.add(key, float(value.detach().cpu()) * batch_size)
        count += batch_size
    return totals.mean(max(count, 1))


class CounterFloat:
    def __init__(self):
        self.values = {}

    def add(self, key, value):
        self.values[key] = self.values.get(key, 0.0) + float(value)

    def mean(self, denom):
        return {key: value / float(denom) for key, value in self.values.items()}


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return DataLoader(**kwargs)


def state_dict(model):
    model = model.module if isinstance(model, nn.DataParallel) else model
    model = getattr(model, '_orig_mod', model)
    return model.state_dict()


def load_state(model, weights):
    model = model.module if isinstance(model, nn.DataParallel) else model
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
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
    model = model.to(device)
    if args.compile and hasattr(torch, 'compile') and not isinstance(model, nn.DataParallel):
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
        seen = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            losses, _ = compute_losses(model, batch, args)
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            optimizer.step()
            batch_size = batch['cn_post'].shape[0]
            for key, value in losses.items():
                totals.add(key, float(value.detach().cpu()) * batch_size)
            seen += batch_size
        train_metrics = totals.mean(max(seen, 1))
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

    best_path = Path(args.exp_dir) / 'models' / 'best_audio_model.pth'
    if best_path.exists():
        load_state(model, torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, args, device)
    (Path(args.exp_dir) / 'test_metrics.json').write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


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
    train_raw = np.load(data_dir / 'train_chunks.npz')
    prosody_mean = train_raw['prosody'].mean(axis=0).astype(np.float32)
    prosody_std = train_raw['prosody'].std(axis=0).astype(np.float32)

    train_set = PCNChunkDataset('train', data_dir, prosody_mean, prosody_std, final_only=False)
    val_set = PCNChunkDataset('val', data_dir, prosody_mean, prosody_std, final_only=False)
    test_set = PCNChunkDataset('test', data_dir, prosody_mean, prosody_std, final_only=False)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)

    phone_dim = int(metadata['phone_dim'])
    seq_len = int(metadata['seq_len'])
    prosody_dim = len(metadata.get('prosody', [])) or int(train_raw['prosody'].shape[-1])
    model = PCNStreamingScorer(
        phone_dim=phone_dim,
        seq_len=seq_len,
        prosody_dim=prosody_dim,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        depth=args.depth,
        gru_dim=args.gru_dim,
        main_context_tokens=args.main_context_tokens,
    )
    config = {
        'data_dir': str(data_dir),
        'metadata_schema': metadata.get('schema'),
        'phone_dim': phone_dim,
        'seq_len': seq_len,
        'prosody_dim': prosody_dim,
        'prosody_norm_mean': prosody_mean.tolist(),
        'prosody_norm_std': prosody_std.tolist(),
        'args': vars(args),
    }
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    train(model, train_loader, val_loader, test_loader, args, device)


if __name__ == '__main__':
    main()
