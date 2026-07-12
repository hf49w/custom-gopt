import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path('/DATA_2/guest/custom-gopt')
sys.path.insert(0, str(REPO / 'src'))
from models import PCNStreamingScorer
from train_streaming_pcn import PCNUtteranceDataset, make_loader, move_batch, reset_state_where_needed, restore_invalid_state, slice_chunk

DIMS = ['accuracy', 'completeness', 'fluency', 'prosody', 'total']


def pcc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_exp(exp_dir):
    exp_dir = Path(exp_dir)
    config = json.load(open(exp_dir / 'config.json', encoding='utf-8'))
    args = config.get('args', {})
    data_dir = Path(config['data_dir'])
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    metadata = json.load(open(data_dir / 'metadata.json', encoding='utf-8'))
    train_raw = np.load(data_dir / 'train_chunks.npz')
    prosody_mean = train_raw['prosody'].mean(axis=0).astype(np.float32)
    prosody_std = train_raw['prosody'].std(axis=0).astype(np.float32)
    slot_prosody_mean = slot_prosody_std = None
    slot_prosody_dim = 0
    if 'slot_prosody' in train_raw.files:
        slot_prosody_dim = int(train_raw['slot_prosody'].shape[-1])
        slot_prosody_mean = train_raw['slot_prosody'].mean(axis=(0, 1)).astype(np.float32)
        slot_prosody_std = train_raw['slot_prosody'].std(axis=(0, 1)).astype(np.float32)
    test_set = PCNUtteranceDataset('test', data_dir, prosody_mean, prosody_std, slot_prosody_mean, slot_prosody_std)
    loader = make_loader(test_set, batch_size=int(args.get('batch_size', 16)), shuffle=False, num_workers=0)
    model = PCNStreamingScorer(
        phone_dim=int(metadata['phone_dim']),
        seq_len=int(metadata['seq_len']),
        prosody_dim=int(config.get('prosody_dim', len(metadata.get('prosody', [])) or 14)),
        embed_dim=int(args.get('embed_dim', 40)),
        num_heads=int(args.get('heads', 2)),
        depth=int(args.get('depth', 2)),
        gru_dim=int(args.get('gru_dim', 32)),
        main_context_tokens=int(args.get('main_context_tokens', 16)),
        utt_pooling_head=str(config.get('utt_pooling_head', args.get('utt_pooling_head', 'gru'))),
        fusion_mode=str(config.get('fusion_mode', args.get('fusion_mode', 'scalar_gate'))),
        slot_prosody_dim=int(config.get('slot_prosody_dim', slot_prosody_dim)),
    )
    weights = torch.load(exp_dir / 'models' / 'best_audio_model.pth', map_location='cpu')
    model.load_state_dict(weights)
    model.eval()
    preds = []
    targets = []
    confs = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, torch.device('cpu'))
            state = None
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
                    slot_prosody=chunk.get('slot_prosody'),
                    prev_state=state,
                    detach_next_state=True,
                )
                state = restore_invalid_state(out['next_state'], state, cur_valid)
                final_mask = (chunk['is_final'] > 0) & (cur_valid > 0)
                if final_mask.any():
                    preds.extend(out['utt_scores'][final_mask].cpu().numpy().tolist())
                    targets.extend(chunk['utt_target'][final_mask].cpu().numpy().tolist())
                    seq_len = chunk['cn_post'].shape[1]
                    idx = torch.arange(seq_len).unsqueeze(0)
                    valid_mask = (idx < chunk['visible_len'].unsqueeze(1)).float()
                    conf_mean = (out['confidence'].squeeze(-1) * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    confs.extend(conf_mean[final_mask].cpu().numpy().tolist())
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    confs = np.asarray(confs, dtype=np.float64)
    out = {'count': int(preds.shape[0])}
    for i, name in enumerate(DIMS):
        out[f'{name}_pcc'] = pcc(preds[:, i], targets[:, i])
        out[f'{name}_mae_norm'] = float(np.mean(np.abs(preds[:, i] - targets[:, i])))
    # coverage pcc per dimension, same confidence sorting as trainer
    for cov in [1.0, 0.9, 0.8, 0.7]:
        keep = max(1, int(math.ceil(preds.shape[0] * cov)))
        order = np.argsort(-confs)[:keep]
        for i, name in enumerate(DIMS):
            out[f'coverage_{int(cov*100)}_{name}_pcc'] = pcc(preds[order, i], targets[order, i])
    return out

root = REPO / 'exp' / 'pcn_extra_20260704_2130'
summary = {}
for exp in ['A_loss_dimmask', 'B_relaxed_softlabel']:
    summary[exp] = load_exp(root / exp)
print(json.dumps(summary, indent=2, ensure_ascii=False))
