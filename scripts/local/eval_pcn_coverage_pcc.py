import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from models import PCNStreamingScorer
from train_streaming_pcn import (
    PCNUtteranceDataset,
    make_loader,
    move_batch,
    reset_state_where_needed,
    restore_invalid_state,
    slice_chunk,
    valid_slot_mask,
)


SENTENCE_DIMS = ['accuracy', 'completeness', 'fluency', 'prosody', 'total']
WORD_DIMS = ['accuracy', 'stress', 'total']
COVERAGES = [1.0, 0.9, 0.8, 0.7]


def get_args():
    parser = argparse.ArgumentParser(description='Post-hoc PCN coverage PCC for phone, word, and sentence scores.')
    parser.add_argument('--exp-root', type=Path, required=True)
    parser.add_argument('--experiments', type=str, default='A_loss_dimmask,B_relaxed_softlabel')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-csv', type=Path, default=None)
    return parser.parse_args()


def pcc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_model(exp_dir, config, metadata, device):
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
    )
    weights = torch.load(exp_dir / 'models' / 'best_audio_model.pth', map_location=device)
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def build_dataset(config, split):
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


def collect_predictions(model, loader, device):
    sentence = {name: {'pred': [], 'target': [], 'conf': []} for name in SENTENCE_DIMS}
    word = {name: {'pred': [], 'target': [], 'conf': []} for name in WORD_DIMS}
    phone = {'phone': {'pred': [], 'target': [], 'conf': []}}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
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
                valid_mask = valid_slot_mask(chunk) * cur_valid.unsqueeze(-1)
                slot_conf = out['confidence'].squeeze(-1)

                final_mask = (chunk['is_final'] > 0) & (cur_valid > 0)
                if final_mask.any():
                    utt_conf = (slot_conf * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    for dim, name in enumerate(SENTENCE_DIMS):
                        sentence[name]['pred'].extend(out['utt_scores'][final_mask, dim].cpu().tolist())
                        sentence[name]['target'].extend(chunk['utt_target'][final_mask, dim].cpu().tolist())
                        sentence[name]['conf'].extend(utt_conf[final_mask].cpu().tolist())

                committed = (chunk['cumulative_commit_mask'] > 0) & (valid_mask > 0)
                phone_mask = committed & (chunk['phone_score_target'] >= 0)
                if phone_mask.any():
                    phone['phone']['pred'].extend(out['phone_score'].squeeze(-1)[phone_mask].cpu().tolist())
                    phone['phone']['target'].extend(chunk['phone_score_target'][phone_mask].cpu().tolist())
                    phone['phone']['conf'].extend(slot_conf[phone_mask].cpu().tolist())
                word_target = chunk['word_score_target']
                for dim, name in enumerate(WORD_DIMS):
                    word_mask = committed & (word_target[:, :, dim] >= 0)
                    if word_mask.any():
                        word[name]['pred'].extend(out['word_scores'][:, :, dim][word_mask].cpu().tolist())
                        word[name]['target'].extend(word_target[:, :, dim][word_mask].cpu().tolist())
                        word[name]['conf'].extend(slot_conf[word_mask].cpu().tolist())
    return {'sentence': sentence, 'word': word, 'phone': phone}


def coverage_rows(experiment, predictions):
    rows = []
    for level, metrics in predictions.items():
        for metric, values in metrics.items():
            pred = np.asarray(values['pred'], dtype=np.float64)
            target = np.asarray(values['target'], dtype=np.float64)
            conf = np.asarray(values['conf'], dtype=np.float64)
            for coverage in COVERAGES:
                row = {
                    'experiment': experiment,
                    'level': level,
                    'metric': metric,
                    'coverage': int(coverage * 100),
                    'count': int(pred.size),
                    'pcc': '',
                }
                if pred.size:
                    keep = max(1, int(math.ceil(pred.size * coverage)))
                    order = np.argsort(-conf)[:keep]
                    row['count'] = int(keep)
                    row['pcc'] = pcc(pred[order], target[order])
                rows.append(row)
    return rows


def main():
    args = get_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    rows = []
    for experiment in [item.strip() for item in args.experiments.split(',') if item.strip()]:
        exp_dir = args.exp_root / experiment
        config = json.loads((exp_dir / 'config.json').read_text(encoding='utf-8'))
        data_dir = Path(config['data_dir'])
        metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
        dataset = build_dataset(config, args.split)
        loader = make_loader(dataset, args.batch_size, False, args.num_workers)
        model = load_model(exp_dir, config, metadata, device)
        rows.extend(coverage_rows(experiment, collect_predictions(model, loader, device)))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['experiment', 'level', 'metric', 'coverage', 'count', 'pcc'])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
