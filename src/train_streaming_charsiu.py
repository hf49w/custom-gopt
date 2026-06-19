import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import StreamingGOPT, StreamingGOPTNoPhn


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def parse_int_choices(raw_value):
    return [int(item.strip()) for item in raw_value.split(',') if item.strip()]


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with train_chunks.npz / val_chunks.npz / test_chunks.npz / metadata.json.')
    parser.add_argument('--exp-dir', type=str, default='./exp_streaming_charsiu')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--embed-dim', type=int, default=24)
    parser.add_argument('--loss-w-phn', type=float, default=1.0)
    parser.add_argument('--loss-w-word', type=float, default=1.0)
    parser.add_argument('--loss-w-word-asr', type=float, default=1.0)
    parser.add_argument('--loss-w-utt', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='streaming_gopt', choices=['streaming_gopt', 'streaming_gopt_nophn'])
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--main-context-tokens', type=str, default='8', help='Comma-separated choices, e.g. 4,8,12')
    parser.add_argument('--right-context-tokens', type=str, default='2', help='Comma-separated choices, e.g. 0,1,2,4')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--resume', action='store_true', help='Resume from exp-dir/last_checkpoint.pt if it exists.')
    return parser.parse_args()


def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_val_mse', 'phone_val_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_val_mse', 'utt_val_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_val_pcc']
    word_header_score = ['accuracy', 'stress', 'total', 'asr_accuracy']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset + '_' + x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset + '_' + x for x in word_header_score]
    return phn_header + utt_header + word_header


class StreamingChunkDataset(Dataset):
    def __init__(self, split, data_dir, metadata, final_only=False):
        archive = np.load(Path(data_dir) / f'{split}_chunks.npz')
        feat = archive['feat']
        phn_id = archive['phn_id']
        phn_score = archive['phn_score']
        word_label = archive['word_label']
        utt_label = archive['utt_label']
        phone_loss_mask = archive['phone_loss_mask']
        word_loss_mask = archive['word_loss_mask']
        word_asr_loss_mask = archive['word_asr_loss_mask']
        word_weight = archive['word_weight'] if 'word_weight' in archive.files else np.ones_like(word_loss_mask, dtype=np.float32)
        word_asr_weight = archive['word_asr_weight'] if 'word_asr_weight' in archive.files else np.ones_like(word_asr_loss_mask, dtype=np.float32)
        utt_loss_mask = archive['utt_loss_mask']
        is_final = archive['is_final']

        if final_only:
            keep = is_final.astype(bool)
            feat = feat[keep]
            phn_id = phn_id[keep]
            phn_score = phn_score[keep]
            word_label = word_label[keep]
            utt_label = utt_label[keep]
            phone_loss_mask = phone_loss_mask[keep]
            word_loss_mask = word_loss_mask[keep]
            word_asr_loss_mask = word_asr_loss_mask[keep]
            word_weight = word_weight[keep]
            word_asr_weight = word_asr_weight[keep]
            utt_loss_mask = utt_loss_mask[keep]
            is_final = is_final[keep]

        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.phn_id = torch.tensor(phn_id, dtype=torch.float32)
        self.phn_score = torch.tensor(phn_score, dtype=torch.float32)
        self.word_label = torch.tensor(word_label, dtype=torch.float32)
        self.utt_label = torch.tensor(utt_label, dtype=torch.float32) / 5.0
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5.0
        self.phone_loss_mask = torch.tensor(phone_loss_mask, dtype=torch.float32)
        self.word_loss_mask = torch.tensor(word_loss_mask, dtype=torch.float32)
        self.word_asr_loss_mask = torch.tensor(word_asr_loss_mask, dtype=torch.float32)
        self.word_weight = torch.tensor(word_weight, dtype=torch.float32)
        self.word_asr_weight = torch.tensor(word_asr_weight, dtype=torch.float32)
        self.utt_loss_mask = torch.tensor(utt_loss_mask, dtype=torch.float32)
        self.is_final = torch.tensor(is_final, dtype=torch.float32)

        norm_mean = float(metadata['train_norm_mean'])
        norm_std = float(metadata['train_norm_std'])
        valid_mask = self.phn_id >= 0
        self.feat[valid_mask] = (self.feat[valid_mask] - norm_mean) / norm_std

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return (
            self.feat[idx],
            self.phn_score[idx],
            self.phn_id[idx],
            self.word_label[idx],
            self.utt_label[idx],
            self.phone_loss_mask[idx],
            self.word_loss_mask[idx],
            self.word_asr_loss_mask[idx],
            self.word_weight[idx],
            self.word_asr_weight[idx],
            self.utt_loss_mask[idx],
        )


def masked_mse(pred, target, mask, weight=None):
    effective_mask = mask if weight is None else mask * weight
    denom = torch.sum(effective_mask)
    if denom.item() <= 0:
        return pred.new_tensor(0.0)
    loss = ((pred - target) ** 2) * effective_mask
    return loss.sum() / denom


def valid_phn(audio_output, target, mask):
    pred = audio_output.squeeze(2)
    valid = mask > 0
    valid_token_pred = pred[valid].cpu().numpy()
    valid_token_target = target[valid].cpu().numpy()
    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr


def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_pred = audio_output[:, i].cpu().numpy()
        cur_target = target[:, i].cpu().numpy()
        mse.append(np.mean((cur_pred - cur_target) ** 2))
        corr.append(np.corrcoef(cur_pred, cur_target)[0, 1])
    return mse, corr


def valid_word(audio_output, target, word_mask, score_slice):
    word_id = target[:, :, -1]
    target_score = target[:, :, score_slice]
    valid_token_pred = []
    valid_token_target = []

    for batch_idx in range(target.shape[0]):
        active_positions = torch.nonzero(word_mask[batch_idx] > 0, as_tuple=False).squeeze(1).tolist()
        if not active_positions:
            continue

        start = active_positions[0]
        prev_word = int(word_id[batch_idx, start].item())
        current_positions = [start]
        for pos in active_positions[1:]:
            cur_word = int(word_id[batch_idx, pos].item())
            if cur_word != prev_word:
                valid_token_pred.append(audio_output[batch_idx, current_positions, :].mean(dim=0).cpu().numpy())
                valid_token_target.append(target_score[batch_idx, current_positions, :].mean(dim=0).cpu().numpy())
                current_positions = [pos]
                prev_word = cur_word
            else:
                current_positions.append(pos)
        valid_token_pred.append(audio_output[batch_idx, current_positions, :].mean(dim=0).cpu().numpy())
        valid_token_target.append(target_score[batch_idx, current_positions, :].mean(dim=0).cpu().numpy())

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    for i in range(target_score.shape[-1]):
        mse_list.append(np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2))
        corr_list.append(np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1])
    return mse_list, corr_list, valid_token_pred, valid_token_target


def validate(audio_model, val_loader, args, best_mse, device, main_context_tokens, right_context_tokens):
    audio_model.eval()
    A_phn, A_phn_target, A_phn_mask = [], [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_w4, A_word_target, A_word_mask, A_word_asr_mask = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            audio_input, phn_label, phns, word_label, utt_label, phone_loss_mask, word_loss_mask, word_asr_loss_mask, word_weight, word_asr_weight, utt_loss_mask = batch
            audio_input = audio_input.to(device)
            phns = phns.to(device)

            u1, u2, u3, u4, u5, p, w1, w2, w3, w4 = audio_model(
                audio_input, phns, main_context_tokens=main_context_tokens, right_context_tokens=right_context_tokens,
            )
            A_phn.append(p.cpu())
            A_phn_target.append(phn_label)
            A_phn_mask.append(phone_loss_mask)
            A_u1.append(u1.cpu())
            A_u2.append(u2.cpu())
            A_u3.append(u3.cpu())
            A_u4.append(u4.cpu())
            A_u5.append(u5.cpu())
            A_utt_target.append(utt_label)
            A_w1.append(w1.cpu())
            A_w2.append(w2.cpu())
            A_w3.append(w3.cpu())
            A_w4.append(w4.cpu())
            A_word_target.append(word_label)
            A_word_mask.append(word_loss_mask)
            A_word_asr_mask.append(word_asr_loss_mask)

    A_phn = torch.cat(A_phn)
    A_phn_target = torch.cat(A_phn_target)
    A_phn_mask = torch.cat(A_phn_mask)
    A_u1 = torch.cat(A_u1)
    A_u2 = torch.cat(A_u2)
    A_u3 = torch.cat(A_u3)
    A_u4 = torch.cat(A_u4)
    A_u5 = torch.cat(A_u5)
    A_utt_target = torch.cat(A_utt_target)
    A_w1 = torch.cat(A_w1)
    A_w2 = torch.cat(A_w2)
    A_w3 = torch.cat(A_w3)
    A_w4 = torch.cat(A_w4)
    A_word_target = torch.cat(A_word_target)
    A_word_mask = torch.cat(A_word_mask)
    A_word_asr_mask = torch.cat(A_word_asr_mask)

    phn_mse, phn_corr = valid_phn(A_phn, A_phn_target, A_phn_mask)
    A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
    utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)
    A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
    word_mse_main, word_corr_main, valid_word_pred, valid_word_target = valid_word(
        A_word, A_word_target, A_word_mask, slice(0, 3),
    )
    word_asr_mse, word_asr_corr, valid_word_asr_pred, valid_word_asr_target = valid_word(
        A_w4, A_word_target, A_word_asr_mask, slice(3, 4),
    )
    word_mse = word_mse_main + word_asr_mse
    word_corr = word_corr_main + word_asr_corr

    if phn_mse < best_mse:
        preds_dir = Path(args.exp_dir) / 'preds'
        preds_dir.mkdir(parents=True, exist_ok=True)
        if not (preds_dir / 'phn_target.npy').exists():
            np.save(preds_dir / 'phn_target.npy', A_phn_target)
            np.save(preds_dir / 'word_target.npy', valid_word_target)
            np.save(preds_dir / 'word_asr_target.npy', valid_word_asr_target)
            np.save(preds_dir / 'utt_target.npy', A_utt_target)
        np.save(preds_dir / 'phn_pred.npy', A_phn)
        np.save(preds_dir / 'word_pred.npy', valid_word_pred)
        np.save(preds_dir / 'word_asr_pred.npy', valid_word_asr_pred)
        np.save(preds_dir / 'utt_pred.npy', A_utt)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr


def pick_streaming_context(args):
    main_context_tokens = random.choice(args.main_context_token_choices)
    right_context_tokens = random.choice(args.right_context_token_choices)
    return max(main_context_tokens, 1), max(right_context_tokens, 0)


def model_state_dict(audio_model):
    model = audio_model.module if isinstance(audio_model, nn.DataParallel) else audio_model
    model = getattr(model, '_orig_mod', model)
    return model.state_dict()


def load_model_state(audio_model, state_dict):
    model = audio_model.module if isinstance(audio_model, nn.DataParallel) else audio_model
    model = getattr(model, '_orig_mod', model)
    model.load_state_dict(state_dict)


def save_checkpoint(exp_dir, audio_model, optimizer, scheduler, result, epoch, global_step, best_epoch, best_mse):
    checkpoint = {
        'model_state': model_state_dict(audio_model),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'result': result,
        'epoch': int(epoch),
        'global_step': int(global_step),
        'best_epoch': int(best_epoch),
        'best_mse': float(best_mse),
    }
    torch.save(checkpoint, exp_dir / 'last_checkpoint.pt')


def load_checkpoint(exp_dir, audio_model, optimizer, scheduler, device, args):
    checkpoint_path = exp_dir / 'last_checkpoint.pt'
    empty_result = np.zeros([args.n_epochs, len(gen_result_header())])
    if not checkpoint_path.exists():
        return 0, 0, 0, 999.0, empty_result

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state(audio_model, checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    saved_result = checkpoint.get('result', empty_result)
    metric_dim = len(gen_result_header())
    result = np.zeros([args.n_epochs, metric_dim])
    usable_epochs = min(saved_result.shape[0], result.shape[0])
    result[:usable_epochs] = saved_result[:usable_epochs]
    return (
        int(checkpoint.get('epoch', 0)),
        int(checkpoint.get('global_step', 0)),
        int(checkpoint.get('best_epoch', 0)),
        float(checkpoint.get('best_mse', 999.0)),
        result,
    )


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


def train(audio_model, train_loader, train_eval_loader, val_loader, test_loader, args, device):
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    if args.compile and hasattr(torch, 'compile') and not isinstance(audio_model, nn.DataParallel):
        audio_model = torch.compile(audio_model)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)
    result = np.zeros([args.n_epochs, len(gen_result_header())])

    if args.resume:
        epoch, global_step, best_epoch, best_mse, result = load_checkpoint(
            exp_dir, audio_model, optimizer, scheduler, device, args,
        )

    eval_main_context_tokens = max(args.main_context_token_choices)
    eval_right_context_tokens = max(args.right_context_token_choices)

    while epoch < args.n_epochs:
        audio_model.train()
        for batch in train_loader:
            audio_input, phn_label, phns, word_label, utt_label, phone_loss_mask, word_loss_mask, word_asr_loss_mask, word_weight, word_asr_weight, utt_loss_mask = batch

            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            phns = phns.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            phone_loss_mask = phone_loss_mask.to(device, non_blocking=True)
            word_loss_mask = word_loss_mask.to(device, non_blocking=True)
            word_asr_loss_mask = word_asr_loss_mask.to(device, non_blocking=True)
            word_weight = word_weight.to(device, non_blocking=True)
            word_asr_weight = word_asr_weight.to(device, non_blocking=True)
            utt_loss_mask = utt_loss_mask.to(device, non_blocking=True)

            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr

            noise = (torch.rand_like(audio_input) - 1) * args.noise
            audio_input = audio_input + noise

            main_context_tokens, right_context_tokens = pick_streaming_context(args)
            u1, u2, u3, u4, u5, p, w1, w2, w3, w4 = audio_model(
                audio_input,
                phns,
                main_context_tokens=main_context_tokens,
                right_context_tokens=right_context_tokens,
            )

            p = p.squeeze(2)
            loss_phn = masked_mse(p, phn_label, phone_loss_mask)

            word_pred = torch.cat((w1, w2, w3), dim=2)
            word_target = word_label[:, :, 0:3]
            loss_word = masked_mse(
                word_pred,
                word_target,
                word_loss_mask.unsqueeze(-1),
                word_weight.unsqueeze(-1),
            )
            loss_word_asr = masked_mse(
                w4,
                word_label[:, :, 3:4],
                word_asr_loss_mask.unsqueeze(-1),
                word_asr_weight.unsqueeze(-1),
            )

            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = masked_mse(utt_preds, utt_label, utt_loss_mask.unsqueeze(-1))

            loss = (
                args.loss_w_phn * loss_phn
                + args.loss_w_word * loss_word
                + args.loss_w_word_asr * loss_word_asr
                + args.loss_w_utt * loss_utt
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr = validate(
            audio_model, train_eval_loader, args, -1, device, eval_main_context_tokens, eval_right_context_tokens,
        )
        val_mse, val_corr, val_utt_mse, val_utt_corr, val_word_mse, val_word_corr = validate(
            audio_model, val_loader, args, best_mse, device, eval_main_context_tokens, eval_right_context_tokens,
        )

        result[epoch, :6] = [epoch, tr_mse, tr_corr, val_mse, val_corr, optimizer.param_groups[0]['lr']]
        result[epoch, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, val_utt_mse, val_utt_corr])
        result[epoch, 26:34] = np.concatenate([tr_word_corr, val_word_corr])
        np.savetxt(exp_dir / 'result.csv', result, delimiter=',', header=','.join(gen_result_header()), comments='')

        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch

        if best_epoch == epoch:
            models_dir = exp_dir / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model_state_dict(audio_model), models_dir / 'best_audio_model.pth')

        if global_step > warm_up_step:
            scheduler.step()
        save_checkpoint(exp_dir, audio_model, optimizer, scheduler, result, epoch + 1, global_step, best_epoch, best_mse)
        epoch += 1

    models_dir = exp_dir / 'models'
    best_model_path = models_dir / 'best_audio_model.pth'
    if best_model_path.exists():
        load_model_state(audio_model, torch.load(best_model_path, map_location=device))
    test_mse, test_corr, test_utt_mse, test_utt_corr, test_word_mse, test_word_corr = validate(
        audio_model, test_loader, args, -1, device, eval_main_context_tokens, eval_right_context_tokens,
    )
    test_summary = {
        'phone_test_mse': float(test_mse),
        'phone_test_pcc': float(test_corr),
        'utt_test_mse': [float(x) for x in test_utt_mse],
        'utt_test_pcc': [float(x) for x in test_utt_corr],
        'word_test_mse': [float(x) for x in test_word_mse],
        'word_test_pcc': [float(x) for x in test_word_corr],
        'word_score_names': ['accuracy', 'stress', 'total', 'asr_accuracy'],
        'best_epoch': int(best_epoch),
    }
    (exp_dir / 'test_metrics.json').write_text(json.dumps(test_summary, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.main_context_token_choices = parse_int_choices(args.main_context_tokens)
    args.right_context_token_choices = parse_int_choices(args.right_context_tokens)

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.tf32 and device.type == 'cuda':
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_dataset = StreamingChunkDataset('train', data_dir, metadata, final_only=False)
    train_eval_dataset = StreamingChunkDataset('train', data_dir, metadata, final_only=True)
    val_dataset = StreamingChunkDataset('val', data_dir, metadata, final_only=True)
    test_dataset = StreamingChunkDataset('test', data_dir, metadata, final_only=True)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    train_eval_loader = make_loader(train_eval_dataset, len(train_eval_dataset), False, args.num_workers)
    val_loader = make_loader(val_dataset, len(val_dataset), False, args.num_workers)
    test_loader = make_loader(test_dataset, len(test_dataset), False, args.num_workers)

    input_dim = int(metadata['feat_dim'])
    phn_num = int(metadata['phn_num'])
    seq_len = int(metadata['seq_len'])

    if args.model == 'streaming_gopt':
        audio_model = StreamingGOPT(
            embed_dim=args.embed_dim,
            num_heads=args.heads,
            depth=args.depth,
            input_dim=input_dim,
            seq_len=seq_len,
            phn_num=phn_num,
        )
    else:
        audio_model = StreamingGOPTNoPhn(
            embed_dim=args.embed_dim,
            num_heads=args.heads,
            depth=args.depth,
            input_dim=input_dim,
            seq_len=seq_len,
            phn_num=phn_num,
        )

    config = {
        'data_dir': str(data_dir),
        'input_dim': input_dim,
        'phn_num': phn_num,
        'seq_len': seq_len,
        'args': vars(args),
    }
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.exp_dir) / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    train(audio_model, train_loader, train_eval_loader, val_loader, test_loader, args, device)


if __name__ == '__main__':
    main()
