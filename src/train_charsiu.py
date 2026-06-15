import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import BaselineLSTM, GOPT, GOPTNoPhn


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), platform.node(), time.asctime()))


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with tr_feat.npy / te_feat.npy / metadata.json.')
    parser.add_argument('--exp-dir', type=str, default='./exp_charsiu', help='Directory to dump experiments')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--goptdepth', type=int, default=3)
    parser.add_argument('--goptheads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--embed-dim', type=int, default=24)
    parser.add_argument('--loss-w-phn', type=float, default=1.0)
    parser.add_argument('--loss-w-word', type=float, default=1.0)
    parser.add_argument('--loss-w-utt', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='gopt', choices=['gopt', 'gopt_nophn', 'lstm'])
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header


class SeqDataset(Dataset):
    def __init__(self, split, data_dir, metadata):
        self.data_dir = Path(data_dir)
        prefix = 'tr' if split == 'train' else 'te'
        self.feat = torch.tensor(np.load(self.data_dir / f'{prefix}_feat.npy'), dtype=torch.float32)
        self.phn_label = torch.tensor(np.load(self.data_dir / f'{prefix}_label_phn.npy'), dtype=torch.float32)
        self.utt_label = torch.tensor(np.load(self.data_dir / f'{prefix}_label_utt.npy'), dtype=torch.float32)
        self.word_label = torch.tensor(np.load(self.data_dir / f'{prefix}_label_word.npy'), dtype=torch.float32)

        norm_mean = float(metadata['train_norm_mean'])
        norm_std = float(metadata['train_norm_std'])
        valid_mask = self.phn_label[:, :, 0] >= 0
        self.feat[valid_mask] = (self.feat[valid_mask] - norm_mean) / norm_std

        self.utt_label = self.utt_label / 5.0
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5.0

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return self.feat[idx], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx], self.word_label[idx]


def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)
    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr


def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr


def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]
    valid_token_pred = []
    valid_token_target = []
    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            if cur_w_id != prev_w_id:
                valid_token_pred.append(np.mean(audio_output[i, start_id: j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id: j, :].numpy(), axis=0))
                if cur_w_id == -1:
                    break
                prev_w_id = cur_w_id
                start_id = j

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)
    mse_list, corr_list = [], []
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target


def validate(audio_model, val_loader, args, best_mse, device):
    audio_model.eval()
    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    with torch.no_grad():
        for audio_input, phn_label, phns, utt_label, word_label in val_loader:
            audio_input = audio_input.to(device)
            phns = phns.to(device)
            u1, u2, u3, u4, u5, p, w1, w2, w3 = audio_model(audio_input, phns)
            p = p.detach().cpu()
            u1, u2, u3, u4, u5 = u1.detach().cpu(), u2.detach().cpu(), u3.detach().cpu(), u4.detach().cpu(), u5.detach().cpu()
            w1, w2, w3 = w1.detach().cpu(), w2.detach().cpu(), w3.detach().cpu()

            A_phn.append(p)
            A_phn_target.append(phn_label)
            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)
            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

    A_phn, A_phn_target = torch.cat(A_phn), torch.cat(A_phn_target)
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)
    A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

    phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)
    A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
    utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)
    A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
    word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

    if phn_mse < best_mse:
        preds_dir = Path(args.exp_dir) / 'preds'
        preds_dir.mkdir(parents=True, exist_ok=True)
        if not (preds_dir / 'phn_target.npy').exists():
            np.save(preds_dir / 'phn_target.npy', A_phn_target)
            np.save(preds_dir / 'word_target.npy', valid_word_target)
            np.save(preds_dir / 'utt_target.npy', A_utt_target)
        np.save(preds_dir / 'phn_pred.npy', A_phn)
        np.save(preds_dir / 'word_pred.npy', valid_word_pred)
        np.save(preds_dir / 'utt_pred.npy', A_utt)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr


def train(audio_model, train_loader, test_loader, args, device):
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)
    loss_fn = nn.MSELoss()
    result = np.zeros([args.n_epochs, 32])

    while epoch < args.n_epochs:
        audio_model.train()
        for audio_input, phn_label, phns, utt_label, word_label in train_loader:
            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            phns = phns.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)

            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr

            noise = (torch.rand_like(audio_input) - 1) * args.noise
            audio_input = audio_input + noise

            u1, u2, u3, u4, u5, p, w1, w2, w3 = audio_model(audio_input, phns)
            mask = (phn_label >= 0)
            p = p.squeeze(2) * mask
            phn_label = phn_label * mask
            loss_phn = loss_fn(p, phn_label)
            loss_phn = loss_phn * (mask.shape[0] * mask.shape[1]) / torch.sum(mask)

            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds, utt_label)

            word_label = word_label[:, :, 0:3]
            mask = (word_label >= 0)
            word_pred = torch.cat((w1, w2, w3), dim=2) * mask
            word_label = word_label * mask
            loss_word = loss_fn(word_pred, word_label)
            loss_word = loss_word * (mask.shape[0] * mask.shape[1] * mask.shape[2]) / torch.sum(mask)

            loss = args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr = validate(audio_model, train_loader, args, -1, device)
        te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr = validate(audio_model, test_loader, args, best_mse, device)

        result[epoch, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]['lr']]
        result[epoch, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])
        result[epoch, 26:32] = np.concatenate([tr_word_corr, te_word_corr])
        np.savetxt(exp_dir / 'result.csv', result, delimiter=',', header=','.join(gen_result_header()), comments='')

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch

        if best_epoch == epoch:
            models_dir = exp_dir / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(audio_model.state_dict(), models_dir / 'best_audio_model.pth')

        if global_step > warm_up_step:
            scheduler.step()
        epoch += 1


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    tr_dataset = SeqDataset('train', data_dir, metadata)
    te_dataset = SeqDataset('test', data_dir, metadata)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    te_loader = DataLoader(te_dataset, batch_size=len(te_dataset), shuffle=False, num_workers=args.num_workers)

    input_dim = int(metadata['feat_dim'])
    phn_num = int(metadata['phn_num'])
    seq_len = int(metadata['seq_len'])

    if args.model == 'gopt':
        audio_model = GOPT(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim, seq_len=seq_len, phn_num=phn_num)
    elif args.model == 'gopt_nophn':
        audio_model = GOPTNoPhn(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim, seq_len=seq_len, phn_num=phn_num)
    else:
        audio_model = BaselineLSTM(embed_dim=args.embed_dim, depth=args.goptdepth, input_dim=input_dim, phn_num=phn_num)

    config = {
        'data_dir': str(data_dir),
        'input_dim': input_dim,
        'phn_num': phn_num,
        'seq_len': seq_len,
        'args': vars(args),
    }
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.exp_dir) / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    train(audio_model, tr_loader, te_loader, args, device)


if __name__ == '__main__':
    main()
