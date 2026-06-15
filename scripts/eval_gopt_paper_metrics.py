import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


NORM_STATS = {
    "librispeech": (3.203, 4.045, 84),
    "paiia": (-0.652, 9.737, 86),
    "paiib": (-0.516, 9.247, 88),
}

REFERENCE_METRICS = {
    "gopt_paper": {
        "phone_pcc": 0.612,
        "word_acc_pcc": 0.533,
        "word_stress_pcc": 0.291,
        "word_total_pcc": 0.549,
        "utt_acc_pcc": 0.714,
        "utt_completeness_pcc": 0.155,
        "utt_fluency_pcc": 0.753,
        "utt_prosodic_pcc": 0.760,
        "utt_total_pcc": 0.742,
    },
    "gopt_readme_librispeech": {
        "phone_pcc": 0.616,
        "word_acc_pcc": 0.536,
        "word_stress_pcc": 0.326,
        "word_total_pcc": 0.552,
        "utt_acc_pcc": 0.718,
        "utt_completeness_pcc": 0.109,
        "utt_fluency_pcc": 0.756,
        "utt_prosodic_pcc": 0.764,
        "utt_total_pcc": 0.743,
    },
    "multipa_gopt_open": {
        "word_acc_pcc": 0.273,
        "word_stress_pcc": 0.067,
        "word_total_pcc": 0.265,
        "utt_acc_pcc": 0.528,
        "utt_fluency_pcc": 0.527,
        "utt_prosodic_pcc": 0.545,
        "utt_total_pcc": 0.528,
    },
    "multipa": {
        "word_acc_pcc": 0.427,
        "word_stress_pcc": 0.239,
        "word_total_pcc": 0.436,
        "utt_acc_pcc": 0.705,
        "utt_fluency_pcc": 0.772,
        "utt_prosodic_pcc": 0.764,
        "utt_total_pcc": 0.730,
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a GOPT-format test subset with paper-style metrics.")
    parser.add_argument("--seq-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--am", type=str, default="librispeech", choices=["librispeech", "paiia", "paiib"])
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prefix", type=str, default="te", choices=["tr", "te"])
    parser.add_argument("--repo-src", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--compare-reference", type=str, default=None, choices=[None, "gopt_paper", "gopt_readme_librispeech", "multipa_gopt_open", "multipa"])
    return parser.parse_args()


class SeqDataset(Dataset):
    def __init__(self, seq_data_dir: Path, prefix: str, norm_mean: float, norm_std: float):
        self.feat = torch.tensor(np.load(seq_data_dir / f"{prefix}_feat.npy"), dtype=torch.float32)
        self.phn_label = torch.tensor(np.load(seq_data_dir / f"{prefix}_label_phn.npy"), dtype=torch.float32)
        self.utt_label = torch.tensor(np.load(seq_data_dir / f"{prefix}_label_utt.npy"), dtype=torch.float32) / 5.0
        self.word_label = torch.tensor(np.load(seq_data_dir / f"{prefix}_label_word.npy"), dtype=torch.float32)
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5.0
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

    @staticmethod
    def norm_valid(feat, norm_mean, norm_std):
        out = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    out[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return out

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return (
            self.feat[idx, :],
            self.phn_label[idx, :, 1],
            self.phn_label[idx, :, 0],
            self.utt_label[idx, :],
            self.word_label[idx, :],
        )


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
    return float(valid_token_mse), float(corr)


def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_pred = audio_output[:, i].cpu().numpy()
        cur_target = target[:, i].cpu().numpy()
        mse.append(float(np.mean((cur_pred - cur_target) ** 2)))
        corr.append(float(np.corrcoef(cur_pred, cur_target)[0, 1]))
    return mse, corr


def valid_word(audio_output, target):
    word_id = target[:, :, 3]
    valid_token_pred = []
    valid_token_target = []

    for i in range(target.shape[0]):
        cur_word_id = 0
        cur_word_pred = []
        cur_word_target = []
        for j in range(target.shape[1]):
            if word_id[i, j] >= 0:
                if word_id[i, j] == cur_word_id:
                    cur_word_pred.append(audio_output[i, j, :].cpu().numpy())
                    cur_word_target.append(target[i, j, 0:3].cpu().numpy())
                else:
                    valid_token_pred.append(np.mean(np.stack(cur_word_pred), axis=0))
                    valid_token_target.append(np.mean(np.stack(cur_word_target), axis=0))
                    cur_word_pred = [audio_output[i, j, :].cpu().numpy()]
                    cur_word_target = [target[i, j, 0:3].cpu().numpy()]
                    cur_word_id = word_id[i, j]
            else:
                break
        if cur_word_pred:
            valid_token_pred.append(np.mean(np.stack(cur_word_pred), axis=0))
            valid_token_target.append(np.mean(np.stack(cur_word_target), axis=0))

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list = []
    corr_list = []
    for i in range(3):
        mse_list.append(float(np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)))
        corr_list.append(float(np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]))
    return mse_list, corr_list


def load_model(repo_src: Path, checkpoint: Path, input_dim: int, embed_dim: int, num_heads: int, depth: int, device: torch.device):
    import sys

    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)
    from models import GOPT

    model = GOPT(embed_dim=embed_dim, num_heads=num_heads, depth=depth, input_dim=input_dim)
    state = torch.load(checkpoint, map_location="cpu")
    new_state = OrderedDict()
    for key, value in state.items():
        new_state[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(new_state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    a_phn, a_phn_target = [], []
    a_u1, a_u2, a_u3, a_u4, a_u5, a_utt_target = [], [], [], [], [], []
    a_w1, a_w2, a_w3, a_word_target = [], [], [], []

    with torch.no_grad():
        for audio_input, phn_label, phns, utt_label, word_label in loader:
            audio_input = audio_input.to(device)
            phns = phns.to(device)
            u1, u2, u3, u4, u5, p, w1, w2, w3 = model(audio_input, phns)
            a_phn.append(p.cpu())
            a_phn_target.append(phn_label)
            a_u1.append(u1.cpu())
            a_u2.append(u2.cpu())
            a_u3.append(u3.cpu())
            a_u4.append(u4.cpu())
            a_u5.append(u5.cpu())
            a_utt_target.append(utt_label)
            a_w1.append(w1.cpu())
            a_w2.append(w2.cpu())
            a_w3.append(w3.cpu())
            a_word_target.append(word_label)

    a_phn = torch.cat(a_phn)
    a_phn_target = torch.cat(a_phn_target)
    a_utt = torch.cat((torch.cat(a_u1), torch.cat(a_u2), torch.cat(a_u3), torch.cat(a_u4), torch.cat(a_u5)), dim=1)
    a_utt_target = torch.cat(a_utt_target)
    a_word = torch.cat((torch.cat(a_w1), torch.cat(a_w2), torch.cat(a_w3)), dim=2)
    a_word_target = torch.cat(a_word_target)

    phone_mse, phone_pcc = valid_phn(a_phn, a_phn_target)
    utt_mse, utt_pcc = valid_utt(a_utt, a_utt_target)
    word_mse, word_pcc = valid_word(a_word, a_word_target)

    return {
        "phone_mse": phone_mse,
        "phone_pcc": phone_pcc,
        "utt_mse": utt_mse,
        "utt_pcc": utt_pcc,
        "word_mse": word_mse,
        "word_pcc": word_pcc,
    }


def main():
    args = get_args()
    norm_mean, norm_std, input_dim = NORM_STATS[args.am]
    dataset = SeqDataset(args.seq_data_dir, args.prefix, norm_mean, norm_std)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    device = torch.device(args.device)
    model = load_model(args.repo_src, args.checkpoint, input_dim, args.embed_dim, args.num_heads, args.depth, device)
    metrics = evaluate(model, loader, device)

    payload = {
        "am": args.am,
        "prefix": args.prefix,
        "num_utterances": len(dataset),
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
    }
    if args.compare_reference:
        payload["reference_name"] = args.compare_reference
        payload["reference_metrics"] = REFERENCE_METRICS[args.compare_reference]

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
