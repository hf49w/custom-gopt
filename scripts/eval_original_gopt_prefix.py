import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


NORM_MEAN = 3.203
NORM_STD = 4.045
SCORE_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the original GOPT checkpoint on truncated test features. "
            "This is an oracle phone-order prefix baseline, not an ASR-prefix system."
        )
    )
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--seq-data-dir", type=Path, required=True)
    parser.add_argument(
        "--keys-phn-csv",
        type=Path,
        default=None,
        help="Phone-key file that defines test-array utterance order.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_model(repo_src, checkpoint, device):
    import sys

    sys.path.insert(0, str(repo_src))
    from models import GOPT

    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    state = torch.load(checkpoint, map_location="cpu")
    clean_state = OrderedDict()
    for key, value in state.items():
        clean_state[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def load_utt_order(keys_path):
    utt_order = []
    previous = None
    for line in keys_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id = line.split(",", 1)[0].rsplit(".", 1)[0]
        if utt_id != previous:
            utt_order.append(utt_id)
            previous = utt_id
    return utt_order


def aggregate_words(word_pred, word_target, prefix_len):
    word_ids = word_target[:prefix_len, 3]
    rows = []
    for word_id in sorted(set(int(value) for value in word_ids if value >= 0)):
        positions = np.where(word_ids == word_id)[0]
        pred = word_pred[positions].mean(axis=0) * 5.0
        target = word_target[positions, :3].mean(axis=0)
        rows.append(
            {
                "word_id": word_id,
                "pred_accuracy": float(pred[0]),
                "pred_stress": float(pred[1]),
                "pred_total": float(pred[2]),
                "target_accuracy": float(target[0]),
                "target_stress": float(target[1]),
                "target_total": float(target[2]),
            }
        )
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.repo_src, args.checkpoint, device)

    if args.keys_phn_csv is not None:
        utt_ids = load_utt_order(args.keys_phn_csv)
    else:
        utt_ids_path = args.seq_data_dir / "te_utt_ids.txt"
        if not utt_ids_path.exists():
            raise FileNotFoundError(
                "Pass --keys-phn-csv when seq-data-dir has no te_utt_ids.txt"
            )
        utt_ids = utt_ids_path.read_text(encoding="utf-8").splitlines()
    utt_to_index = {utt_id.strip(): index for index, utt_id in enumerate(utt_ids)}
    feat_all = np.load(args.seq_data_dir / "te_feat.npy", mmap_mode="r")
    phn_all = np.load(args.seq_data_dir / "te_label_phn.npy", mmap_mode="r")
    word_all = np.load(args.seq_data_dir / "te_label_word.npy", mmap_mode="r")
    if feat_all.shape[0] != len(utt_ids):
        raise ValueError(
            f"Array/order mismatch: feat_rows={feat_all.shape[0]} utt_ids={len(utt_ids)}"
        )
    rows = [
        json.loads(line)
        for line in args.prefix_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch_feat = []
            batch_phn = []
            prefix_lengths = []
            source_indices = []
            started = time.perf_counter()

            for row in batch_rows:
                utt_id = row["utt_id"]
                if utt_id not in utt_to_index:
                    raise KeyError(f"utt_id={utt_id} not found in te_utt_ids.txt")
                source_index = utt_to_index[utt_id]
                full_feat = np.asarray(feat_all[source_index], dtype=np.float32)
                full_phn = np.asarray(phn_all[source_index, :, 0], dtype=np.int64)
                full_len = int(np.sum(full_feat[:, 0] != 0))
                prefix_len = min(
                    int(row["visible_phone_count"]), full_len, full_feat.shape[0]
                )

                feat = np.zeros_like(full_feat)
                if prefix_len > 0:
                    feat[:prefix_len] = (
                        full_feat[:prefix_len] - NORM_MEAN
                    ) / NORM_STD
                phn = np.full_like(full_phn, -1)
                phn[:prefix_len] = full_phn[:prefix_len]
                batch_feat.append(feat)
                batch_phn.append(phn)
                prefix_lengths.append(prefix_len)
                source_indices.append(source_index)

            x = torch.from_numpy(np.stack(batch_feat)).to(device)
            p = torch.from_numpy(np.stack(batch_phn)).to(device)
            with torch.no_grad():
                outputs = model(x, p)
            elapsed_per_row = (time.perf_counter() - started) / len(batch_rows)
            utt_pred = torch.cat(outputs[:5], dim=1).cpu().numpy() * 5.0
            word_pred = torch.cat(outputs[6:9], dim=2).cpu().numpy()

            for local_index, row in enumerate(batch_rows):
                source_index = source_indices[local_index]
                prefix_len = prefix_lengths[local_index]
                record = dict(row)
                record.update(
                    {
                        "model": "original_gopt",
                        "mode": "oracle_reference_phone_prefix",
                        "status": "ok" if prefix_len > 0 else "empty_prefix",
                        "uses_reference_phone_order": True,
                        "prefix_count_source": "visible_phone_count",
                        "batch_size": args.batch_size,
                        "timing_mode": (
                            "online_single_prefix"
                            if args.batch_size == 1
                            else "batch_amortized"
                        ),
                        "effective_phone_count": prefix_len,
                        "process_time_sec": elapsed_per_row,
                        "scores": {
                            name: float(utt_pred[local_index, index])
                            for index, name in enumerate(SCORE_NAMES)
                        },
                        "word_scores": aggregate_words(
                            word_pred[local_index],
                            np.asarray(word_all[source_index]),
                            prefix_len,
                        ),
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            print(
                f"[original-gopt] {min(batch_start + len(batch_rows), len(rows))}/{len(rows)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
