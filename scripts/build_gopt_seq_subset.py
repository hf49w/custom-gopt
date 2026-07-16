import argparse
import json
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Build a GOPT seq-data subset for a target utterance list."
    )
    parser.add_argument("--utt-id-list", type=Path, required=True)
    parser.add_argument("--seq-data-dir", type=Path, required=True, help="Directory containing te_feat.npy / te_label_*.npy")
    parser.add_argument("--keys-phn-csv", type=Path, required=True, help="te_keys_phn.csv matching the seq_data test arrays")
    parser.add_argument("--prefix", type=str, default="te", choices=["tr", "te"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_utt_order(keys_path: Path):
    keys = np.loadtxt(keys_path, delimiter=",", dtype=str)
    utt_order = []
    prev = None
    for key in keys.tolist():
        utt_id = str(key).split(".")[0]
        if utt_id != prev:
            utt_order.append(utt_id)
            prev = utt_id
    return utt_order


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_ids = [line.strip() for line in args.utt_id_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    utt_order = load_utt_order(args.keys_phn_csv)
    id_to_idx = {utt_id: idx for idx, utt_id in enumerate(utt_order)}

    missing = [utt_id for utt_id in target_ids if utt_id not in id_to_idx]
    if missing:
        raise KeyError(f"Missing utt_ids in keys file: {missing[:10]}")

    indices = np.array([id_to_idx[utt_id] for utt_id in target_ids], dtype=np.int64)

    feat = np.load(args.seq_data_dir / f"{args.prefix}_feat.npy")
    label_phn = np.load(args.seq_data_dir / f"{args.prefix}_label_phn.npy")
    label_word = np.load(args.seq_data_dir / f"{args.prefix}_label_word.npy")
    label_utt = np.load(args.seq_data_dir / f"{args.prefix}_label_utt.npy")

    if feat.shape[0] != len(utt_order):
        raise ValueError(
            f"Row count mismatch: feat_rows={feat.shape[0]} utt_order={len(utt_order)}"
        )

    np.save(args.output_dir / f"{args.prefix}_feat.npy", feat[indices])
    np.save(args.output_dir / f"{args.prefix}_label_phn.npy", label_phn[indices])
    np.save(args.output_dir / f"{args.prefix}_label_word.npy", label_word[indices])
    np.save(args.output_dir / f"{args.prefix}_label_utt.npy", label_utt[indices])
    (args.output_dir / f"{args.prefix}_utt_ids.txt").write_text(
        "".join(f"{utt_id}\n" for utt_id in target_ids),
        encoding="utf-8",
    )

    summary = {
        "count": int(len(target_ids)),
        "prefix": args.prefix,
        "output_dir": str(args.output_dir),
        "feat_shape": list(feat[indices].shape),
        "utt_ids_path": str(args.output_dir / f"{args.prefix}_utt_ids.txt"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
