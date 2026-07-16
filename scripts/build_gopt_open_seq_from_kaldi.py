import argparse
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import kaldi_io
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Build GOPT-compatible seq arrays from a Kaldi GOP feature SCP and pseudo open-set scores."
    )
    parser.add_argument("--feature-scp", type=Path, required=True)
    parser.add_argument("--pseudo-scores-json", type=Path, required=True)
    parser.add_argument("--reference-train-labels-phn-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="te")
    parser.add_argument("--max-seq-len", type=int, default=50)
    return parser.parse_args()


def strip_phone_markers(phone: str) -> str:
    return re.sub(r"[_\d].*$", "", phone)


def load_reference_phone_map(path: Path):
    labels = np.loadtxt(path, delimiter=",", dtype=str)
    phn_dict = OrderedDict()
    for row in labels:
        phone = str(row[0])
        if phone not in phn_dict:
            phn_dict[phone] = len(phn_dict)
    return phn_dict


def load_pseudo_scores(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_rxfile(rxfile: str, scp_path: Path) -> str:
    # Keep Kaldi pipes/specifiers unchanged.
    if rxfile.startswith(("ark:", "scp:", "|")):
        return rxfile

    path_part = rxfile
    offset = None
    if ":" in rxfile:
        maybe_path, maybe_offset = rxfile.rsplit(":", 1)
        if maybe_offset.isdigit():
            path_part = maybe_path
            offset = maybe_offset

    candidate_path = Path(path_part)
    if candidate_path.is_absolute():
        resolved = candidate_path
    else:
        resolved = None
        for anchor in [scp_path.parent, *scp_path.parents]:
            probe = (anchor / candidate_path).resolve()
            if probe.exists():
                resolved = probe
                break
        if resolved is None:
            resolved = candidate_path

    if offset is not None:
        return f"{resolved}:{offset}"
    return str(resolved)


def materialize_resolved_scp(feature_scp: Path) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix="gopt_open_feat_", suffix=".scp")
    tmp_path = Path(tmp_name)
    with feature_scp.open("r", encoding="utf-8") as src, os.fdopen(fd, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, rxfile = line.split(maxsplit=1)
            dst.write(f"{key} {resolve_rxfile(rxfile, feature_scp)}\n")
    return tmp_path


def build_key_map(pseudo_scores: dict):
    key_map = {}
    utt_word_text = {}
    for utt_id, utt in pseudo_scores.items():
        phone_num = 0
        words = []
        for word_id, word in enumerate(utt["words"]):
            word_text = str(word["text"]).lower()
            words.append(word_text)
            for phone in word["phones"]:
                key = f"{utt_id}.{phone_num}"
                key_map[key] = {
                    "utt_id": utt_id,
                    "token_idx": phone_num,
                    "pure_phone": strip_phone_markers(phone),
                    "word_id": word_id,
                    "word_text": word_text,
                }
                phone_num += 1
        utt_word_text[utt_id] = words
    return key_map, utt_word_text


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phn_dict = load_reference_phone_map(args.reference_train_labels_phn_csv)
    pseudo_scores = load_pseudo_scores(args.pseudo_scores_json)
    key_map, utt_word_text = build_key_map(pseudo_scores)

    feat_rows = OrderedDict()
    phn_rows = OrderedDict()
    word_rows = OrderedDict()
    skipped_keys = []
    feat_dim = None

    resolved_scp = materialize_resolved_scp(args.feature_scp)
    try:
        for key, feat in kaldi_io.read_vec_flt_scp(str(resolved_scp)):
            meta = key_map.get(key)
            if meta is None:
                skipped_keys.append(key)
                continue

            utt_id = meta["utt_id"]
            token_idx = meta["token_idx"]
            pure_phone = meta["pure_phone"]
            word_id = meta["word_id"]

            if pure_phone not in phn_dict:
                raise KeyError(f"Phone '{pure_phone}' not found in reference phone map.")

            cur_feat = np.asarray(feat[1:], dtype=np.float32)
            if feat_dim is None:
                feat_dim = int(cur_feat.shape[0])
            if token_idx >= args.max_seq_len:
                raise ValueError(f"{utt_id} token_idx={token_idx} exceeds max_seq_len={args.max_seq_len}")

            if utt_id not in feat_rows:
                feat_rows[utt_id] = np.zeros((args.max_seq_len, feat_dim), dtype=np.float32)
                phn_rows[utt_id] = np.full((args.max_seq_len, 2), -1.0, dtype=np.float32)
                word_rows[utt_id] = np.full((args.max_seq_len, 4), -1.0, dtype=np.float32)

            feat_rows[utt_id][token_idx] = cur_feat
            phn_rows[utt_id][token_idx, 0] = float(phn_dict[pure_phone])
            phn_rows[utt_id][token_idx, 1] = 2.0
            word_rows[utt_id][token_idx, 0:3] = 0.0
            word_rows[utt_id][token_idx, 3] = float(word_id)
    finally:
        resolved_scp.unlink(missing_ok=True)

    utt_ids = list(feat_rows.keys())
    if not utt_ids:
        raise ValueError("No utterances were recovered from feature-scp.")

    feat_arr = np.stack([feat_rows[utt_id] for utt_id in utt_ids], axis=0)
    phn_arr = np.stack([phn_rows[utt_id] for utt_id in utt_ids], axis=0)
    word_arr = np.stack([word_rows[utt_id] for utt_id in utt_ids], axis=0)

    np.save(args.output_dir / f"{args.prefix}_feat.npy", feat_arr)
    np.save(args.output_dir / f"{args.prefix}_label_phn.npy", phn_arr)
    np.save(args.output_dir / f"{args.prefix}_label_word.npy", word_arr)
    (args.output_dir / f"{args.prefix}_utt_ids.txt").write_text("".join(f"{utt_id}\n" for utt_id in utt_ids), encoding="utf-8")
    with (args.output_dir / f"{args.prefix}_word_text.json").open("w", encoding="utf-8") as f:
        json.dump({utt_id: utt_word_text.get(utt_id, []) for utt_id in utt_ids}, f, ensure_ascii=False, indent=2)

    summary = {
        "utt_count": len(utt_ids),
        "feat_shape": list(feat_arr.shape),
        "phn_shape": list(phn_arr.shape),
        "word_shape": list(word_arr.shape),
        "skipped_key_count": len(skipped_keys),
        "skipped_key_examples": skipped_keys[:20],
        "resolved_feature_scp_from": str(args.feature_scp),
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
