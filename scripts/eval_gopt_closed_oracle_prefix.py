import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


NORM_MEAN = 3.203
NORM_STD = 4.045
SCORE_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]
WORD_SCORE_NAMES = ["accuracy", "stress", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the original GOPT checkpoint on oracle GT prefixes. "
            "For all-chunk streaming evaluation, each chunk is converted to a "
            "prefix-level closed-set sample using complete GT words only."
        )
    )
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--seq-data-dir", type=Path, required=True)
    parser.add_argument("--keys-phn-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--word-count-source",
        choices=["gt_time", "manifest_field"],
        default="gt_time",
        help="gt_time uses Charsiu GT word end times; manifest_field uses --word-count-field.",
    )
    parser.add_argument(
        "--word-count-field",
        default="cumulative_committed_word_count",
        help="Manifest field used when --word-count-source=manifest_field or as an optional cap.",
    )
    parser.add_argument(
        "--time-field",
        choices=["audio_end", "commit_time"],
        default="audio_end",
        help="Prefix cutoff for gt_time. audio_end includes right context; commit_time is stricter causal time.",
    )
    parser.add_argument(
        "--cap-word-count-field",
        default=None,
        help="Optional manifest word-count field used as an upper cap after gt_time counting.",
    )
    parser.add_argument("--multipa-repo-root", type=Path, default=None)
    parser.add_argument("--aligner", default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--align-device", default=None)
    parser.add_argument("--word-time-cache", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_word(word):
    return str(word or "").strip().lower()


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def text_from_scores(score_item):
    if score_item.get("text"):
        return str(score_item["text"])
    words = score_item.get("words") or []
    return " ".join(str(word.get("text", "")) for word in words)


def load_utt_order(keys_path, seq_data_dir):
    utt_ids_path = seq_data_dir / "te_utt_ids.txt"
    if utt_ids_path.exists():
        return [
            line.strip()
            for line in utt_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if keys_path is None:
        raise FileNotFoundError("Pass --keys-phn-csv when seq-data-dir has no te_utt_ids.txt")
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


def load_model(repo_src, checkpoint, device):
    repo_src = str(repo_src)
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from models import GOPT

    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    state = torch.load(checkpoint, map_location="cpu")
    clean_state = OrderedDict()
    for key, value in state.items():
        clean_state[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def load_charsiu(multipa_repo_root, aligner_name, align_device):
    if multipa_repo_root is None:
        raise ValueError("--multipa-repo-root is required when --word-count-source=gt_time")
    root = str(multipa_repo_root.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for module_name in ["models", "Charsiu", "utils_assessment"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file and root not in str(Path(module_file).resolve()):
            sys.modules.pop(module_name, None)
    charsiu_module = importlib.import_module("Charsiu")
    utils_module = importlib.import_module("utils_assessment")
    kwargs = {"aligner": aligner_name}
    if align_device:
        kwargs["device"] = align_device
    return charsiu_module.charsiu_forced_aligner(**kwargs), utils_module.get_match_index


def cache_key(text, wav_path):
    raw = f"{wav_path}\n{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_word_rows(pred_words):
    rows = []
    for item in np.asarray(pred_words).tolist():
        if len(item) < 3:
            continue
        start = safe_float(item[0])
        end = safe_float(item[1])
        if start is None or end is None or end <= start:
            continue
        rows.append({"start": start, "end": end, "word": normalize_word(item[2])})
    return rows


def align_gt_words(aligner, get_match_index, wav_path, text, cache_dir):
    text = " ".join(str(text or "").split())
    if not text:
        return []
    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{cache_key(text, wav_path)}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("status") == "ok":
                return payload.get("words", [])
            return []
    try:
        pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(
            audio=str(wav_path),
            text=text,
        )
        try:
            selected = get_match_index(pred_words, words)
            pred_words = np.asarray(pred_words)[selected]
        except Exception:
            pass
        rows = parse_word_rows(pred_words)
        payload = {"status": "ok", "words": rows}
    except Exception as exc:
        rows = []
        payload = {"status": "error", "error": repr(exc), "words": []}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return rows


def valid_word_ids(word_label):
    ids = []
    seen = set()
    for value in word_label[:, 3]:
        word_id = int(value)
        if word_id < 0:
            continue
        if word_id not in seen:
            seen.add(word_id)
            ids.append(word_id)
    return ids


def phone_len_for_word_count(word_label, word_count):
    if word_count <= 0:
        return 0
    ids = valid_word_ids(word_label)
    keep = set(ids[:word_count])
    if not keep:
        return 0
    positions = np.flatnonzero(np.isin(word_label[:, 3].astype(np.int64), list(keep)))
    if positions.size == 0:
        return 0
    return int(positions.max() + 1)


def aggregate_words(word_pred, word_target, phone_len):
    rows = []
    word_ids = word_target[:phone_len, 3].astype(np.int64)
    for word_id in valid_word_ids(word_target[:phone_len]):
        positions = np.where(word_ids == int(word_id))[0]
        if positions.size == 0:
            continue
        pred = word_pred[positions].mean(axis=0) * 5.0
        target = word_target[positions, :3].mean(axis=0)
        rows.append(
            {
                "word_id": int(word_id),
                "pred_accuracy": float(pred[0]),
                "pred_stress": float(pred[1]),
                "pred_total": float(pred[2]),
                "target_accuracy": float(target[0]),
                "target_stress": float(target[1]),
                "target_total": float(target[2]),
                "target_valid": True,
            }
        )
    return rows


def aggregate_phones(phone_pred, phone_target, phone_len):
    rows = []
    for idx in range(phone_len):
        target = float(phone_target[idx, 1])
        if target < 0:
            continue
        rows.append(
            {
                "phone_index": int(idx),
                "pred_accuracy": float(phone_pred[idx, 0]),
                "target_accuracy": target,
                "target_valid": True,
            }
        )
    return rows


def resolve_source_utt_id(row):
    return str(row.get("source_utt_id") or str(row["utt_id"]).split("_c", 1)[0])


def count_words_by_time(word_times, cutoff):
    return sum(1 for item in word_times if safe_float(item.get("end")) is not None and float(item["end"]) <= cutoff + 1e-6)


def get_manifest_word_count(row, field):
    value = row.get(field)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def main():
    args = parse_args()
    rows = load_jsonl(args.prefix_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    scores = json.loads(args.scores_json.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = load_model(args.repo_src, args.checkpoint, device)

    utt_ids = load_utt_order(args.keys_phn_csv, args.seq_data_dir)
    utt_to_index = {utt_id: index for index, utt_id in enumerate(utt_ids)}
    feat_all = np.load(args.seq_data_dir / "te_feat.npy", mmap_mode="r")
    phn_all = np.load(args.seq_data_dir / "te_label_phn.npy", mmap_mode="r")
    word_all = np.load(args.seq_data_dir / "te_label_word.npy", mmap_mode="r")

    aligner = get_match_index = None
    gt_word_time_by_utt = {}
    if args.word_count_source == "gt_time":
        aligner, get_match_index = load_charsiu(args.multipa_repo_root, args.aligner, args.align_device)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "ok": 0, "empty_prefix": 0, "missing_utt": 0, "gt_alignment_failed": 0}
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            batch_feat = []
            batch_phn = []
            batch_meta = []
            started = time.perf_counter()
            for row in batch_rows:
                counts["rows"] += 1
                source_utt_id = resolve_source_utt_id(row)
                if source_utt_id not in utt_to_index:
                    record = dict(row)
                    record.update({"status": "missing_utt", "source_utt_id": source_utt_id})
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["missing_utt"] += 1
                    continue
                source_index = utt_to_index[source_utt_id]
                word_label = np.asarray(word_all[source_index], dtype=np.float32)
                max_word_count = len(valid_word_ids(word_label))
                if args.word_count_source == "manifest_field":
                    word_count = get_manifest_word_count(row, args.word_count_field)
                    word_count_source = args.word_count_field
                else:
                    if source_utt_id not in gt_word_time_by_utt:
                        score_item = scores.get(source_utt_id, {})
                        text = text_from_scores(score_item)
                        wav_path = Path(row.get("source_wav_path") or row.get("full_wav_path") or row.get("wav_path"))
                        if row.get("source_wav_path") is None and row.get("source_utt_id"):
                            # Prefix-audio manifests store cropped wav_path. Prefer full wav from the raw PCN row when available.
                            wav_path = Path(score_item.get("wav_path", wav_path))
                        gt_word_time_by_utt[source_utt_id] = align_gt_words(
                            aligner,
                            get_match_index,
                            wav_path,
                            text,
                            args.word_time_cache,
                        )
                    word_times = gt_word_time_by_utt[source_utt_id]
                    if not word_times:
                        record = dict(row)
                        record.update(
                            {
                                "status": "gt_alignment_failed",
                                "source_utt_id": source_utt_id,
                                "word_count_source": "charsiu_gt_word_time",
                            }
                        )
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts["gt_alignment_failed"] += 1
                        continue
                    cutoff = float(row[args.time_field])
                    word_count = count_words_by_time(word_times, cutoff)
                    word_count_source = f"charsiu_gt_word_end_lte_{args.time_field}"
                    if args.cap_word_count_field:
                        cap = get_manifest_word_count(row, args.cap_word_count_field)
                        if cap is not None:
                            word_count = min(word_count, cap)
                            word_count_source += f"_capped_by_{args.cap_word_count_field}"
                word_count = min(max_word_count, max(0, int(word_count or 0)))
                phone_len = phone_len_for_word_count(word_label, word_count)
                if phone_len <= 0:
                    record = dict(row)
                    record.update(
                        {
                            "status": "empty_prefix",
                            "source_utt_id": source_utt_id,
                            "model": "original_gopt",
                            "mode": "oracle_prefix_closed",
                            "word_count_source": word_count_source,
                            "effective_word_count": int(word_count),
                            "effective_phone_count": 0,
                        }
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["empty_prefix"] += 1
                    continue

                full_feat = np.asarray(feat_all[source_index], dtype=np.float32)
                full_phn = np.asarray(phn_all[source_index, :, 0], dtype=np.int64)
                feat = np.zeros_like(full_feat)
                feat[:phone_len] = (full_feat[:phone_len] - NORM_MEAN) / NORM_STD
                phn = np.full_like(full_phn, -1)
                phn[:phone_len] = full_phn[:phone_len]
                batch_feat.append(feat)
                batch_phn.append(phn)
                batch_meta.append((row, source_utt_id, source_index, word_count, phone_len, word_count_source))

            if not batch_meta:
                continue
            x = torch.from_numpy(np.stack(batch_feat)).to(device)
            p = torch.from_numpy(np.stack(batch_phn)).to(device)
            with torch.no_grad():
                outputs = model(x, p)
            elapsed_per_row = (time.perf_counter() - started) / max(1, len(batch_meta))
            utt_pred = torch.cat(outputs[:5], dim=1).cpu().numpy() * 5.0
            phone_pred = outputs[5].cpu().numpy()
            word_pred = torch.cat(outputs[6:9], dim=2).cpu().numpy()

            for local_index, meta in enumerate(batch_meta):
                row, source_utt_id, source_index, word_count, phone_len, word_count_source = meta
                target_scores = row.get("target_scores")
                if target_scores is None and source_utt_id in scores:
                    target_scores = {name: float(scores[source_utt_id][name]) for name in SCORE_NAMES}
                record = dict(row)
                record.update(
                    {
                        "status": "ok",
                        "source_utt_id": source_utt_id,
                        "model": "original_gopt",
                        "mode": "oracle_prefix_closed",
                        "uses_reference_text": True,
                        "uses_reference_phone_order": True,
                        "word_count_source": word_count_source,
                        "effective_word_count": int(word_count),
                        "effective_phone_count": int(phone_len),
                        "target_scores": target_scores,
                        "scores": {
                            name: float(utt_pred[local_index, score_index])
                            for score_index, name in enumerate(SCORE_NAMES)
                        },
                        "phone_scores": aggregate_phones(
                            phone_pred[local_index],
                            np.asarray(phn_all[source_index], dtype=np.float32),
                            phone_len,
                        ),
                        "word_scores": aggregate_words(
                            word_pred[local_index],
                            np.asarray(word_all[source_index], dtype=np.float32),
                            phone_len,
                        ),
                        "process_time_sec": elapsed_per_row,
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["ok"] += 1
            output.flush()
            print(
                f"[gopt-closed-oracle-prefix] {min(batch_start + len(batch_rows), len(rows))}/{len(rows)}",
                flush=True,
            )

    print(json.dumps({"output_jsonl": str(args.output_jsonl), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
