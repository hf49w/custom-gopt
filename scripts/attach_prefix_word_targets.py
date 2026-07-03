import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np


WORD_SCORE_NAMES = ["accuracy", "stress", "total"]
DEFAULT_NO_OVERLAP_TARGET = {"accuracy": 0.0, "stress": 5.0, "total": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach GT word targets to per-prefix GOPT-open/MultiPA JSONL outputs "
            "by Charsiu word-time overlap."
        )
    )
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument(
        "--prefix-manifest",
        type=Path,
        required=True,
        help="Shared prefix manifest with original/full wav paths.",
    )
    parser.add_argument(
        "--prefix-audio-manifest",
        type=Path,
        default=None,
        help="Optional prefix-audio manifest. Used to resolve cropped prefix wav paths.",
    )
    parser.add_argument("--multipa-repo-root", type=Path, required=True)
    parser.add_argument("--aligner", type=str, default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--align-device", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--no-overlap-policy",
        choices=["skip", "default"],
        default="skip",
        help="skip leaves target_valid=false; default writes MultiPA open-eval fallback targets.",
    )
    parser.add_argument("--min-overlap-sec", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_word(word):
    return str(word or "").strip().lower()


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def interval_from_row(row):
    start = safe_float(row.get("start") if isinstance(row, dict) else row[0])
    end = safe_float(row.get("end") if isinstance(row, dict) else row[1])
    if start is None or end is None or end <= start:
        return None
    return start, end


def cache_key(*parts):
    raw = "\n".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fit_list(values, size, fill_value=None):
    values = list(values or [])
    if len(values) >= size:
        return values[:size]
    return values + [fill_value for _ in range(size - len(values))]


def load_manifest_maps(prefix_manifest, prefix_audio_manifest):
    by_index = {}
    by_eval_id = {}
    by_utt_chunk = {}
    for row in load_jsonl(prefix_manifest):
        idx = int(row.get("manifest_row_index", len(by_index)))
        by_index[idx] = row
        by_utt_chunk[(str(row["utt_id"]), int(row["chunk_id"]))] = row
    audio_by_eval = {}
    if prefix_audio_manifest is not None and prefix_audio_manifest.exists():
        for row in load_jsonl(prefix_audio_manifest):
            eval_id = str(row["utt_id"])
            audio_by_eval[eval_id] = row
            if "manifest_row_index" in row:
                by_eval_id[eval_id] = row
    return by_index, by_eval_id, by_utt_chunk, audio_by_eval


def resolve_source_row(row, by_index, by_eval_id, by_utt_chunk):
    if "manifest_row_index" in row:
        item = by_index.get(int(row["manifest_row_index"]))
        if item is not None:
            return item
    eval_id = str(row.get("eval_id", ""))
    if eval_id and eval_id in by_eval_id:
        return by_eval_id[eval_id]
    utt_id = str(row.get("source_utt_id", row.get("utt_id", "")))
    chunk_id = row.get("chunk_id")
    if chunk_id is not None:
        item = by_utt_chunk.get((utt_id, int(chunk_id)))
        if item is not None:
            return item
    return row


def resolve_eval_id(row):
    if row.get("eval_id"):
        return str(row["eval_id"])
    if row.get("source_utt_id") and row.get("chunk_id") is not None:
        return f"{row['source_utt_id']}_c{int(row['chunk_id']):04d}"
    if row.get("utt_id") and row.get("chunk_id") is not None:
        return f"{row['utt_id']}_c{int(row['chunk_id']):04d}"
    return str(row.get("utt_id", "unknown"))


def resolve_prefix_wav(row, audio_by_eval):
    eval_id = resolve_eval_id(row)
    audio_row = audio_by_eval.get(eval_id)
    if audio_row is not None and audio_row.get("wav_path"):
        return Path(audio_row["wav_path"])
    return Path(row["wav_path"])


def load_charsiu(multipa_repo_root, aligner_name, align_device):
    root = str(multipa_repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    charsiu_module = importlib.import_module("Charsiu")
    utils_module = importlib.import_module("utils_assessment")
    kwargs = {"aligner": aligner_name}
    if align_device:
        kwargs["device"] = align_device
    return charsiu_module.charsiu_forced_aligner(**kwargs), utils_module.get_match_index


def charsiu_align_words(aligner, get_match_index, audio_path, text, cache_path=None):
    text = " ".join(str(text or "").split())
    if not text:
        return []
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(
        audio=str(audio_path),
        text=text,
    )
    selected = get_match_index(pred_words, words)
    pred_words = np.asarray(pred_words)[selected]
    rows = []
    for item in pred_words.tolist():
        if len(item) < 3:
            continue
        start = safe_float(item[0])
        end = safe_float(item[1])
        if start is None or end is None or end <= start:
            continue
        rows.append({"start": start, "end": end, "word": normalize_word(item[2])})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def gt_text_and_scores(scores, utt_id):
    item = scores[str(utt_id)]
    text = item.get("text") or " ".join(word["text"] for word in item.get("words", []))
    word_scores = []
    for word in item.get("words", []):
        word_scores.append(
            {
                "word": normalize_word(word.get("text")),
                "accuracy": float(word.get("accuracy", 0.0)),
                "stress": float(word.get("stress", 0.0)),
                "total": float(word.get("total", 0.0)),
            }
        )
    return text, word_scores


def prediction_text(row):
    word_scores = row.get("word_scores") or []
    words = [normalize_word(item.get("word")) for item in word_scores if item.get("word")]
    if words:
        return " ".join(words)
    if row.get("asr_text"):
        return str(row["asr_text"])
    asr = row.get("asr")
    if isinstance(asr, dict):
        return asr.get("word_model_text") or asr.get("sentence_model_text") or ""
    return ""


def existing_pred_times(row):
    rows = []
    for item in row.get("word_scores") or []:
        interval = interval_from_row(item)
        if interval is None:
            return []
        rows.append(
            {
                "start": interval[0],
                "end": interval[1],
                "word": normalize_word(item.get("word")),
            }
        )
    return rows


def overlap_amount(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def attach_targets_to_words(row, pred_times, gt_times, gt_scores, no_overlap_policy, min_overlap_sec):
    word_scores = list(row.get("word_scores") or [])
    if not word_scores:
        row["word_target_status"] = "no_word_scores"
        return row
    pred_times = fit_list(pred_times, len(word_scores), None)
    for idx, item in enumerate(word_scores):
        pred_time = pred_times[idx]
        if pred_time is not None:
            item["start"] = pred_time["start"]
            item["end"] = pred_time["end"]
            if pred_time.get("word") and not item.get("word"):
                item["word"] = pred_time["word"]
        overlaps = []
        if pred_time is not None:
            for gt_idx, gt_time in enumerate(gt_times[: len(gt_scores)]):
                amount = overlap_amount(pred_time, gt_time)
                if amount >= min_overlap_sec:
                    overlaps.append((gt_idx, amount))
        if overlaps:
            denom = sum(weight for _, weight in overlaps)
            for name in WORD_SCORE_NAMES:
                item[f"target_{name}"] = float(
                    sum(gt_scores[gt_idx][name] * weight for gt_idx, weight in overlaps) / denom
                )
            item["target_valid"] = True
            item["target_source"] = "charsiu_time_overlap"
            item["target_gt_indices"] = [int(gt_idx) for gt_idx, _ in overlaps]
            item["target_overlap_sec"] = float(denom)
        elif no_overlap_policy == "default":
            for name, value in DEFAULT_NO_OVERLAP_TARGET.items():
                item[f"target_{name}"] = value
            item["target_valid"] = True
            item["target_source"] = "no_overlap_default"
            item["target_gt_indices"] = []
            item["target_overlap_sec"] = 0.0
        else:
            item["target_valid"] = False
            item["target_source"] = "no_overlap"
            item["target_gt_indices"] = []
            item["target_overlap_sec"] = 0.0
        word_scores[idx] = item
    row["word_scores"] = word_scores
    row["word_target_status"] = "attached"
    return row


def load_completed(path):
    completed = {}
    if not path.exists():
        return completed
    for row in load_jsonl(path):
        completed[(str(row.get("utt_id")), int(row.get("chunk_id", -1)), resolve_eval_id(row))] = row
    return completed


def main():
    args = parse_args()
    rows = load_jsonl(args.prediction_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    scores = json.loads(args.scores_json.read_text(encoding="utf-8"))
    by_index, by_eval_id, by_utt_chunk, audio_by_eval = load_manifest_maps(
        args.prefix_manifest,
        args.prefix_audio_manifest,
    )
    cache_dir = args.cache_dir or (args.output_jsonl.parent / "charsiu_word_time_cache")
    completed = load_completed(args.output_jsonl) if args.resume else {}
    aligner, get_match_index = load_charsiu(args.multipa_repo_root, args.aligner, args.align_device)

    output_rows = []
    counts = {"rows": 0, "ok": 0, "reused": 0, "attached": 0, "errors": 0}
    for row_idx, row in enumerate(rows, start=1):
        counts["rows"] += 1
        key = (str(row.get("utt_id")), int(row.get("chunk_id", -1)), resolve_eval_id(row))
        if key in completed:
            output_rows.append(completed[key])
            counts["reused"] += 1
            continue
        if row.get("status", "ok") != "ok":
            output_rows.append(row)
            continue
        try:
            source_row = resolve_source_row(row, by_index, by_eval_id, by_utt_chunk)
            source_utt_id = str(source_row.get("utt_id", row.get("source_utt_id", row.get("utt_id"))))
            full_wav = Path(source_row["wav_path"])
            prefix_wav = resolve_prefix_wav(row, audio_by_eval)
            gt_text, gt_scores = gt_text_and_scores(scores, source_utt_id)
            gt_key = cache_key("gt", source_utt_id, full_wav, gt_text)
            gt_times = charsiu_align_words(
                aligner,
                get_match_index,
                full_wav,
                gt_text,
                cache_dir / "gt" / f"{source_utt_id}_{gt_key}.json",
            )
            pred_times = existing_pred_times(row)
            if not pred_times:
                pred_text = prediction_text(row)
                pred_key = cache_key("pred", resolve_eval_id(row), prefix_wav, pred_text)
                pred_times = charsiu_align_words(
                    aligner,
                    get_match_index,
                    prefix_wav,
                    pred_text,
                    cache_dir / "pred" / f"{resolve_eval_id(row)}_{pred_key}.json",
                )
            row = attach_targets_to_words(
                row,
                pred_times,
                gt_times,
                gt_scores,
                args.no_overlap_policy,
                args.min_overlap_sec,
            )
            row["word_target_alignment"] = {
                "gt_word_time_source": "charsiu_forced_alignment_full_audio_gt_text",
                "pred_word_time_source": "existing_word_times_or_charsiu_forced_alignment_prefix_audio_pred_text",
                "no_overlap_policy": args.no_overlap_policy,
            }
            counts["attached"] += 1
            counts["ok"] += 1
        except Exception as exc:
            row["word_target_status"] = "error"
            row["word_target_error"] = repr(exc)
            counts["errors"] += 1
        output_rows.append(row)
        if row_idx % 100 == 0 or row_idx == len(rows):
            print(f"[attach-word-targets] {row_idx}/{len(rows)} {counts}", flush=True)

    write_jsonl(args.output_jsonl, output_rows)
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
