import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attach_prefix_word_targets import (  # noqa: E402
    attach_targets_to_words,
    cache_key,
    existing_pred_times,
    fit_list,
    gt_text_and_scores,
    interval_from_row,
    load_completed,
    load_jsonl,
    load_manifest_maps,
    normalize_word,
    overlap_amount,
    prediction_text,
    resolve_eval_id,
    resolve_prefix_wav,
    resolve_source_row,
    safe_float,
    write_jsonl,
)


PHONE_NO_OVERLAP_TARGET = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach GT word and phone targets to GOPT-open/MultiPA prefix JSONL outputs. "
            "Word and phone targets are assigned by Charsiu time-overlap; GT is used only "
            "for evaluation labels."
        )
    )
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--prefix-audio-manifest", type=Path, default=None)
    parser.add_argument("--multipa-repo-root", type=Path, required=True)
    parser.add_argument("--aligner", type=str, default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--align-device", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--min-overlap-sec", type=float, default=1e-4)
    parser.add_argument(
        "--no-overlap-policy",
        choices=["skip", "default"],
        default="skip",
        help="For words, mirrors attach_prefix_word_targets.py. For phones, default writes 0.0.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_charsiu(multipa_repo_root, aligner_name, align_device):
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


def parse_time_rows(rows, include_phone=True):
    parsed = []
    for item in np.asarray(rows).tolist():
        if len(item) < 3:
            continue
        start = safe_float(item[0])
        end = safe_float(item[1])
        label = str(item[2])
        if start is None or end is None or end <= start:
            continue
        if include_phone and label == "[SIL]":
            continue
        parsed.append({"start": start, "end": end, "phone" if include_phone else "word": label})
    return parsed


def select_words(pred_words, words, get_match_index):
    try:
        selected = get_match_index(pred_words, words)
        pred_words = np.asarray(pred_words)[selected]
    except Exception:
        pass
    parsed = []
    for item in np.asarray(pred_words).tolist():
        if len(item) < 3:
            continue
        start = safe_float(item[0])
        end = safe_float(item[1])
        if start is None or end is None or end <= start:
            continue
        word = normalize_word(item[2])
        if word == "[sil]":
            continue
        parsed.append({"start": start, "end": end, "word": word})
    return parsed


def align_phone_word_times(aligner, get_match_index, audio_path, text, cache_path=None):
    text = " ".join(str(text or "").split())
    if not text:
        return {"phones": [], "words": []}
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(
        audio=str(audio_path),
        text=text,
    )
    payload = {
        "phones": parse_time_rows(pred_phones, include_phone=True),
        "words": select_words(pred_words, words, get_match_index),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def flat_gt_phone_scores(scores, utt_id):
    item = scores[str(utt_id)]
    rows = []
    for word_id, word in enumerate(item.get("words") or []):
        word_text = normalize_word(word.get("text"))
        phones = list(word.get("phones") or [])
        phone_scores = list(word.get("phones-accuracy") or [])
        for phone_in_word, phone in enumerate(phones):
            target = phone_scores[phone_in_word] if phone_in_word < len(phone_scores) else None
            try:
                target = float(target)
            except (TypeError, ValueError):
                target = None
            if target is None or not math.isfinite(target):
                continue
            rows.append(
                {
                    "phone": str(phone),
                    "word": word_text,
                    "word_id": int(word_id),
                    "phone_in_word": int(phone_in_word),
                    "accuracy": target,
                }
            )
    return rows


def attach_targets_to_phones(row, pred_times, gt_times, gt_phone_scores, no_overlap_policy, min_overlap_sec):
    phone_scores = list(row.get("phone_scores") or [])
    if not phone_scores:
        row["phone_target_status"] = "no_phone_scores"
        return row

    gt_times = fit_list(gt_times, len(gt_phone_scores), None)
    gt_entries = []
    for idx, target in enumerate(gt_phone_scores):
        if idx >= len(gt_times) or gt_times[idx] is None:
            continue
        entry = dict(gt_times[idx])
        entry["target_accuracy"] = float(target["accuracy"])
        entry["gt_phone_index"] = int(idx)
        entry["gt_phone"] = target.get("phone")
        entry["gt_word_id"] = target.get("word_id")
        gt_entries.append(entry)

    pred_times = fit_list(pred_times, len(phone_scores), None)
    for idx, item in enumerate(phone_scores):
        pred_time = pred_times[idx]
        if pred_time is not None:
            item["start"] = pred_time["start"]
            item["end"] = pred_time["end"]
            item.setdefault("phone", pred_time.get("phone"))

        overlaps = []
        if pred_time is not None:
            for gt in gt_entries:
                amount = overlap_amount(pred_time, gt)
                if amount >= min_overlap_sec:
                    overlaps.append((gt, amount))

        if overlaps:
            denom = sum(weight for _, weight in overlaps)
            item["target_accuracy"] = float(
                sum(gt["target_accuracy"] * weight for gt, weight in overlaps) / denom
            )
            item["target_valid"] = True
            item["target_source"] = "charsiu_phone_time_overlap"
            item["target_gt_indices"] = [int(gt["gt_phone_index"]) for gt, _ in overlaps]
            item["target_overlap_sec"] = float(denom)
        elif no_overlap_policy == "default":
            item["target_accuracy"] = PHONE_NO_OVERLAP_TARGET
            item["target_valid"] = True
            item["target_source"] = "no_overlap_default"
            item["target_gt_indices"] = []
            item["target_overlap_sec"] = 0.0
        else:
            item["target_valid"] = False
            item["target_source"] = "no_overlap"
            item["target_gt_indices"] = []
            item["target_overlap_sec"] = 0.0
        phone_scores[idx] = item
    row["phone_scores"] = phone_scores
    row["phone_target_status"] = "attached"
    return row


def completed_key(row):
    return (str(row.get("utt_id")), int(row.get("chunk_id", -1)), resolve_eval_id(row))


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
    cache_dir = args.cache_dir or (args.output_jsonl.parent / "charsiu_phone_word_time_cache")
    completed = load_completed(args.output_jsonl) if args.resume else {}
    aligner, get_match_index = load_charsiu(args.multipa_repo_root, args.aligner, args.align_device)

    output_rows = []
    counts = {"rows": 0, "ok": 0, "reused": 0, "attached": 0, "errors": 0}
    for row_idx, row in enumerate(rows, start=1):
        counts["rows"] += 1
        key = completed_key(row)
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
            gt_text, gt_word_scores = gt_text_and_scores(scores, source_utt_id)
            gt_phone_scores = flat_gt_phone_scores(scores, source_utt_id)
            gt_key = cache_key("gt", source_utt_id, full_wav, gt_text)
            gt_aligned = align_phone_word_times(
                aligner,
                get_match_index,
                full_wav,
                gt_text,
                cache_dir / "gt" / f"{source_utt_id}_{gt_key}.json",
            )

            pred_text = prediction_text(row)
            pred_key = cache_key("pred", resolve_eval_id(row), prefix_wav, pred_text)
            pred_aligned = align_phone_word_times(
                aligner,
                get_match_index,
                prefix_wav,
                pred_text,
                cache_dir / "pred" / f"{resolve_eval_id(row)}_{pred_key}.json",
            )

            pred_word_times = existing_pred_times(row) or pred_aligned["words"]
            row = attach_targets_to_words(
                row,
                pred_word_times,
                gt_aligned["words"],
                gt_word_scores,
                args.no_overlap_policy,
                args.min_overlap_sec,
            )
            row = attach_targets_to_phones(
                row,
                pred_aligned["phones"],
                gt_aligned["phones"],
                gt_phone_scores,
                args.no_overlap_policy,
                args.min_overlap_sec,
            )
            row["target_alignment"] = {
                "gt_time_source": "charsiu_forced_alignment_full_audio_gt_text",
                "pred_time_source": "charsiu_forced_alignment_prefix_audio_pred_text",
                "phone_target_source": "speechocean_phones_accuracy_time_overlap",
                "word_target_source": "speechocean_word_scores_time_overlap",
                "no_overlap_policy": args.no_overlap_policy,
            }
            counts["attached"] += 1
            counts["ok"] += 1
        except Exception as exc:
            row["target_status"] = "error"
            row["target_error"] = repr(exc)
            counts["errors"] += 1
        output_rows.append(row)
        if row_idx % 100 == 0 or row_idx == len(rows):
            print(f"[attach-phone-word-targets] {row_idx}/{len(rows)} {counts}", flush=True)

    write_jsonl(args.output_jsonl, output_rows)
    print(json.dumps({"output_jsonl": str(args.output_jsonl), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
