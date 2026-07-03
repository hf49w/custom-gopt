import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


WORD_SCORE_KEYS = ["accuracy", "stress", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach GT word scores to prefix JSONL word predictions via Charsiu word-time overlap."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--source-prefix-manifest", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--multipa-repo-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--aligner", default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_words(text: str):
    return [word.lower() for word in re.findall(r"[A-Za-z']+", text or "")]


def load_charsiu(multipa_repo_root: Path, aligner_name: str, device: Optional[str]):
    root = str(multipa_repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in ["Charsiu", "utils_assessment"]:
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file and root not in str(Path(module_file).resolve()):
            sys.modules.pop(name, None)
    from Charsiu import charsiu_forced_aligner
    from utils_assessment import get_match_index

    kwargs = {"aligner": aligner_name}
    if device:
        kwargs["device"] = device
    return charsiu_forced_aligner(**kwargs), get_match_index


def as_word_rows(value):
    rows = []
    arr = np.asarray(value)
    for item in arr.tolist():
        if len(item) < 3:
            continue
        try:
            rows.append({"start": float(item[0]), "end": float(item[1]), "word": str(item[2]).lower()})
        except Exception:
            continue
    return rows


def align_text(aligner, get_match_index, wav_path: Path, text: str):
    if not text.strip():
        return []
    pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(
        audio=str(wav_path),
        text=text,
    )
    try:
        selected = get_match_index(pred_words, words)
        pred_words = np.asarray(pred_words)[selected]
    except Exception:
        pass
    return as_word_rows(pred_words)


def cached_alignment(cache_path: Path, aligner, get_match_index, wav_path: Path, text: str):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = align_text(aligner, get_match_index, wav_path, text)
        payload = {"status": "ok", "rows": rows}
    except Exception as exc:
        payload = {"status": "error", "error": repr(exc), "rows": []}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def source_key(row):
    return f"{row.get('utt_id')}::c{int(row.get('chunk_id', -1)):04d}"


def build_source_maps(source_rows):
    by_key = {}
    by_eval_id = {}
    for row in source_rows:
        key = source_key(row)
        by_key[key] = row
        by_eval_id[f"{row['utt_id']}_c{int(row['chunk_id']):04d}"] = row
    return by_key, by_eval_id


def row_source(row, by_key, by_eval_id):
    utt_id = str(row.get("source_utt_id", row.get("utt_id", "")))
    chunk_id = int(row.get("chunk_id", -1))
    key = f"{utt_id}::c{chunk_id:04d}"
    if key in by_key:
        return by_key[key]
    eval_id = str(row.get("eval_id", ""))
    return by_eval_id.get(eval_id)


def get_hypothesis_text(row):
    if row.get("asr_text"):
        return row["asr_text"]
    if isinstance(row.get("asr"), dict):
        return row["asr"].get("word_model_text") or row["asr"].get("sentence_model_text") or ""
    words = [item.get("word") for item in row.get("word_scores", []) if item.get("word")]
    return " ".join(words)


def existing_word_times(row):
    result = []
    for item in row.get("word_scores", []):
        if "start" not in item or "end" not in item:
            return []
        try:
            result.append(
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "word": str(item.get("word", "")).lower(),
                }
            )
        except Exception:
            return []
    return result


def weighted_gt_score(pred_span, gt_rows, gt_words):
    weights = []
    selected = []
    start = float(pred_span["start"])
    end = float(pred_span["end"])
    for idx, gt in enumerate(gt_rows):
        overlap = max(0.0, min(end, float(gt["end"])) - max(start, float(gt["start"])))
        if overlap <= 0:
            continue
        if idx >= len(gt_words):
            continue
        weights.append(overlap)
        selected.append((idx, gt_words[idx]))
    if not selected:
        return None
    total = sum(weights)
    out = {"gt_word_indices": [idx for idx, _ in selected], "overlap_sec": total}
    for key in WORD_SCORE_KEYS:
        out[f"target_{key}"] = float(
            sum(float(word.get(key, 0.0)) * weight for weight, (_, word) in zip(weights, selected)) / total
        )
    out["target_words"] = [str(word.get("text", "")).lower() for _, word in selected]
    return out


def main():
    args = parse_args()
    input_rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        input_rows = input_rows[: args.limit]
    source_rows = load_jsonl(args.source_prefix_manifest)
    source_by_key, source_by_eval_id = build_source_maps(source_rows)
    scores = json.loads(args.scores_json.read_text(encoding="utf-8"))

    aligner, get_match_index = load_charsiu(args.multipa_repo_root, args.aligner, args.device)
    outputs = []
    stats = {
        "rows": 0,
        "rows_ok": 0,
        "word_predictions": 0,
        "word_targets_attached": 0,
        "gt_alignment_errors": 0,
        "hyp_alignment_errors": 0,
    }

    for index, row in enumerate(input_rows, start=1):
        out = dict(row)
        stats["rows"] += 1
        if row.get("status") != "ok" or not row.get("word_scores"):
            outputs.append(out)
            continue

        source = row_source(row, source_by_key, source_by_eval_id)
        if source is None:
            out["word_target_alignment_status"] = "missing_source_manifest_row"
            outputs.append(out)
            continue

        utt_id = str(source["utt_id"])
        score_item = scores.get(utt_id)
        if not score_item:
            out["word_target_alignment_status"] = "missing_scores"
            outputs.append(out)
            continue

        gt_text = score_item.get("text") or source.get("reference_text") or ""
        gt_cache = args.cache_dir / "gt" / f"{utt_id}.json"
        gt_payload = cached_alignment(
            gt_cache,
            aligner,
            get_match_index,
            Path(source["wav_path"]),
            gt_text,
        )
        gt_rows = gt_payload.get("rows", [])
        if gt_payload.get("status") != "ok" or not gt_rows:
            stats["gt_alignment_errors"] += 1
            out["word_target_alignment_status"] = "gt_alignment_failed"
            outputs.append(out)
            continue

        hyp_rows = existing_word_times(row)
        if not hyp_rows:
            hyp_text = get_hypothesis_text(row)
            eval_id = str(row.get("eval_id") or f"{utt_id}_c{int(row.get('chunk_id', -1)):04d}")
            hyp_cache = args.cache_dir / "hyp" / f"{eval_id}.json"
            hyp_payload = cached_alignment(
                hyp_cache,
                aligner,
                get_match_index,
                Path(row.get("wav_path", "")),
                hyp_text,
            )
            hyp_rows = hyp_payload.get("rows", [])
            if hyp_payload.get("status") != "ok" or not hyp_rows:
                stats["hyp_alignment_errors"] += 1
                out["word_target_alignment_status"] = "hyp_alignment_failed"
                outputs.append(out)
                continue

        word_scores = [dict(item) for item in row.get("word_scores", [])]
        hyp_words = normalize_words(get_hypothesis_text(row))
        attached = 0
        for word_index, word_score in enumerate(word_scores):
            stats["word_predictions"] += 1
            if word_index < len(hyp_rows):
                word_score.setdefault("start", hyp_rows[word_index]["start"])
                word_score.setdefault("end", hyp_rows[word_index]["end"])
                if hyp_rows[word_index].get("word"):
                    word_score.setdefault("word", hyp_rows[word_index]["word"])
            elif word_index < len(hyp_words):
                word_score.setdefault("word", hyp_words[word_index])

            if word_index >= len(hyp_rows):
                continue
            target = weighted_gt_score(hyp_rows[word_index], gt_rows, score_item.get("words", []))
            if not target:
                continue
            word_score.update(target)
            attached += 1

        stats["word_targets_attached"] += attached
        stats["rows_ok"] += 1
        out["word_scores"] = word_scores
        out["word_target_alignment_status"] = "ok"
        out["word_target_alignment"] = {
            "gt_word_count": len(gt_rows),
            "hyp_word_count": len(hyp_rows),
            "attached_word_targets": attached,
        }
        outputs.append(out)
        if index % 100 == 0:
            print(f"[align-prefix-word-targets] {index}/{len(input_rows)}", flush=True)

    write_jsonl(args.output_jsonl, outputs)
    print(
        json.dumps(
            {
                "input_jsonl": str(args.input_jsonl),
                "output_jsonl": str(args.output_jsonl),
                "cache_dir": str(args.cache_dir),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
