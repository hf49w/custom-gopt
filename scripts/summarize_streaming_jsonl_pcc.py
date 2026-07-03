import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


UTT_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]
WORD_NAMES = ["accuracy", "stress", "total"]
PHONE_NAMES = ["accuracy"]


def parse_model_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected MODEL_NAME=/path/to/predictions.jsonl")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Empty model name")
    return name, Path(path)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pcc(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.size < 2 or target.size < 2:
        return None
    if np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return None
    value = float(np.corrcoef(pred, target)[0, 1])
    return value if np.isfinite(value) else None


def metric(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return {"n": 0, "pcc": None, "mae": None}
    return {
        "n": int(pred.size),
        "pcc": pcc(pred, target),
        "mae": float(np.mean(np.abs(pred - target))),
    }


def row_ok(row):
    return row.get("status", "ok") == "ok" and isinstance(row.get("scores"), dict)


def utterance_metrics(rows):
    out = {}
    for name in UTT_NAMES:
        pred = []
        target = []
        for row in rows:
            if not row_ok(row):
                continue
            scores = row.get("scores") or {}
            targets = row.get("target_scores") or {}
            if name not in scores or name not in targets:
                continue
            pred.append(scores[name])
            target.append(targets[name])
        out[name] = metric(pred, target)
    return out


def word_metrics(rows):
    pred = {name: [] for name in WORD_NAMES}
    target = {name: [] for name in WORD_NAMES}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        for word in row.get("word_scores") or []:
            if word.get("target_valid", True) is False:
                continue
            for name in WORD_NAMES:
                pred_key = f"pred_{name}"
                target_key = f"target_{name}"
                if pred_key not in word or target_key not in word:
                    continue
                pred[name].append(word[pred_key])
                target[name].append(word[target_key])
    return {name: metric(pred[name], target[name]) for name in WORD_NAMES}


def phone_metrics(rows):
    pred = {name: [] for name in PHONE_NAMES}
    target = {name: [] for name in PHONE_NAMES}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        for phone in row.get("phone_scores") or []:
            if phone.get("target_valid", True) is False:
                continue
            for name in PHONE_NAMES:
                pred_key = f"pred_{name}"
                target_key = f"target_{name}"
                if pred_key not in phone or target_key not in phone:
                    continue
                pred[name].append(phone[pred_key])
                target[name].append(phone[target_key])
    return {name: metric(pred[name], target[name]) for name in PHONE_NAMES}


def summarize_rows(rows, final_only=False):
    selected = [row for row in rows if (not final_only or bool(row.get("is_final")))]
    status = Counter(row.get("status", "ok") for row in selected)
    utterance_ids = {str(row.get("utt_id")) for row in selected}
    return {
        "rows": len(selected),
        "utterances": len(utterance_ids),
        "status_counts": dict(status),
        "phone": phone_metrics(selected),
        "utterance": utterance_metrics(selected),
        "word": word_metrics(selected),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize all-chunk and final/full PCC from streaming prefix JSONL predictions."
    )
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    output = {
        "protocol": {
            "all_streaming_chunks": "all rows in each JSONL with status=ok for metric-specific calculations",
            "final_full_utterance": "rows where is_final=true from the same JSONL",
            "phone_metrics": "phone_scores entries with pred_* and target_* fields; target_valid=false entries are skipped",
            "word_metrics": "word_scores entries with pred_* and target_* fields; target_valid=false entries are skipped",
        },
        "models": {},
    }
    for name, path in args.model:
        rows = load_jsonl(path)
        output["models"][name] = {
            "path": str(path),
            "all_streaming_chunks": summarize_rows(rows, final_only=False),
            "final_full_utterance": summarize_rows(rows, final_only=True),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
