import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SENTENCE_DIMS = [
    ("accuracy", "sent_acc"),
    ("completeness", "sent_comp"),
    ("fluency", "sent_flu"),
    ("prosodic", "sent_pros"),
    ("total", "sent_total"),
]
WORD_DIMS = [
    ("accuracy", "word_acc"),
    ("stress", "word_stress"),
    ("total", "word_total"),
]
COVERAGES = [1.0, 0.9, 0.8, 0.7]


def pcc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    out = float(np.corrcoef(x, y)[0, 1])
    return out if np.isfinite(out) else None


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def mean_numeric(values):
    vals = [as_float(item) for item in values or []]
    vals = [item for item in vals if item is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def row_confidence(row):
    candidates = [
        ("mean_asr_confidence", as_float(row.get("mean_asr_confidence"))),
        ("mean_word_confidence", mean_numeric(row.get("word_confidences"))),
        ("mean_token_confidence", mean_numeric(row.get("token_confidences"))),
        ("matched_ratio", as_float(row.get("matched_ratio"))),
        ("progress", as_float(row.get("progress"))),
        ("coverage_ratio", as_float(row.get("coverage_ratio"))),
    ]
    for name, value in candidates:
        if value is not None:
            return value, name
    return 1.0, "constant"


def item_confidence(row, item, level):
    for key in ["confidence", "word_confidence", "phone_confidence", "asr_confidence"]:
        value = as_float(item.get(key))
        if value is not None:
            return value, key
    if level == "word":
        word_id = item.get("word_id")
        try:
            word_id = int(word_id)
        except (TypeError, ValueError):
            word_id = None
        word_conf = row.get("word_confidences") or []
        if word_id is not None and 0 <= word_id < len(word_conf):
            value = as_float(word_conf[word_id])
            if value is not None:
                return value, "row_word_confidences"
    if level == "phone":
        phone_index = item.get("phone_index")
        try:
            phone_index = int(phone_index)
        except (TypeError, ValueError):
            phone_index = None
        token_conf = row.get("token_confidences") or []
        if phone_index is not None and 0 <= phone_index < len(token_conf):
            value = as_float(token_conf[phone_index])
            if value is not None:
                return value, "row_token_confidences"
    return row_confidence(row)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def add_point(bucket, pred, target, conf, source):
    pred = as_float(pred)
    target = as_float(target)
    conf = as_float(conf)
    if pred is None or target is None:
        return
    bucket["pred"].append(pred)
    bucket["target"].append(target)
    bucket["conf"].append(1.0 if conf is None else conf)
    bucket["sources"][source] += 1


def collect_jsonl(path):
    data = {
        "sentence": {out_name: {"pred": [], "target": [], "conf": [], "sources": Counter()} for _, out_name in SENTENCE_DIMS},
        "word": {out_name: {"pred": [], "target": [], "conf": [], "sources": Counter()} for _, out_name in WORD_DIMS},
        "phone": {"phone": {"pred": [], "target": [], "conf": [], "sources": Counter()}},
    }
    row_count = 0
    ok_count = 0
    final_count = 0
    for row in load_jsonl(path):
        row_count += 1
        if row.get("status", "ok") != "ok":
            continue
        ok_count += 1
        row_conf, row_conf_source = row_confidence(row)
        if bool(row.get("is_final")):
            final_count += 1
            scores = row.get("scores") or {}
            targets = row.get("target_scores") or {}
            for in_name, out_name in SENTENCE_DIMS:
                if in_name not in scores or in_name not in targets:
                    continue
                add_point(data["sentence"][out_name], scores.get(in_name), targets.get(in_name), row_conf, row_conf_source)
        for word in row.get("word_scores") or []:
            if word.get("target_valid", True) is False:
                continue
            conf, source = item_confidence(row, word, "word")
            for in_name, out_name in WORD_DIMS:
                add_point(
                    data["word"][out_name],
                    word.get(f"pred_{in_name}"),
                    word.get(f"target_{in_name}"),
                    conf,
                    source,
                )
        for phone in row.get("phone_scores") or []:
            if phone.get("target_valid", True) is False:
                continue
            conf, source = item_confidence(row, phone, "phone")
            add_point(data["phone"]["phone"], phone.get("pred_accuracy"), phone.get("target_accuracy"), conf, source)
    return data, {"rows": row_count, "ok_rows": ok_count, "final_rows": final_count}


def coverage_rows(model, data):
    rows = []
    for level, metrics in data.items():
        for metric, values in metrics.items():
            pred = np.asarray(values["pred"], dtype=np.float64)
            target = np.asarray(values["target"], dtype=np.float64)
            conf = np.asarray(values["conf"], dtype=np.float64)
            source = ",".join(f"{k}:{v}" for k, v in values["sources"].most_common())
            for coverage in COVERAGES:
                keep = int(math.ceil(pred.size * coverage)) if pred.size else 0
                if pred.size:
                    keep = max(1, keep)
                    order = np.argsort(-conf, kind="mergesort")[:keep]
                    value = pcc(pred[order], target[order])
                else:
                    value = None
                rows.append(
                    {
                        "model": model,
                        "level": level,
                        "metric": metric,
                        "coverage": int(coverage * 100),
                        "count": keep,
                        "pcc": "" if value is None else value,
                        "confidence_source": source,
                    }
                )
    return rows


def pivot_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["coverage"])
        grouped[key][row["metric"]] = row["pcc"]
        grouped[key][f"{row['metric']}_n"] = row["count"]
    out = []
    columns = ["sent_acc", "sent_comp", "sent_flu", "sent_pros", "sent_total", "word_acc", "word_stress", "word_total", "phone"]
    for (model, coverage), vals in sorted(grouped.items()):
        row = {"model": model, "coverage": coverage}
        for col in columns:
            row[col] = vals.get(col, "")
        for col in columns:
            row[f"{col}_n"] = vals.get(f"{col}_n", "")
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=/path/to/jsonl")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-pivot-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    meta = {}
    for item in args.model:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        data, cur_meta = collect_jsonl(path)
        meta[name] = {"path": str(path), **cur_meta}
        rows.extend(coverage_rows(name, data))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "level", "metric", "coverage", "count", "pcc", "confidence_source"])
        writer.writeheader()
        writer.writerows(rows)

    pivot = pivot_rows(rows)
    pivot_fields = ["model", "coverage", "sent_acc", "sent_comp", "sent_flu", "sent_pros", "sent_total", "word_acc", "word_stress", "word_total", "phone"]
    pivot_fields += [f"{name}_n" for name in pivot_fields[2:]]
    with args.output_pivot_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pivot_fields)
        writer.writeheader()
        writer.writerows(pivot)

    args.output_json.write_text(json.dumps({"meta": meta, "rows": rows, "pivot": pivot}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"meta": meta, "pivot": pivot}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
