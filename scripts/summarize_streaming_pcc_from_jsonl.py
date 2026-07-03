import argparse
import json
from pathlib import Path

import numpy as np


UTT_KEYS = ["accuracy", "completeness", "fluency", "prosodic", "total"]
WORD_KEYS = ["accuracy", "stress", "total"]


def pcc(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size < 2 or target.size < 2:
        return None
    if np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return None
    value = float(np.corrcoef(pred, target)[0, 1])
    return value if np.isfinite(value) else None


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_rows(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("scores")]
    utterance = {}
    for key in UTT_KEYS:
        pred = []
        target = []
        for row in ok_rows:
            if key not in row.get("scores", {}) or key not in row.get("target_scores", {}):
                continue
            pred.append(row["scores"][key])
            target.append(row["target_scores"][key])
        utterance[key] = pcc(pred, target)

    word = {}
    word_n = 0
    for key in WORD_KEYS:
        pred = []
        target = []
        for row in ok_rows:
            for item in row.get("word_scores", []):
                pred_key = f"pred_{key}"
                target_key = f"target_{key}"
                if pred_key not in item or target_key not in item or item[target_key] is None:
                    continue
                pred.append(item[pred_key])
                target.append(item[target_key])
        word[key] = pcc(pred, target)
        if key == "accuracy":
            word_n = len(pred)

    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "utterances": len({str(row.get("utt_id")) for row in ok_rows}),
        "utterance_pcc": utterance,
        "word_pcc": word,
        "word_eval_count": word_n,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize all-chunk and final/full PCC from JSONL outputs.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="MODEL_NAME=PATH_TO_JSONL. May be passed multiple times.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = {"all_streaming_chunks": {}, "final_full_utterance": {}}
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"--input must be MODEL=PATH, got {item!r}")
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        rows = load_jsonl(path)
        payload["all_streaming_chunks"][name] = summarize_rows(rows)
        payload["final_full_utterance"][name] = summarize_rows([row for row in rows if row.get("is_final")])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
