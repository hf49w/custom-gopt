import argparse
import ast
import json
from pathlib import Path


SCORE_NAMES = ["accuracy", "fluency", "prosodic", "total"]
WORD_NAMES = ["accuracy", "stress", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MultiPA test_open.py semicolon output into prefix-eval JSONL."
    )
    parser.add_argument("--prefix-audio-manifest", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def parse_score_list(text):
    text = text.strip()
    if not text:
        return []
    if "," not in text:
        return [float(text)]
    return [float(item) for item in text.split(",") if item.strip()]


def parse_prediction_line(line):
    parts = [part.strip() for part in line.strip().split(";")]
    if not parts or not parts[0]:
        return None
    eval_id = parts[0].replace(".wav", "")
    fields = {"eval_id": eval_id}
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_alignment(value):
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    rows = []
    for item in parsed:
        if len(item) < 3:
            continue
        try:
            rows.append(
                {
                    "start": float(item[0]),
                    "end": float(item[1]),
                    "word": str(item[2]).lower(),
                }
            )
        except Exception:
            continue
    return rows


def load_manifest(path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["utt_id"])] = row
    return rows


def main():
    args = parse_args()
    manifest = load_manifest(args.prefix_audio_manifest)
    outputs = []
    seen = set()

    with args.prediction_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = parse_prediction_line(line)
            if parsed is None:
                continue
            eval_id = parsed["eval_id"]
            base = manifest.get(eval_id, {"utt_id": eval_id})
            record = dict(base)
            record["eval_id"] = eval_id
            record["utt_id"] = str(base.get("source_utt_id", base.get("utt_id", eval_id)))
            record["model"] = "multipa"
            record["mode"] = "audio_prefix_recompute"
            record["status"] = "ok" if parsed.get("Valid") == "T" else "invalid"
            record["asr"] = {
                "sentence_model_text": parsed.get("ASR_s", ""),
                "word_model_text": parsed.get("ASR_w", ""),
            }

            if record["status"] == "ok":
                record["scores"] = {
                    "accuracy": float(parsed.get("A", 0.0)),
                    "fluency": float(parsed.get("F", 0.0)),
                    "prosodic": float(parsed.get("P", 0.0)),
                    "total": float(parsed.get("T", 0.0)),
                }
                alignment = parse_alignment(parsed.get("alignment", ""))
                word_values = {
                    "accuracy": parse_score_list(parsed.get("w_a", "")),
                    "stress": parse_score_list(parsed.get("w_s", "")),
                    "total": parse_score_list(parsed.get("w_t", "")),
                }
                max_len = max([len(alignment)] + [len(values) for values in word_values.values()])
                word_scores = []
                for idx in range(max_len):
                    item = {"word_id": idx}
                    if idx < len(alignment):
                        item.update(alignment[idx])
                    for name in WORD_NAMES:
                        values = word_values[name]
                        if idx < len(values):
                            item[f"pred_{name}"] = float(values[idx])
                    word_scores.append(item)
                record["word_scores"] = word_scores
            outputs.append(record)
            seen.add(eval_id)

    for eval_id, base in manifest.items():
        if eval_id in seen:
            continue
        record = dict(base)
        record["eval_id"] = eval_id
        record["utt_id"] = str(base.get("source_utt_id", base.get("utt_id", eval_id)))
        record["model"] = "multipa"
        record["mode"] = "audio_prefix_recompute"
        record["status"] = "missing_prediction"
        outputs.append(record)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input_predictions": str(args.prediction_path),
                "prefix_audio_manifest": str(args.prefix_audio_manifest),
                "output_jsonl": str(args.output_jsonl),
                "rows": len(outputs),
                "parsed_predictions": len(seen),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
