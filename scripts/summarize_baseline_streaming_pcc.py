import argparse
import json
from pathlib import Path

import numpy as np


UTT_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]
WORD_NAMES = ["accuracy", "stress", "total"]


def pcc(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.size < 2 or target.size < 2:
        return None
    if np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return None
    value = float(np.corrcoef(pred, target)[0, 1])
    return value if np.isfinite(value) else None


def load_json(path):
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start > 0:
        text = text[start:]
    return json.loads(text)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prefix_utt_metrics(path, final_only=False):
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("status") == "ok" and row.get("scores") and row.get("target_scores")
    ]
    if final_only:
        rows = [row for row in rows if row.get("is_final")]
    return {
        name: pcc(
            [row["scores"].get(name) for row in rows],
            [row["target_scores"].get(name) for row in rows],
        )
        for name in UTT_NAMES
    } | {"n": len(rows)}


def prefix_word_metrics(path, final_only=False):
    pred = {name: [] for name in WORD_NAMES}
    target = {name: [] for name in WORD_NAMES}
    rows = [row for row in load_jsonl(path) if row.get("status") == "ok"]
    if final_only:
        rows = [row for row in rows if row.get("is_final")]
    for row in rows:
        for word in row.get("word_scores", []):
            if not all(f"target_{name}" in word and f"pred_{name}" in word for name in WORD_NAMES):
                continue
            for name in WORD_NAMES:
                pred[name].append(word[f"pred_{name}"])
                target[name].append(word[f"target_{name}"])
    return {name: pcc(pred[name], target[name]) for name in WORD_NAMES} | {"n": len(pred["accuracy"])}


def closed_full_metrics(path):
    obj = load_json(path)
    metrics = obj["metrics"] if "metrics" in obj else obj
    return {
        "phone": {"accuracy": metrics.get("phone_pcc")},
        "word": {
            "accuracy": metrics["word_pcc"][0],
            "stress": metrics["word_pcc"][1],
            "total": metrics["word_pcc"][2],
        },
        "utterance": {
            "accuracy": metrics["utt_pcc"][0],
            "completeness": metrics["utt_pcc"][1],
            "fluency": metrics["utt_pcc"][2],
            "prosodic": metrics["utt_pcc"][3],
            "total": metrics["utt_pcc"][4],
        },
        "n": obj.get("num_utterances"),
    }


def open_full_metrics(path):
    obj = load_json(path)
    metrics = obj["metrics"]
    return {
        "word": metrics.get("word_pcc", {}),
        "utterance": metrics.get("utterance_pcc", {}),
        "n": metrics.get("utterance_count"),
        "word_n": metrics.get("word_eval_word_count"),
    }


def pcn_metrics(path):
    obj = load_json(path)
    def block(section):
        row = obj[section]
        return {
            "phone": row["phone"]["pcc"],
            "word": row["word"]["pcc"],
            "utterance": row["utterance"]["pcc"],
            "n": row["utterance"]["n"],
            "word_n": row["word"]["n"],
            "phone_n": row["phone"]["n"],
        }
    return {
        "all_streaming_chunks": block("all_streaming_chunks"),
        "final_full_utterance": block("final_chunks_full_utterance"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("downloads/custom-gopt-252/eval"))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    prefix_root = args.root / "prefix_streaming"
    result = {
        "notes": [
            "all_streaming_chunks for baselines uses prefix_streaming JSONL rows.",
            "GOPT-open and MultiPA prefix JSONL files do not contain GT-aligned word targets, so all-chunk word PCC is not reported for them.",
            "final_full_utterance for GOPT-open/MultiPA uses the full-utterance open-eval summaries with word alignment.",
        ],
        "all_streaming_chunks": {
            "gopt_closed_oracle_prefix": {
                "utterance": prefix_utt_metrics(prefix_root / "original_gopt.jsonl"),
                "word": prefix_word_metrics(prefix_root / "original_gopt.jsonl"),
            },
            "gopt_open_whisper_base_prefix": {
                "utterance": prefix_utt_metrics(prefix_root / "gopt_whisper_base" / "predictions.jsonl"),
                "word": None,
            },
            "multipa_prefix": {
                "utterance": prefix_utt_metrics(prefix_root / "multipa.jsonl"),
                "word": None,
            },
        },
        "final_full_utterance": {
            "gopt_closed": closed_full_metrics(args.root / "gopt_closed_test_subset" / "gopt_closed_metrics.json"),
            "gopt_open_whisper_base": open_full_metrics(args.root / "gopt_open_base" / "summary.json"),
            "multipa": open_full_metrics(args.root / "multipa_open_streaming_test" / "summary.json"),
        },
    }

    pcn_path = args.root / "streaming_pcn_gopt_v2_stateful_teacher_state" / "pcn_streaming_per_metric_pcc.json"
    if pcn_path.exists():
        pcn = pcn_metrics(pcn_path)
        result["all_streaming_chunks"]["pcn_teacher_state"] = pcn["all_streaming_chunks"]
        result["final_full_utterance"]["pcn_teacher_state"] = pcn["final_full_utterance"]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
