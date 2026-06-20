import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_BINS = [0.25, 0.5, 0.75, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize common prefix-streaming JSONL outputs.")
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bins", type=float, nargs="+", default=DEFAULT_BINS)
    parser.add_argument("--convergence-epsilon", type=float, default=0.5)
    return parser.parse_args()


def safe_pcc(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(pred) < 2 or np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return None
    return float(np.corrcoef(pred, target)[0, 1])


def score_metrics(rows, score_name):
    pred = [row["scores"][score_name] for row in rows]
    target = [row["target_scores"][score_name] for row in rows]
    errors = np.asarray(pred) - np.asarray(target)
    return {
        "count": len(rows),
        "pcc": safe_pcc(pred, target),
        "mse": float(np.mean(errors ** 2)) if len(rows) else None,
        "mae": float(np.mean(np.abs(errors))) if len(rows) else None,
    }


def select_bin_rows(grouped, progress_bin):
    selected = []
    for rows in grouped.values():
        candidates = [row for row in rows if float(row["progress"]) <= progress_bin + 1e-9]
        if candidates:
            selected.append(max(candidates, key=lambda row: float(row["progress"])))
    return selected


def convergence_progress(rows, score_name, epsilon):
    final_score = rows[-1]["scores"][score_name]
    for index, row in enumerate(rows):
        if all(
            abs(later["scores"][score_name] - final_score) <= epsilon
            for later in rows[index:]
        ):
            return float(row["progress"])
    return 1.0


def summarize(path, bins, epsilon):
    all_rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_counts = defaultdict(int)
    for row in all_rows:
        status_counts[row.get("status", "unknown")] += 1
    rows = [row for row in all_rows if row.get("scores") and row.get("status") == "ok"]
    all_utterance_ids = set(row["utt_id"] for row in all_rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["utt_id"]].append(row)
    for utt_rows in grouped.values():
        utt_rows.sort(key=lambda row: (float(row["progress"]), int(row["chunk_id"])))

    score_names = sorted(
        set.intersection(*(set(row["scores"]) for row in rows))
    ) if rows else []
    bin_metrics = {}
    for progress_bin in bins:
        selected = select_bin_rows(grouped, progress_bin)
        bin_metrics[str(progress_bin)] = {
            "coverage": len(selected) / max(len(all_utterance_ids), 1),
            "scores": {
                score_name: score_metrics(selected, score_name)
                for score_name in score_names
            },
        }

    auc_pcc = {}
    for score_name in score_names:
        points = [
            (
                float(progress_bin),
                bin_metrics[str(progress_bin)]["scores"][score_name]["pcc"],
            )
            for progress_bin in bins
        ]
        points = [(x, y) for x, y in points if y is not None]
        if len(points) >= 2 and points[-1][0] > points[0][0]:
            xs = np.asarray([point[0] for point in points], dtype=np.float64)
            ys = np.asarray([point[1] for point in points], dtype=np.float64)
            auc_pcc[score_name] = float(np.trapz(ys, xs) / (xs[-1] - xs[0]))
        else:
            auc_pcc[score_name] = None

    stability = {}
    convergence = {}
    for score_name in score_names:
        deltas = []
        convergence_values = []
        for utt_rows in grouped.values():
            values = [row["scores"][score_name] for row in utt_rows]
            deltas.extend(abs(b - a) for a, b in zip(values, values[1:]))
            convergence_values.append(
                convergence_progress(utt_rows, score_name, epsilon)
            )
        stability[score_name] = {
            "mean_abs_adjacent_delta": float(np.mean(deltas)) if deltas else 0.0,
            "p90_abs_adjacent_delta": float(np.percentile(deltas, 90)) if deltas else 0.0,
        }
        convergence[score_name] = {
            "median_progress": float(np.median(convergence_values)),
            "p90_progress": float(np.percentile(convergence_values, 90)),
        }

    process_times = [float(row.get("process_time_sec", 0.0)) for row in rows]
    return {
        "input_jsonl": str(path),
        "model": rows[0].get("model") if rows else None,
        "mode": rows[0].get("mode") if rows else None,
        "timing_modes": sorted(
            set(row.get("timing_mode", "unspecified") for row in all_rows)
        ),
        "total_records": len(all_rows),
        "status_counts": dict(status_counts),
        "rows": len(rows),
        "utterances": len(grouped),
        "source_utterances": len(all_utterance_ids),
        "valid_utterance_coverage": len(grouped) / max(len(all_utterance_ids), 1),
        "score_names": score_names,
        "bins": bin_metrics,
        "normalized_auc_pcc_over_bins": auc_pcc,
        "stability": stability,
        "convergence_epsilon": epsilon,
        "convergence": convergence,
        "latency_sec": {
            "median": float(np.median(process_times)) if process_times else None,
            "p95": float(np.percentile(process_times, 95)) if process_times else None,
        },
    }


def main():
    args = parse_args()
    payload = {
        "protocol": {
            "bin_selection": "latest available prefix not exceeding each progress bin",
            "bins": args.bins,
            "convergence_epsilon": args.convergence_epsilon,
        },
        "models": [
            summarize(path, args.bins, args.convergence_epsilon)
            for path in args.input_jsonl
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
