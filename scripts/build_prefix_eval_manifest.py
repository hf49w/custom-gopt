import argparse
import json
import wave
from collections import defaultdict
from pathlib import Path


SCORE_KEYS = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one shared prefix manifest for StreamingGOPT, GOPT, and MultiPA."
    )
    parser.add_argument("--streaming-data-root", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def load_wav_map(dataset_root):
    wav_map = {}
    dataset_root = dataset_root.absolute()
    for partition in ["train", "test"]:
        scp_path = dataset_root / partition / "wav.scp"
        if not scp_path.exists():
            continue
        for line in scp_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, wav_path = line.split(maxsplit=1)
            resolved = Path(wav_path)
            if not resolved.is_absolute():
                resolved = dataset_root / resolved
            wav_map[str(utt_id)] = str(resolved)
    return wav_map


def mean_numeric(values, default=0.0):
    if not isinstance(values, list) or not values:
        return default
    numeric = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
    if not numeric:
        return default
    return sum(numeric) / len(numeric)


def main():
    args = parse_args()
    scores = json.loads(args.scores_json.read_text(encoding="utf-8"))
    wav_map = load_wav_map(args.dataset_root)
    manifest_path = args.streaming_data_root / f"{args.split}_manifest.jsonl"

    source_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_audio_end = {}
    chunk_counts = defaultdict(int)
    for row in source_rows:
        utt_id = str(row["utt_id"])
        if row.get("is_final", False):
            final_audio_end[utt_id] = float(row["audio_end"])
        chunk_counts[utt_id] += 1

    durations = {}
    for utt_id in chunk_counts:
        if utt_id in final_audio_end:
            durations[utt_id] = final_audio_end[utt_id]
            continue
        with wave.open(str(wav_map[utt_id]), "rb") as handle:
            durations[utt_id] = handle.getnframes() / float(handle.getframerate())

    schedule_index = defaultdict(int)
    output_rows = []
    for source_index, row in enumerate(source_rows):
        utt_id = str(row["utt_id"])
        if utt_id not in scores:
            raise KeyError(f"Missing scores for utt_id={utt_id}")
        if utt_id not in wav_map:
            raise KeyError(f"Missing wav path for utt_id={utt_id}")

        duration = durations[utt_id]
        target = scores[utt_id]
        output_rows.append(
            {
                "manifest_row_index": source_index,
                "utt_id": utt_id,
                "split": args.split,
                "schedule_index": schedule_index[utt_id],
                "schedule_count": chunk_counts[utt_id],
                "chunk_id": int(row["chunk_id"]),
                "commit_time": float(row["commit_time"]),
                "audio_end": float(row["audio_end"]),
                "duration": duration,
                "progress": min(1.0, float(row["commit_time"]) / max(duration, 1e-8)),
                "is_final": bool(row["is_final"]),
                "visible_phone_count": int(
                    row.get(
                        "visible_phone_count",
                        row.get("visible_len", row.get("top_phone_count", 0)),
                    )
                ),
                "committed_phone_count": int(
                    row.get(
                        "committed_phone_count",
                        row.get("visible_len", row.get("top_phone_count", 0)),
                    )
                ),
                "matched_ratio": float(
                    row.get("matched_ratio", row.get("coverage_ratio", 0.0))
                ),
                "mean_asr_confidence": float(
                    row.get(
                        "mean_asr_confidence",
                        mean_numeric(
                            row.get("word_confidences"),
                            mean_numeric(row.get("token_confidences"), 0.0),
                        ),
                    )
                ),
                "wav_path": wav_map[utt_id],
                "reference_text": target["text"],
                "target_scores": {
                    key: float(target[key]) for key in SCORE_KEYS
                },
            }
        )
        schedule_index[utt_id] += 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "rows": len(output_rows),
                "utterances": len(chunk_counts),
                "source_manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
