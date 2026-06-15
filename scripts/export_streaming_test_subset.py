import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Export the final-utterance subset used by the current streaming GOPT test split."
    )
    parser.add_argument("--streaming-data-root", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None, help="SpeechOcean762 root containing train/test wav.scp")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_scores(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_wav_map(dataset_root: Path):
    wav_map = {}
    for part in ["train", "test"]:
        scp_path = dataset_root / part / "wav.scp"
        if not scp_path.exists():
            continue
        for line in scp_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path = line.split(maxsplit=1)
            wav_path_obj = Path(wav_path)
            if not wav_path_obj.is_absolute():
                wav_path_obj = dataset_root / wav_path_obj
            wav_map[utt_id] = str(wav_path_obj.resolve())
    return wav_map


def load_final_rows(streaming_root: Path, split: str):
    rows = []
    manifest_path = streaming_root / f"{split}_manifest.jsonl"
    seen = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("is_final", False):
            continue
        utt_id = str(row["utt_id"])
        if utt_id in seen:
            raise ValueError(f"Duplicate final chunk for utt_id={utt_id}")
        seen.add(utt_id)
        rows.append(row)
    return rows


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(args.scores_json)
    wav_map = load_wav_map(args.dataset_root) if args.dataset_root is not None else {}
    final_rows = load_final_rows(args.streaming_data_root, args.split)

    records = []
    for row in final_rows:
        utt_id = str(row["utt_id"])
        score = scores[utt_id]
        record = {
            "utt_id": utt_id,
            "split": args.split,
            "wav_path": wav_map.get(utt_id),
            "text": score["text"],
            "scores": {
                "accuracy": score["accuracy"],
                "completeness": score["completeness"],
                "fluency": score["fluency"],
                "prosodic": score["prosodic"],
                "total": score["total"],
            },
            "streaming_final": {
                "chunk_id": int(row["chunk_id"]),
                "commit_time": float(row["commit_time"]),
                "audio_end": float(row["audio_end"]),
                "matched_ratio": float(row["matched_ratio"]),
                "utt_loss_mask": float(row["utt_loss_mask"]),
            },
        }
        records.append(record)

    records.sort(key=lambda item: item["utt_id"])
    manifest_out = args.output_dir / f"{args.split}_streaming_subset.jsonl"
    utt_ids_out = args.output_dir / f"{args.split}_utt_ids.txt"

    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    utt_ids_out.write_text("".join(f"{row['utt_id']}\n" for row in records), encoding="utf-8")

    summary = {
        "split": args.split,
        "count": len(records),
        "manifest": str(manifest_out),
        "utt_ids": str(utt_ids_out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
