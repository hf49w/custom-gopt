import argparse
import json
import wave
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop every shared streaming row into a uniquely named prefix WAV."
    )
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def crop_wav(source_path, output_path, audio_end):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        frame_count = min(
            params.nframes,
            max(1, int(round(float(audio_end) * params.framerate))),
        )
        frames = source.readframes(frame_count)
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.setnframes(frame_count)
        target.writeframes(frames)


def main():
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.prefix_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    wav_dir = args.output_root / "wav"
    output_manifest = args.output_root / "prefix_audio_manifest.jsonl"
    args.output_root.mkdir(parents=True, exist_ok=True)

    with output_manifest.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            eval_id = "{}_c{:04d}".format(row["utt_id"], int(row["chunk_id"]))
            prefix_path = wav_dir / f"{eval_id}.wav"
            crop_wav(Path(row["wav_path"]), prefix_path, row["audio_end"])
            record = dict(row)
            record.update(
                {
                    "source_utt_id": str(row["utt_id"]),
                    "utt_id": eval_id,
                    "wav_path": str(prefix_path.absolute()),
                }
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 100 == 0 or index == len(rows):
                print(f"[prefix-wav] {index}/{len(rows)}", flush=True)

    print(
        json.dumps(
            {
                "rows": len(rows),
                "wav_dir": str(wav_dir),
                "manifest": str(output_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
