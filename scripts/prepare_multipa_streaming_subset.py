import argparse
import json
import os
import shutil
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare a flat wav subset for running original MultiPA on the streaming-test utterance set."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    args = get_args()
    rows = load_manifest(args.manifest_jsonl)
    if not rows:
        raise ValueError("Empty manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "wav_flat"
    wav_dir.mkdir(parents=True, exist_ok=True)

    list_lines = []
    mapping = {}
    missing = []

    for row in rows:
        utt_id = str(row["utt_id"])
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            missing.append({"utt_id": utt_id, "wav_path": str(wav_path)})
            continue
        flat_name = f"{utt_id}.wav"
        flat_path = wav_dir / flat_name
        ensure_link_or_copy(wav_path, flat_path)
        list_lines.append(flat_name)
        mapping[utt_id] = {
            "source_wav_path": str(wav_path),
            "flat_wav_path": str(flat_path),
            "flat_name": flat_name,
        }

    datalist_path = args.output_dir / "test_streaming_subset.txt"
    datalist_path.write_text("\n".join(list_lines) + ("\n" if list_lines else ""), encoding="utf-8")

    with (args.output_dir / "utt_map.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "missing_wavs.json").open("w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    summary = {
        "manifest_count": len(rows),
        "prepared_count": len(list_lines),
        "missing_count": len(missing),
        "wav_dir": str(wav_dir),
        "datalist": str(datalist_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
