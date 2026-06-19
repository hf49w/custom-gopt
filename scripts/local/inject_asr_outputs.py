import argparse
import json
import re
import shutil
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Inject ASR full-text outputs into an existing selected-sample result directory.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--whisper-model-path", type=str, required=True)
    parser.add_argument("--asr-json", type=Path, required=True, help="JSON file mapping utt_id -> ASR text.")
    return parser.parse_args()


def main():
    args = get_args()
    asr_map = json.loads(args.asr_json.read_text(encoding="utf-8-sig"))

    if args.dst_root.exists():
        shutil.rmtree(args.dst_root)
    shutil.copytree(args.src_root, args.dst_root)

    summary_path = args.dst_root / "summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    obj.setdefault("model_paths", {})["whisper_best_model"] = args.whisper_model_path

    for item in obj["results"]:
        uid = item["utt_id"]
        item["whisper_model_path"] = args.whisper_model_path
        item["asr_full_output"] = {"text": asr_map.get(uid, "")}

    summary_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in obj["results"]:
        subset = item["subset"]
        uid = item["utt_id"]
        txt_path = args.dst_root / subset / f"{uid}.txt"
        text = txt_path.read_text(encoding="utf-8")
        whisper_line = f"WHISPER_MODEL_PATH: {item['whisper_model_path']}"
        asr_line = f"ASR_FULL_TEXT: {item['asr_full_output']['text']}"

        text = re.sub(r"^WHISPER_MODEL_PATH:.*$", whisper_line, text, count=1, flags=re.MULTILINE)
        if re.search(r"^ASR_FULL_TEXT:.*$", text, flags=re.MULTILINE):
            text = re.sub(r"^ASR_FULL_TEXT:.*$", asr_line, text, count=1, flags=re.MULTILINE)
        else:
            marker = "STREAMING_CONTEXT:"
            if marker in text:
                text = text.replace(marker, asr_line + "\n" + marker, 1)
        txt_path.write_text(text, encoding="utf-8")

    print(args.dst_root)


if __name__ == "__main__":
    main()
