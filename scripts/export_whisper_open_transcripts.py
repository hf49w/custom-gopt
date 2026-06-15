import argparse
import json
import os
import string
import unicodedata
from pathlib import Path
from typing import Optional

try:
    import num2words
except Exception:
    num2words = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Export Whisper transcripts for the streaming-test utterance subset."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default="openai/whisper-medium.en")
    parser.add_argument(
        "--backend",
        choices=["openai-whisper", "transformers"],
        default="transformers",
        help="transformers is forced offline; openai-whisper uses ~/.cache/whisper.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--openai-whisper-cache",
        type=Path,
        default=Path("~/.cache/whisper"),
    )
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing utterances and keep existing transcript rows.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="multipa_open",
        choices=["none", "multipa_open"],
        help="Transcript normalization strategy.",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def remove_pun_except_apostrophe(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation.replace("'", ""))
    return text.translate(translator).replace("  ", " ")


def convert_num_to_word(text: str) -> str:
    if num2words is None:
        return text
    try:
        int(text.replace(" ", ""))
        text = " ".join([char for char in text])
        text = " ".join([num2words.num2words(i) if i.isdigit() else i for i in text.split()])
        return text.replace("  ", " ")
    except Exception:
        return " ".join([num2words.num2words(i) if i.isdigit() else i for i in text.split()])


def ascii_fold(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_transcript(text: str, mode: str) -> str:
    text = (text or "").strip()
    if mode == "none":
        return " ".join(ascii_fold(text).split())
    if mode == "multipa_open":
        text = remove_pun_except_apostrophe(text).lower()
        text = convert_num_to_word(text)
        text = ascii_fold(text)
        return " ".join(text.split())
    raise ValueError(f"Unsupported normalize mode: {mode}")


def resolve_wav_path(raw_path: str, dataset_root: Optional[Path]) -> Path:
    wav_path = Path(raw_path)
    if wav_path.is_absolute():
        return wav_path
    if dataset_root is not None:
        candidate = dataset_root / wav_path
        if candidate.exists():
            return candidate
    return wav_path


def build_generate_kwargs(asr, model_name: str):
    generation_config = getattr(asr.model, "generation_config", None)
    is_multilingual = getattr(generation_config, "is_multilingual", None)
    if is_multilingual is None:
        tokenizer = getattr(asr, "tokenizer", None)
        is_multilingual = getattr(tokenizer, "is_multilingual", None)
    if is_multilingual is None:
        is_multilingual = not model_name.lower().endswith(".en")
    if is_multilingual:
        return {"language": "english", "task": "transcribe"}
    return {}


def normalize_openai_whisper_model_name(model_name: str) -> str:
    name = model_name.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.startswith("whisper-"):
        name = name[len("whisper-") :]
    return name


def transcribe_with_transformers(args, rows, completed, handle):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import pipeline

    if args.device.startswith("cuda"):
        pipe_device = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    else:
        pipe_device = -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=pipe_device,
        framework="pt",
    )
    generate_kwargs = build_generate_kwargs(asr, args.model)
    generate_kwargs.update(
        {
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        }
    )
    if hasattr(asr.feature_extractor, "return_attention_mask"):
        asr.feature_extractor.return_attention_mask = True
    pending_rows = [row for row in rows if str(row["utt_id"]) not in completed]
    processed = 0
    for batch_start in range(0, len(pending_rows), args.batch_size):
        batch_rows = pending_rows[batch_start : batch_start + args.batch_size]
        inputs = [
            str(resolve_wav_path(row["wav_path"], args.dataset_root))
            for row in batch_rows
        ]
        results = asr(
            inputs,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
        )
        for row, result in zip(batch_rows, results):
            transcript = normalize_transcript(
                result.get("text") or "", args.normalize_mode
            )
            handle.write(f"{row['utt_id']}\t{transcript}\n")
            processed += 1
        handle.flush()
        print(
            f"[whisper-transformers] {processed}/{len(pending_rows)}",
            flush=True,
        )
    return processed, generate_kwargs


def transcribe_with_openai_whisper(args, rows, completed, handle):
    import whisper

    model_name = normalize_openai_whisper_model_name(args.model)
    cache_dir = args.openai_whisper_cache.expanduser()
    checkpoint_path = cache_dir / f"{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing local Whisper checkpoint: {checkpoint_path}. "
            "Download it first or select --backend transformers."
        )
    model = whisper.load_model(
        model_name,
        device=args.device,
        download_root=str(cache_dir),
    )
    processed = 0
    for index, row in enumerate(rows, start=1):
        if str(row["utt_id"]) in completed:
            continue
        wav_path = resolve_wav_path(row["wav_path"], args.dataset_root)
        result = model.transcribe(
            str(wav_path),
            language="en",
            task="transcribe",
            fp16=args.device.startswith("cuda"),
            verbose=False,
            condition_on_previous_text=False,
        )
        transcript = normalize_transcript(result.get("text") or "", args.normalize_mode)
        handle.write(f"{row['utt_id']}\t{transcript}\n")
        handle.flush()
        processed += 1
        print(f"[whisper-openai] {index}/{len(rows)} {row['utt_id']}", flush=True)
    return processed, {"language": "en", "task": "transcribe"}


def main():
    args = get_args()
    rows = load_manifest(args.manifest_jsonl)
    if not rows:
        raise ValueError("Empty manifest")
    if args.limit > 0:
        rows = rows[: args.limit]

    completed = set()
    if args.resume and args.output_tsv.exists():
        for line in args.output_tsv.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(line.split("\t", 1)[0])

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output_tsv.open(mode, encoding="utf-8") as handle:
        if args.backend == "openai-whisper":
            processed, generate_kwargs = transcribe_with_openai_whisper(
                args, rows, completed, handle
            )
        else:
            processed, generate_kwargs = transcribe_with_transformers(
                args, rows, completed, handle
            )

    print(
        json.dumps(
            {
                "count": len(rows),
                "processed": processed,
                "skipped_existing": len(rows) - processed,
                "output_tsv": str(args.output_tsv),
                "model": args.model,
                "backend": args.backend,
                "generate_kwargs": generate_kwargs,
                "normalize_mode": args.normalize_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
