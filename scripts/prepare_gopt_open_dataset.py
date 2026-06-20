import argparse
import json
import re
import shutil
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from g2p_en import G2p
except Exception:
    G2p = None

try:
    import num2words
except Exception:
    num2words = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare a pseudo open-set Speechocean subset for Kaldi/GOPT from Whisper transcripts."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--transcript-tsv", type=Path, required=True)
    parser.add_argument("--lexicon-txt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--max-phones",
        type=int,
        default=0,
        help="Truncate canonical text at a word boundary to this phone count; 0 disables.",
    )
    parser.add_argument(
        "--reference-text-json",
        type=Path,
        default=None,
        help="Optional scores.json for reporting GT/reference text alongside the pseudo dataset.",
    )
    parser.add_argument(
        "--make-train-copy",
        action="store_true",
        help="Also duplicate the valid subset into output_root/train for workflows that expect train+test.",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_transcripts(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, transcript = line.split("\t", maxsplit=1)
            mapping[utt_id] = transcript
    return mapping


def load_reference_text(path: Optional[Path]):
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lexicon(path: Path):
    lexicon = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].upper()
            phones = parts[1:]
            if word not in lexicon:
                lexicon[word] = phones
    return lexicon


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


def normalize_transcript(text: str) -> str:
    text = remove_pun_except_apostrophe((text or "").strip()).lower()
    text = convert_num_to_word(text)
    text = ascii_fold(text)
    return " ".join(text.split())


def tokenize_transcript(text: str):
    text = normalize_transcript(text)
    words = []
    for token in text.split():
        token = token.strip()
        if token:
            words.append(token.upper())
    return words


def strip_phone_markers(phone: str):
    return re.sub(r"[_\d].*$", "", phone)


def phones_to_bie(phones):
    if not phones:
        return []
    if len(phones) == 1:
        return [f"{phones[0]}_S"]
    out = []
    for idx, phone in enumerate(phones):
        if idx == 0:
            out.append(f"{phone}_B")
        elif idx == len(phones) - 1:
            out.append(f"{phone}_E")
        else:
            out.append(f"{phone}_I")
    return out


def resolve_word_phones(word: str, lexicon: dict, g2p):
    if word in lexicon:
        return lexicon[word], "lexicon"
    if g2p is not None:
        phones = [token for token in g2p(word.lower()) if re.fullmatch(r"[A-Z]+[0-2]?", str(token))]
        if phones:
            return phones, "g2p_en"
    raise KeyError(word)


def build_records(manifest_rows, transcript_map, lexicon, reference_text, max_phones=0):
    g2p = G2p() if G2p is not None else None

    valid_rows = []
    pseudo_scores = {}
    text_phone_lines = []
    added_lexicon = {}
    skipped = {}
    phone_source_counter = Counter()

    for row in manifest_rows:
        utt_id = str(row["utt_id"])
        wav_path = row.get("wav_path")
        transcript = transcript_map.get(utt_id, "")
        words = tokenize_transcript(transcript)
        skip_reason = None
        if not wav_path:
            skip_reason = "missing_wav_path"
        elif not words:
            skip_reason = "empty_transcript"

        word_entries = []
        accepted_words = []
        utt_text_phone_lines = []
        utt_added_lexicon = {}
        utt_phone_source_counter = Counter()
        if skip_reason is None:
            for word_idx, word in enumerate(words):
                try:
                    phones, source = resolve_word_phones(word, lexicon, g2p)
                    utt_phone_source_counter[source] += 1
                    if word not in lexicon:
                        utt_added_lexicon[word] = phones
                except KeyError:
                    skip_reason = f"unresolved_word:{word}"
                    break

                if max_phones > 0:
                    current_phone_count = sum(
                        len(entry["phones"]) for entry in word_entries
                    )
                    if current_phone_count + len(phones) > max_phones:
                        break

                utt_text_phone_lines.append(f"{utt_id}.{word_idx}\t{' '.join(phones_to_bie(phones))}")
                accepted_words.append(word)
                word_entries.append(
                    {
                        "text": word,
                        "accuracy": 0.0,
                        "stress": 0.0,
                        "total": 0.0,
                        "phones": phones,
                        "phones-accuracy": [2.0] * len(phones),
                    }
                )

        if skip_reason is None and not accepted_words:
            skip_reason = "no_words_within_phone_limit"

        if skip_reason is not None:
            skipped[utt_id] = {
                "reason": skip_reason,
                "wav_path": wav_path,
                "transcript": transcript,
                "reference_text": reference_text.get(utt_id, {}).get("text") if reference_text else None,
            }
            continue

        text_phone_lines.extend(utt_text_phone_lines)
        added_lexicon.update(utt_added_lexicon)
        phone_source_counter.update(utt_phone_source_counter)
        valid_rows.append(
            {
                "utt_id": utt_id,
                "wav_path": wav_path,
                "transcript": " ".join(accepted_words),
            }
        )
        pseudo_scores[utt_id] = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "fluency": 0.0,
            "prosodic": 0.0,
            "total": 0.0,
            "words": word_entries,
        }

    return valid_rows, pseudo_scores, text_phone_lines, added_lexicon, skipped, phone_source_counter


def write_split_dir(split_dir: Path, rows):
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = []
    text_lines = []
    utt2spk = []
    spk2utt = {}
    for row in rows:
        utt_id = row["utt_id"]
        wav_path = row["wav_path"]
        spk = f"SPK_{utt_id[:5]}"
        wav_scp.append(f"{utt_id}\t{wav_path}")
        text_lines.append(f"{utt_id}\t{row['transcript']}")
        utt2spk.append(f"{utt_id}\t{spk}")
        spk2utt.setdefault(spk, []).append(utt_id)

    (split_dir / "wav.scp").write_text("\n".join(wav_scp) + "\n", encoding="utf-8")
    (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (split_dir / "utt2spk").write_text("\n".join(utt2spk) + "\n", encoding="utf-8")
    spk2utt_lines = [f"{spk}\t{' '.join(utts)}" for spk, utts in sorted(spk2utt.items())]
    (split_dir / "spk2utt").write_text("\n".join(spk2utt_lines) + "\n", encoding="utf-8")


def main():
    args = get_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest_jsonl)
    transcript_map = load_transcripts(args.transcript_tsv)
    reference_text = load_reference_text(args.reference_text_json)
    lexicon = load_lexicon(args.lexicon_txt)

    valid_rows, pseudo_scores, text_phone_lines, added_lexicon, skipped, phone_source_counter = build_records(
        manifest_rows,
        transcript_map,
        lexicon,
        reference_text,
        max_phones=args.max_phones,
    )

    resource_dir = args.output_root / "resource"
    resource_dir.mkdir(parents=True, exist_ok=True)

    lexicon_out = resource_dir / "lexicon.txt"
    shutil.copyfile(args.lexicon_txt, lexicon_out)
    if added_lexicon:
        with lexicon_out.open("a", encoding="utf-8") as f:
            for word, phones in sorted(added_lexicon.items()):
                f.write(f"{word} {' '.join(phones)}\n")

    (resource_dir / "text-phone").write_text("\n".join(text_phone_lines) + ("\n" if text_phone_lines else ""), encoding="utf-8")
    with (resource_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(pseudo_scores, f, ensure_ascii=False, indent=2)

    write_split_dir(args.output_root / args.split, valid_rows)
    if args.make_train_copy:
        write_split_dir(args.output_root / "train", valid_rows)

    with (args.output_root / "valid_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in valid_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_root / "skipped_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    summary = {
        "input_manifest_count": len(manifest_rows),
        "transcript_count": len(transcript_map),
        "valid_count": len(valid_rows),
        "skipped_count": len(skipped),
        "added_lexicon_words": len(added_lexicon),
        "phone_source_counter": dict(phone_source_counter),
        "output_root": str(args.output_root),
        "split_dir": str(args.output_root / args.split),
        "resource_dir": str(resource_dir),
    }
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
