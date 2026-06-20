import argparse
import importlib
import importlib.util
import json
import re
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Run original GOPT on open-set GOP features and emit a MultiPA-open-eval compatible prediction file."
    )
    parser.add_argument("--repo-src", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--seq-data-dir", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--transcript-tsv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--input-dim", type=int, default=84)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--multipa-repo-root", type=Path, default=None)
    parser.add_argument("--aligner", type=str, default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--align-device", type=str, default=None, help="Defaults to auto in Charsiu.")
    parser.add_argument("--invalid-utt-json", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
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
            mapping[utt_id] = transcript.strip()
    return mapping


def load_invalid_utt(path: Optional[Path]):
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        cleaned = {}
        for key, value in checkpoint.items():
            cleaned[key[7:] if key.startswith("module.") else key] = value
        checkpoint = cleaned
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def load_model(args):
    gopt_py = args.repo_src / "models" / "gopt.py"
    spec = importlib.util.spec_from_file_location("local_gopt_module", gopt_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GOPT module from {gopt_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    GOPT = module.GOPT

    model = GOPT(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        input_dim=args.input_dim,
    )
    ckpt_info = load_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device).eval()
    return model, device, ckpt_info


def load_seq_data(seq_data_dir: Path):
    feat = np.load(seq_data_dir / "te_feat.npy")
    phn = np.load(seq_data_dir / "te_label_phn.npy")
    word = np.load(seq_data_dir / "te_label_word.npy")
    utt_ids = [line.strip() for line in (seq_data_dir / "te_utt_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return feat, phn, word, utt_ids


def aggregate_word_scores(word_score_phone_level: np.ndarray, word_ids: np.ndarray):
    grouped = []
    prev_word_id = None
    cur_values = []
    for tok_idx in range(word_ids.shape[0]):
        word_id = int(word_ids[tok_idx])
        if word_id < 0:
            break
        value = word_score_phone_level[tok_idx]
        if prev_word_id is None:
            prev_word_id = word_id
        if word_id != prev_word_id:
            grouped.append(float(np.mean(cur_values)))
            cur_values = [value]
            prev_word_id = word_id
        else:
            cur_values.append(value)
    if cur_values:
        grouped.append(float(np.mean(cur_values)))
    return grouped


def get_audio_duration_sec(wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def build_uniform_alignment(words, duration_sec: float):
    if not words:
        return []
    step = duration_sec / max(len(words), 1)
    out = []
    for idx, word in enumerate(words):
        start = round(idx * step, 2)
        end = round((idx + 1) * step if idx < len(words) - 1 else duration_sec, 2)
        out.append([f"{start:.2f}", f"{end:.2f}", word])
    return out


def load_charsiu_aligner(multipa_repo_root: Optional[Path], aligner_name: str, align_device: Optional[str]):
    if multipa_repo_root is None:
        return None, None
    if not multipa_repo_root.exists():
        return None, None

    multipa_root_str = str(multipa_repo_root)
    if multipa_root_str not in sys.path:
        sys.path.insert(0, multipa_root_str)

    for module_name in ["Charsiu", "models", "utils", "processors", "utils_assessment"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            module_path = Path(module_file).resolve()
        except Exception:
            continue
        if multipa_repo_root.resolve() not in module_path.parents:
            sys.modules.pop(module_name, None)

    charsiu_module = importlib.import_module("Charsiu")
    utils_assessment_module = importlib.import_module("utils_assessment")
    charsiu_forced_aligner = charsiu_module.charsiu_forced_aligner
    get_match_index = utils_assessment_module.get_match_index

    kwargs = {"aligner": aligner_name}
    if align_device:
        kwargs["device"] = align_device
    aligner = charsiu_forced_aligner(**kwargs)
    return aligner, get_match_index


def charsiu_word_alignment(aligner, get_match_index, wav_path: Path, transcript: str):
    pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(audio=str(wav_path), text=transcript)
    selected_idx = get_match_index(pred_words, words)
    pred_words = np.asarray(pred_words)
    pred_words = pred_words[selected_idx]
    return pred_words.tolist()


def format_score_list(values):
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return ",".join(f"{value:.6f}" for value in values)


def main():
    args = get_args()
    model, device, ckpt_info = load_model(args)
    feat, phn, word, available_utt_ids = load_seq_data(args.seq_data_dir)
    manifest_rows = load_manifest(args.manifest_jsonl)
    transcript_map = load_transcripts(args.transcript_tsv)
    invalid_utts = load_invalid_utt(args.invalid_utt_json)
    aligner, get_match_index = load_charsiu_aligner(args.multipa_repo_root, args.aligner, args.align_device)

    utt_to_row = {utt_id: idx for idx, utt_id in enumerate(available_utt_ids)}
    outputs = []
    stats = {
        "available_feature_utts": len(available_utt_ids),
        "manifest_utts": len(manifest_rows),
        "predicted_valid_utts": 0,
        "invalid_utts": 0,
        "alignment_fallback_utts": 0,
        "alignment_charsiu_utts": 0,
        "checkpoint_info": ckpt_info,
    }

    for row in manifest_rows:
        utt_id = str(row["utt_id"])
        wav_path = Path(row["wav_path"])
        transcript = transcript_map.get(utt_id, "").strip().lower()
        invalid_reason = invalid_utts.get(utt_id, {}).get("reason")

        if invalid_reason or utt_id not in utt_to_row or not transcript:
            outputs.append(
                f"{utt_id}.wav; A:; F:; P:; T:; Valid:F; ASR_s:{transcript}; ASR_w:{transcript}; w_a:; w_s:; w_t:; alignment:"
            )
            stats["invalid_utts"] += 1
            continue

        row_idx = utt_to_row[utt_id]
        x = torch.tensor(feat[row_idx : row_idx + 1], dtype=torch.float32, device=device)
        phns = torch.tensor(phn[row_idx : row_idx + 1, :, 0], dtype=torch.float32, device=device)

        with torch.no_grad():
            u1, u2, u3, u4, u5, p, w1, w2, w3 = model(x, phns)

        utt_scores = [float(u.squeeze().cpu().item()) for u in [u1, u3, u4, u5]]
        word_ids = word[row_idx, :, 3]
        word_acc = aggregate_word_scores(w1.squeeze(0).squeeze(-1).cpu().numpy(), word_ids)
        word_stress = aggregate_word_scores(w2.squeeze(0).squeeze(-1).cpu().numpy(), word_ids)
        word_total = aggregate_word_scores(w3.squeeze(0).squeeze(-1).cpu().numpy(), word_ids)

        try:
            if aligner is not None and get_match_index is not None:
                alignment = charsiu_word_alignment(aligner, get_match_index, wav_path, transcript)
                stats["alignment_charsiu_utts"] += 1
            else:
                raise RuntimeError("charsiu_not_available")
        except Exception:
            duration_sec = get_audio_duration_sec(wav_path)
            alignment = build_uniform_alignment(transcript.split(), duration_sec)
            stats["alignment_fallback_utts"] += 1

        outputs.append(
            f"{utt_id}.wav; "
            f"A:{utt_scores[0]:.6f}; "
            f"F:{utt_scores[1]:.6f}; "
            f"P:{utt_scores[2]:.6f}; "
            f"T:{utt_scores[3]:.6f}; "
            f"Valid:T; "
            f"ASR_s:{transcript}; "
            f"ASR_w:{transcript}; "
            f"w_a:{format_score_list(word_acc)}; "
            f"w_s:{format_score_list(word_stress)}; "
            f"w_t:{format_score_list(word_total)}; "
            f"alignment:{alignment}"
        )
        stats["predicted_valid_utts"] += 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(outputs) + ("\n" if outputs else ""), encoding="utf-8")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
