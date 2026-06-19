import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path, PureWindowsPath

import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate selected SpeechOcean samples with a trained StreamingGOPT model."
    )
    parser.add_argument("--selected-samples-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with train/val/test npz + manifests + metadata.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with best_audio_model.pth and config.json.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Optional full checkpoint path; if set, load checkpoint['model_state'] instead of best_audio_model.pth.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True, help="Repo src dir containing models/ and train_streaming_charsiu.py.")
    parser.add_argument("--whisper-model-dir", type=Path, default=None, help="Optional Whisper best_model dir used to produce ASR outputs.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-asr", action="store_true", help="Also run ASR on each streaming prefix chunk. This is slower and may be unstable on some local setups.")
    parser.add_argument("--chunk-asr-no-timestamps", action="store_true", help="Disable word timestamps for per-chunk ASR to reduce memory use. Chunk ASR will return text only.")
    parser.add_argument("--device", type=str, default="cpu", help="Deprecated fallback device used when --model-device/--asr-device are not set.")
    parser.add_argument("--model-device", type=str, default=None, help="Device for the StreamingGOPT model, e.g. cuda:0 or cpu.")
    parser.add_argument("--asr-device", type=str, default=None, help="Device for Whisper ASR, e.g. cuda:1 or cpu.")
    parser.add_argument("--main-context-tokens", type=int, default=8)
    parser.add_argument("--right-context-tokens", type=int, default=2)
    return parser.parse_args()


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def safe_corrcoef(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or ys.size < 2:
        return None
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def load_selected_samples(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for subset_key, subset_name in [("correct_samples", "correct"), ("error_samples", "error")]:
        for item in payload[subset_key]:
            row = dict(item)
            row["subset"] = subset_name
            rows.append(row)
    return payload, rows


def load_archives(data_root):
    archives = {}
    for split in ["train", "val", "test"]:
        manifest = []
        with (data_root / f"{split}_manifest.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                manifest.append(json.loads(line))
        npz = np.load(data_root / f"{split}_chunks.npz")
        archives[split] = {
            "manifest": manifest,
            "feat": npz["feat"].astype(np.float32),
            "phn_id": npz["phn_id"].astype(np.int64),
            "word_label": npz["word_label"].astype(np.float32),
            "utt_label": npz["utt_label"].astype(np.float32),
            "phone_loss_mask": npz["phone_loss_mask"].astype(np.float32),
            "word_loss_mask": npz["word_loss_mask"].astype(np.float32),
            "utt_loss_mask": npz["utt_loss_mask"].astype(np.float32),
            "is_final": npz["is_final"].astype(np.int8),
            "visible_len": npz["visible_len"].astype(np.int32),
        }
    return archives


def build_manifest_index(archives):
    index = {}
    for split, bundle in archives.items():
        for idx, row in enumerate(bundle["manifest"]):
            index.setdefault(row["utt_id"], []).append((split, idx))
    for rows in index.values():
        rows.sort(key=lambda item: (item[0], item[1]))
    return index


def load_model(model_dir, repo_src, device, checkpoint_path=None):
    import sys

    sys.path.insert(0, str(repo_src))
    from models import StreamingGOPT, StreamingGOPTNoPhn
    from train_streaming_charsiu import load_model_state

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    model_cls = StreamingGOPT if args["model"] == "streaming_gopt" else StreamingGOPTNoPhn
    model = model_cls(
        embed_dim=args["embed_dim"],
        num_heads=args["heads"],
        depth=args["depth"],
        input_dim=cfg["input_dim"],
        seq_len=cfg["seq_len"],
        phn_num=cfg["phn_num"],
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint["model_state"]
    else:
        state = torch.load(model_dir / "best_audio_model.pth", map_location=device)
    load_model_state(model, state)
    model = model.to(device)
    model.eval()
    return model, cfg


def resolve_runtime_devices(args):
    fallback = args.device
    model_device = args.model_device or fallback
    asr_device = args.asr_device or fallback
    return torch.device(model_device), model_device, asr_device


def load_asr_pipeline(whisper_model_dir, device, chunk_asr_no_timestamps=False):
    if whisper_model_dir is None:
        return None, None, None

    from transformers import pipeline

    if str(device).startswith("cuda"):
        pipe_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        pipe_dtype = torch.float16
    else:
        pipe_device = -1
        pipe_dtype = torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(whisper_model_dir),
        tokenizer=str(whisper_model_dir),
        feature_extractor=str(whisper_model_dir),
        framework="pt",
        device=pipe_device,
        dtype=pipe_dtype,
    )
    if hasattr(pipe.model, "generation_config"):
        pipe.model.generation_config.use_cache = False

    generate_kwargs = {
        "language": "english",
        "task": "transcribe",
        "max_new_tokens": 128,
        "use_cache": False,
    }
    full_kwargs = {
        "return_timestamps": "word",
        "generate_kwargs": dict(generate_kwargs),
    }
    if chunk_asr_no_timestamps:
        chunk_kwargs = {
            "generate_kwargs": dict(generate_kwargs),
        }
    else:
        chunk_kwargs = {
            "return_timestamps": "word",
            "generate_kwargs": dict(generate_kwargs),
        }
    return pipe, full_kwargs, chunk_kwargs


def run_asr(pipe, pipe_kwargs, wav_path, sample_rate, audio_end=None):
    if pipe is None:
        return None

    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    if audio_end is not None:
        end_frame = max(1, int(round(float(audio_end) * sr)))
        audio = audio[: min(len(audio), end_frame)]

    result = pipe({"raw": audio, "sampling_rate": sr}, **pipe_kwargs)
    chunks = result.get("chunks") or []
    return {
        "text": (result.get("text") or "").strip(),
        "words": [
            {
                "text": (chunk.get("text") or "").strip(),
                "start": None if (chunk.get("timestamp") or (None, None))[0] is None else float((chunk.get("timestamp") or (None, None))[0]),
                "end": None if (chunk.get("timestamp") or (None, None))[1] is None else float((chunk.get("timestamp") or (None, None))[1]),
            }
            for chunk in chunks
        ],
    }


def aggregate_word_scores(word_outputs, word_targets):
    word_ids = word_targets[:, 4]
    valid_positions = np.where(word_ids >= 0)[0].tolist()
    if not valid_positions:
        return []

    rows = []
    start = valid_positions[0]
    prev_word = int(word_ids[start])
    positions = [start]
    for pos in valid_positions[1:]:
        cur_word = int(word_ids[pos])
        if cur_word != prev_word:
            rows.append((prev_word, positions))
            positions = [pos]
            prev_word = cur_word
        else:
            positions.append(pos)
    rows.append((prev_word, positions))

    aggregated = []
    for word_id, positions in rows:
        pred = word_outputs[positions].mean(axis=0)
        pred_main = pred[0:3] * 5.0
        target = word_targets[positions, 0:4].mean(axis=0)
        aggregated.append(
            {
                "word_id": int(word_id),
                "pred_accuracy": float(pred_main[0]),
                "pred_stress": float(pred_main[1]),
                "pred_total": float(pred_main[2]),
                "pred_asr_accuracy": float(pred[3]),
                "target_accuracy": float(target[0]),
                "target_stress": float(target[1]),
                "target_total": float(target[2]),
                "target_asr_accuracy": float(target[3]),
            }
        )
    return aggregated


def predict_chunk(model, archive, idx, norm_mean, norm_std, device, main_context_tokens, right_context_tokens):
    feat = archive["feat"][idx].copy()
    phn_id = archive["phn_id"][idx].copy()
    valid = phn_id >= 0
    feat[valid] = (feat[valid] - norm_mean) / norm_std

    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    p = torch.from_numpy(phn_id).unsqueeze(0).to(device)
    with torch.no_grad():
        u1, u2, u3, u4, u5, phn_out, w1, w2, w3, w4 = model(
            x,
            p,
            main_context_tokens=main_context_tokens,
            right_context_tokens=right_context_tokens,
        )
    utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1).squeeze(0).cpu().numpy() * 5.0
    word_outputs = torch.cat([w1, w2, w3, w4], dim=2).squeeze(0).cpu().numpy()
    word_scores = aggregate_word_scores(word_outputs, archive["word_label"][idx])
    return utt_pred, word_scores


def format_txt_report(result):
    lines = []
    lines.append(f"WAV_REL_PATH: {result['wav_relpath']}")
    lines.append(f"WAV_PATH: {result['wav_path']}")
    lines.append(f"REF_TEXT: {result['ref_text']}")
    lines.append(f"SPLIT: {result['split']}")
    lines.append(f"MODEL_PATH: {result['model_path']}")
    lines.append(f"WHISPER_MODEL_PATH: {result.get('whisper_model_path')}")
    if result.get("asr_full_output"):
        lines.append(f"ASR_FULL_TEXT: {result['asr_full_output']['text']}")
    lines.append(
        f"STREAMING_CONTEXT: main_context_tokens={result['main_context_tokens']} right_context_tokens={result['right_context_tokens']}"
    )
    lines.append("")
    lines.append("GOPT_FINAL_UTTERANCE_SCORES:")
    for key in ["accuracy", "completeness", "fluency", "prosodic", "total"]:
        pred = result["final_result"]["utterance_scores"][key]
        target = result["dataset_scores"][key]
        lines.append(f"  {key}: pred={pred:.6f} target={target:.6f} abs_err={abs(pred - target):.6f}")

    lines.append("")
    lines.append("GOPT_FINAL_WORD_SCORES:")
    if result["final_result"]["word_scores"]:
        for row in result["final_result"]["word_scores"]:
            word_text = row["word_text"]
            lines.append(
                "  "
                f"{row['word_id']:02d}  word={word_text}  "
                f"pred_acc={row['pred_accuracy']:.6f}  pred_stress={row['pred_stress']:.6f}  pred_total={row['pred_total']:.6f}  "
                f"pred_asr_acc={row['pred_asr_accuracy']:.6f}  "
                f"target_acc={row['target_accuracy']:.6f}  target_stress={row['target_stress']:.6f}  target_total={row['target_total']:.6f}  "
                f"target_asr_acc={row['target_asr_accuracy']:.6f}"
            )
    else:
        lines.append("  <none>")

    lines.append("")
    lines.append("STREAMING_CHUNKS:")
    for row in result["streaming_results"]:
        scores = row["utterance_scores"]
        lines.append(
            "  "
            f"chunk={row['chunk_id']:02d} "
            f"commit={row['commit_time']:.3f}s "
            f"audio_end={row['audio_end']:.3f}s "
            f"visible={row['visible_phone_count']:03d} "
            f"committed={row['committed_phone_count']:03d} "
            f"matched_ratio={row['matched_ratio']:.3f} "
            f"is_final={int(row['is_final'])} "
            f"asr={json.dumps(row['asr_output']['text'], ensure_ascii=False) if row.get('asr_output') else 'null'} "
            f"acc={scores['accuracy']:.4f} "
            f"comp={scores['completeness']:.4f} "
            f"flu={scores['fluency']:.4f} "
            f"pro={scores['prosodic']:.4f} "
            f"total={scores['total']:.4f}"
        )

    lines.append("")
    lines.append("DATASET_SCORES (utterance-level from selected_samples.json):")
    for key in ["accuracy", "completeness", "fluency", "prosodic", "total"]:
        lines.append(f"  {key}: {result['dataset_scores'][key]}")
    lines.append(f"  text: {result['dataset_scores']['text']}")

    lines.append("")
    lines.append("DATASET_WORD_SCORES (per-word from selected_samples.json):")
    for word in result["dataset_scores"]["words"]:
        phones = " ".join(word["phones"])
        phones_acc = ", ".join(str(x) for x in word["phones-accuracy"])
        lines.append(
            f"  {word['text']:<16} total={word['total']}  acc={word['accuracy']}  stress={word['stress']}  "
            f"phones=[{phones}]  phones-acc=[{phones_acc}]"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = get_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "correct").mkdir(parents=True, exist_ok=True)
    (args.output_root / "error").mkdir(parents=True, exist_ok=True)

    selected_payload, selected_rows = load_selected_samples(args.selected_samples_json)
    archives = load_archives(args.data_root)
    manifest_index = build_manifest_index(archives)
    metadata = json.loads((args.data_root / "metadata.json").read_text(encoding="utf-8"))
    norm_mean = float(metadata["train_norm_mean"])
    norm_std = float(metadata["train_norm_std"])

    model_device, model_device_str, asr_device_str = resolve_runtime_devices(args)
    model, cfg = load_model(args.model_dir, args.repo_src, model_device, checkpoint_path=args.checkpoint_path)
    asr_pipe, asr_full_kwargs, asr_chunk_kwargs = load_asr_pipeline(
        args.whisper_model_dir,
        asr_device_str,
        chunk_asr_no_timestamps=args.chunk_asr_no_timestamps,
    )

    results = []
    started = time.time()
    for sample in selected_rows:
        utt_id = sample["utt_id"]
        print(f"[eval] start utt_id={utt_id} subset={sample['subset']} split={sample['split']}", flush=True)
        if utt_id not in manifest_index:
            raise KeyError(f"utt_id {utt_id} not found in manifests")

        try:
            rows = manifest_index[utt_id]
            per_chunk = []
            wav_name = PureWindowsPath(sample["copied_to"]).name
            wav_path = args.selected_samples_json.parent / sample["subset"] / wav_name
            full_asr_output = run_asr(asr_pipe, asr_full_kwargs, wav_path, args.sample_rate) if asr_pipe else None
            for split, idx in rows:
                archive = archives[split]
                manifest_row = archive["manifest"][idx]
                utt_pred, word_scores = predict_chunk(
                    model=model,
                    archive=archive,
                    idx=idx,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    device=model_device,
                    main_context_tokens=args.main_context_tokens,
                    right_context_tokens=args.right_context_tokens,
                )
                asr_output = run_asr(
                    asr_pipe,
                    asr_chunk_kwargs,
                    wav_path,
                    args.sample_rate,
                    audio_end=float(manifest_row["audio_end"]),
                ) if (asr_pipe and args.chunk_asr) else None
                score_keys = ["accuracy", "completeness", "fluency", "prosodic", "total"]
                per_chunk.append(
                    {
                        "split": split,
                        "index": int(idx),
                        "chunk_id": int(manifest_row["chunk_id"]),
                        "commit_time": float(manifest_row["commit_time"]),
                        "audio_end": float(manifest_row["audio_end"]),
                        "visible_phone_count": int(manifest_row["visible_phone_count"]),
                        "committed_phone_count": int(manifest_row["committed_phone_count"]),
                        "matched_ratio": float(manifest_row["matched_ratio"]),
                        "is_final": bool(manifest_row["is_final"]),
                        "visible_len": int(archive["visible_len"][idx]),
                        "asr_output": asr_output,
                        "utterance_scores": {
                            key: float(value) for key, value in zip(score_keys, utt_pred.tolist())
                        },
                        "word_scores": word_scores,
                    }
                )
        except Exception:
            print(f"[eval] failed utt_id={utt_id}", flush=True)
            traceback.print_exc()
            raise

        per_chunk.sort(key=lambda row: (row["commit_time"], row["chunk_id"]))
        final_chunk = next((row for row in per_chunk if row["is_final"]), per_chunk[-1])

        dataset_words = sample["scores"]["words"]
        aligned_word_scores = []
        for i, row in enumerate(final_chunk["word_scores"]):
            out = dict(row)
            out["word_text"] = dataset_words[i]["text"].lower() if i < len(dataset_words) else f"<word_{i}>"
            aligned_word_scores.append(out)
        final_chunk = dict(final_chunk)
        final_chunk["word_scores"] = aligned_word_scores

        windows_wav_path = PureWindowsPath(sample["copied_to"])
        wav_name = windows_wav_path.name
        wav_stem = Path(wav_name).stem
        try:
            wav_relpath = str(wav_path.relative_to(args.selected_samples_json.parent)).replace("\\", "/")
        except ValueError:
            wav_relpath = wav_name

        result = {
            "utt_id": utt_id,
            "subset": sample["subset"],
            "split": sample["split"],
            "wavname": wav_name,
            "wav_path": str(wav_path).replace("\\", "/"),
            "wav_relpath": wav_relpath,
            "ref_text": sample["scores"]["text"],
            "dataset_scores": sample["scores"],
            "stats": sample["stats"],
            "model_path": str(args.checkpoint_path if args.checkpoint_path is not None else args.model_dir / "best_audio_model.pth"),
            "whisper_model_path": None if args.whisper_model_dir is None else str(args.whisper_model_dir),
            "model_device": model_device_str,
            "asr_device": asr_device_str if args.whisper_model_dir is not None else None,
            "chunk_asr_no_timestamps": bool(args.chunk_asr_no_timestamps),
            "data_root": str(args.data_root),
            "main_context_tokens": int(args.main_context_tokens),
            "right_context_tokens": int(args.right_context_tokens),
            "asr_full_output": full_asr_output,
            "final_result": final_chunk,
            "streaming_results": per_chunk,
        }
        results.append(result)

        txt_path = args.output_root / sample["subset"] / f"{wav_stem}.txt"
        txt_path.write_text(format_txt_report(result), encoding="utf-8")

    aggregate = {}
    score_names = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    for subset in ["correct", "error"]:
        subset_rows = [row for row in results if row["subset"] == subset]
        aggregate[subset] = {
            "count": len(subset_rows),
            "final_score_mean": {
                name: float(np.mean([row["final_result"]["utterance_scores"][name] for row in subset_rows]))
                for name in score_names
            },
            "dataset_score_mean": {
                name: float(np.mean([row["dataset_scores"][name] for row in subset_rows]))
                for name in score_names
            },
            "final_score_mae": {
                name: float(
                    np.mean(
                        [
                            abs(row["final_result"]["utterance_scores"][name] - row["dataset_scores"][name])
                            for row in subset_rows
                        ]
                    )
                )
                for name in score_names
            },
        }

    aggregate["gap"] = {
        name: aggregate["correct"]["final_score_mean"][name] - aggregate["error"]["final_score_mean"][name]
        for name in score_names
    }
    aggregate["all_samples"] = {
        "count": len(results),
        "final_dataset_corr": {
            name: safe_corrcoef(
                [row["final_result"]["utterance_scores"][name] for row in results],
                [row["dataset_scores"][name] for row in results],
            )
            for name in score_names
        },
    }

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_root": str(args.selected_samples_json.parent),
        "output_root": str(args.output_root),
        "model_paths": {
            "best_audio_model": str(args.model_dir / "best_audio_model.pth"),
            "config": str(args.model_dir / "config.json"),
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else None,
            "whisper_best_model": None if args.whisper_model_dir is None else str(args.whisper_model_dir),
        },
        "runtime": {
            "model_device": model_device_str,
            "asr_device": asr_device_str if args.whisper_model_dir is not None else None,
            "chunk_asr": bool(args.chunk_asr),
            "chunk_asr_no_timestamps": bool(args.chunk_asr_no_timestamps),
        },
        "data_root": str(args.data_root),
        "streaming_context": {
            "main_context_tokens": int(args.main_context_tokens),
            "right_context_tokens": int(args.right_context_tokens),
        },
        "source": {
            "selected_samples_json": str(args.selected_samples_json),
            "metadata_json": str(args.data_root / "metadata.json"),
        },
        "count": len(results),
        "aggregate": to_builtin(aggregate),
        "results": to_builtin(results),
        "process_time_sec": time.time() - started,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(to_builtin(summary["aggregate"]), ensure_ascii=False, indent=2))
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
