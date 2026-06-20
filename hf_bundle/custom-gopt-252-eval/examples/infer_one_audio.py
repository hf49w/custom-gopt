import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch


SCORE_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Run one-audio inference with the bundled Whisper + Charsiu + Streaming GOPT pipeline."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to one wav audio file.")
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo-root", type=Path, required=True, help="Path to the cloned custom-gopt repository root.")
    parser.add_argument("--charsiu-src-dir", type=Path, required=True, help="Path to the official Charsiu repo root or its src directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu. Defaults to cuda if available.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--main-context-tokens", type=int, default=None)
    parser.add_argument("--right-context-tokens", type=int, default=None)
    return parser.parse_args()


def add_repo_paths(repo_root):
    repo_src = repo_root / "src"
    prep_src = repo_src / "prep_data"
    for path in [repo_src, prep_src]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_asr_pipeline(whisper_model_dir, device):
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    if str(device).startswith("cuda"):
        pipe_device = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        torch_dtype = torch.float16
    else:
        pipe_device = -1
        torch_dtype = torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(whisper_model_dir),
        tokenizer=str(whisper_model_dir),
        feature_extractor=str(whisper_model_dir),
        framework="pt",
        device=pipe_device,
        dtype=torch_dtype,
    )
    if hasattr(pipe.model, "generation_config"):
        pipe.model.generation_config.use_cache = False
    kwargs = {
        "return_timestamps": "word",
        "generate_kwargs": {
            "language": "english",
            "task": "transcribe",
            "max_new_tokens": 128,
            "use_cache": False,
        },
    }
    return pipe, kwargs


def load_audio_for_asr(audio_path, sample_rate):
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return audio, sr


def transcribe_audio(asr_pipe, pipe_kwargs, audio_path, sample_rate, normalize_word):
    audio, sr = load_audio_for_asr(audio_path, sample_rate)
    result = asr_pipe({"raw": audio, "sampling_rate": sr}, **pipe_kwargs)
    transcript = (result.get("text") or "").strip()
    words = []
    for chunk in result.get("chunks", []):
        text = normalize_word(chunk.get("text", ""))
        timestamp = chunk.get("timestamp") or (None, None)
        if not text or timestamp[0] is None or timestamp[1] is None:
            continue
        words.append(
            {
                "text": text,
                "start": float(timestamp[0]),
                "end": float(timestamp[1]),
            }
        )
    return transcript, words


def load_model(model_dir, repo_root, device):
    add_repo_paths(repo_root)
    from models import StreamingGOPT, StreamingGOPTNoPhn

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_args = cfg["args"]
    model_cls = StreamingGOPT if model_args["model"] == "streaming_gopt" else StreamingGOPTNoPhn
    model = model_cls(
        embed_dim=int(model_args["embed_dim"]),
        num_heads=int(model_args["heads"]),
        depth=int(model_args["depth"]),
        input_dim=int(cfg["input_dim"]),
        seq_len=int(cfg["seq_len"]),
        phn_num=int(cfg["phn_num"]),
    )
    state = torch.load(model_dir / "best_audio_model.pth", map_location=device)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "mlp_head_word4.0.weight",
        "mlp_head_word4.0.bias",
        "mlp_head_word4.1.weight",
        "mlp_head_word4.1.bias",
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    disallowed_missing = missing - allowed_missing
    if disallowed_missing:
        raise RuntimeError(f"Missing required checkpoint keys: {sorted(disallowed_missing)}")
    model = model.to(device)
    model.eval()
    return model, cfg


def build_phone_segments(
    audio_path,
    transcript,
    repo_root,
    charsiu_src_dir,
    charsiu_model_dir,
    sample_rate,
    device,
    phn_dict,
    expected_feat_dim,
):
    add_repo_paths(repo_root)
    for module_name in ["Charsiu", "models", "utils", "processors"]:
        sys.modules.pop(module_name, None)
    from build_charsiu_seq_data import (
        audio_logits,
        build_model_phone_map,
        build_silence_keep_mask,
        import_official_charsiu_forced_align,
        load_official_charsiu_aligner,
        normalize_phone,
        segment_feature,
    )

    charsiu = load_official_charsiu_aligner(
        model_name=str(charsiu_model_dir),
        device=str(device),
        sample_rate=sample_rate,
        sil_threshold=4,
        lang="en",
        charsiu_src_dir=str(charsiu_src_dir),
    )
    phone_to_frame_id, _, _ = build_model_phone_map(charsiu)
    phone_groups, words = charsiu.charsiu_processor.get_phones_and_words(transcript)
    if not phone_groups:
        raise ValueError("ASR transcript cannot be converted into phones.")

    flat_records = []
    for word_id, phones in enumerate(phone_groups):
        word_text = str(words[word_id]).lower() if word_id < len(words) else f"word_{word_id}"
        for phone in phones:
            norm_phone = normalize_phone(phone)
            if not norm_phone or norm_phone == "SIL":
                continue
            if norm_phone not in phone_to_frame_id:
                raise ValueError(f"Phone not found in Charsiu frame-classifier vocab: {norm_phone}")
            if norm_phone not in phn_dict:
                raise ValueError(f"Phone not found in GOPT phone vocab: {norm_phone}")
            flat_records.append(
                {
                    "phone": norm_phone,
                    "word_id": int(word_id),
                    "word_text": word_text,
                }
            )

    if not flat_records:
        raise ValueError("No valid phones remained after transcript normalization.")

    phone_ids = charsiu.charsiu_processor.get_phone_ids(phone_groups)
    target_phone_ids = list(phone_ids[1:-1])
    if len(target_phone_ids) != len(flat_records):
        raise ValueError(
            f"Phone count mismatch after G2P: target_phone_ids={len(target_phone_ids)} flat_records={len(flat_records)}"
        )

    probs, audio_duration = audio_logits(
        audio_path=str(audio_path),
        processor=charsiu.charsiu_processor,
        model=charsiu.aligner,
        sample_rate=sample_rate,
        device=device,
    )
    keep_mask = build_silence_keep_mask(charsiu, probs)
    kept_indices = np.flatnonzero(keep_mask)
    kept_probs = probs[keep_mask]
    if kept_probs.shape[0] < len(target_phone_ids):
        raise ValueError(
            f"Not enough non-silence frames for the recognized phones: frames={kept_probs.shape[0]} phones={len(target_phone_ids)}"
        )

    forced_align = import_official_charsiu_forced_align(str(charsiu_src_dir))
    aligned_phone_ids = np.asarray(forced_align(kept_probs, target_phone_ids), dtype=np.int32)
    frame_step = float(audio_duration) / max(len(probs), 1)

    segments = []
    for phone_idx, record in enumerate(flat_records):
        token_frames = np.flatnonzero(aligned_phone_ids == phone_idx)
        if token_frames.size == 0:
            raise ValueError(f"Empty aligned segment for phone index {phone_idx} ({record['phone']}).")

        segment_probs = kept_probs[token_frames]
        target_id = int(phone_to_frame_id[record["phone"]])
        base_feature = segment_feature(segment_probs, target_id, frame_step).astype(np.float32)
        expected_base_dim = int(expected_feat_dim) - 1
        expected_phone_prob_dim = expected_base_dim - 4
        current_phone_prob_dim = int(base_feature.shape[0]) - 4
        if current_phone_prob_dim == expected_phone_prob_dim + 1:
            # Current local Charsiu exports an extra [PAD] probability channel at the end.
            # The released GOPT checkpoint was trained without that channel.
            base_feature = np.concatenate([base_feature[:expected_phone_prob_dim], base_feature[-4:]], axis=0)
        elif current_phone_prob_dim != expected_phone_prob_dim:
            raise ValueError(
                f"Unexpected phone probability dimension: current={current_phone_prob_dim} expected={expected_phone_prob_dim}"
            )
        feature = np.concatenate([base_feature, np.array([0.0], dtype=np.float32)], axis=0)

        start_frame = int(kept_indices[token_frames[0]])
        end_frame = int(kept_indices[token_frames[-1]]) + 1
        start_time = float(start_frame * frame_step)
        end_time = float(min(audio_duration, end_frame * frame_step))

        segments.append(
            {
                "phone": record["phone"],
                "phone_id": int(phn_dict[record["phone"]]),
                "word_id": int(record["word_id"]),
                "word_text": record["word_text"],
                "start_time": start_time,
                "end_time": end_time,
                "feature": feature,
            }
        )
    return segments


def prepare_model_inputs(segments, seq_len, feat_dim, norm_mean, norm_std, device):
    if len(segments) > seq_len:
        raise ValueError(f"Phone sequence too long for this model: {len(segments)} > seq_len({seq_len})")

    feat = np.zeros((seq_len, feat_dim), dtype=np.float32)
    phn = np.full((seq_len,), -1, dtype=np.int64)
    for idx, segment in enumerate(segments):
        if int(segment["feature"].shape[-1]) != feat_dim:
            raise ValueError(
                f"Feature dimension mismatch at segment {idx}: got {segment['feature'].shape[-1]}, expected {feat_dim}"
            )
        feat[idx] = segment["feature"]
        phn[idx] = int(segment["phone_id"])

    valid = phn >= 0
    feat[valid] = (feat[valid] - float(norm_mean)) / float(norm_std)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    p = torch.from_numpy(phn).unsqueeze(0).to(device)
    return x, p


def predict_scores(model, x, p, main_context_tokens, right_context_tokens):
    with torch.no_grad():
        u1, u2, u3, u4, u5, _, _, _, _, _ = model(
            x,
            p,
            main_context_tokens=int(main_context_tokens),
            right_context_tokens=int(right_context_tokens),
        )
    values = torch.cat([u1, u2, u3, u4, u5], dim=1).squeeze(0).cpu().numpy() * 5.0
    values = np.clip(values, 0.0, 10.0)
    return {name: float(value) for name, value in zip(SCORE_NAMES, values.tolist())}


def build_output(audio_path, transcript, asr_words, scores, segments, device, bundle_dir):
    return {
        "status": "ok",
        "audio_path": str(audio_path.resolve()).replace("\\", "/"),
        "bundle_dir": str(bundle_dir.resolve()).replace("\\", "/"),
        "device": str(device),
        "transcript": transcript,
        "utterance_scores": scores,
        "overall_score": float(scores["total"]),
        "num_phone_segments": int(len(segments)),
        "num_asr_words": int(len(asr_words)),
        "recognized_words": [str(word["text"]).lower() for word in asr_words],
    }


def main():
    warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
    args = get_args()
    bundle_dir = args.bundle_dir.resolve()
    repo_root = args.repo_root.resolve()
    charsiu_src_dir = args.charsiu_src_dir.resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    model_dir = bundle_dir / "streaming_gopt_best"
    whisper_model_dir = bundle_dir / "whisper_best_model"
    charsiu_model_dir = bundle_dir / "charsiu_en_w2v2_tiny_fc_10ms"
    inference_assets_path = model_dir / "inference_assets.json"

    inference_assets = json.loads(inference_assets_path.read_text(encoding="utf-8"))
    sample_rate = int(inference_assets["sample_rate"])
    norm_mean = float(inference_assets["train_norm_mean"])
    norm_std = float(inference_assets["train_norm_std"])
    phn_dict = {str(key): int(value) for key, value in inference_assets["phn_dict"].items()}

    model, cfg = load_model(model_dir, repo_root, device)
    add_repo_paths(repo_root)
    from build_charsiu_seq_data import normalize_word

    asr_pipe, asr_kwargs = load_asr_pipeline(whisper_model_dir, device)
    transcript, asr_words = transcribe_audio(asr_pipe, asr_kwargs, args.audio, sample_rate, normalize_word)
    if not transcript:
        raise ValueError("Whisper produced an empty transcript.")

    segments = build_phone_segments(
        audio_path=args.audio,
        transcript=transcript,
        repo_root=repo_root,
        charsiu_src_dir=charsiu_src_dir,
        charsiu_model_dir=charsiu_model_dir,
        sample_rate=sample_rate,
        device=device,
        phn_dict=phn_dict,
        expected_feat_dim=int(cfg["input_dim"]),
    )

    seq_len = int(cfg["seq_len"])
    feat_dim = int(cfg["input_dim"])
    x, p = prepare_model_inputs(
        segments=segments,
        seq_len=seq_len,
        feat_dim=feat_dim,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
    )

    model_args = cfg["args"]
    main_context_choices = model_args.get("main_context_token_choices") or [
        int(item.strip()) for item in str(model_args["main_context_tokens"]).split(",") if item.strip()
    ]
    right_context_choices = model_args.get("right_context_token_choices") or [
        int(item.strip()) for item in str(model_args["right_context_tokens"]).split(",") if item.strip()
    ]

    main_context_tokens = int(args.main_context_tokens if args.main_context_tokens is not None else max(main_context_choices))
    right_context_tokens = int(
        args.right_context_tokens if args.right_context_tokens is not None else max(right_context_choices)
    )
    scores = predict_scores(model, x, p, main_context_tokens, right_context_tokens)
    payload = build_output(args.audio, transcript, asr_words, scores, segments, device, bundle_dir)

    result_json = json.dumps(payload, ensure_ascii=False, indent=2)
    print(result_json)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(result_json, encoding="utf-8")


if __name__ == "__main__":
    main()
