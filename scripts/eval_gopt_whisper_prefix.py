import argparse
import importlib.util
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


NORM_MEAN = 3.203
NORM_STD = 4.045
SCORE_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate original GOPT from Whisper-text-conditioned prefix GOP features."
    )
    parser.add_argument("--prefix-audio-manifest", type=Path, required=True)
    parser.add_argument("--transcript-tsv", type=Path, required=True)
    parser.add_argument("--seq-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--asr-model-name", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_model(args, device):
    gopt_path = args.repo_src / "models" / "gopt.py"
    spec = importlib.util.spec_from_file_location("prefix_gopt_module", gopt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean_state = OrderedDict()
    for key, value in state.items():
        clean_state[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def load_transcripts(path):
    transcripts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            utt_id, text = line.split("\t", 1)
            transcripts[utt_id] = text
    return transcripts


def normalize_features(feat):
    output = np.zeros_like(feat, dtype=np.float32)
    valid = feat[:, 0] != 0
    output[valid] = (feat[valid] - NORM_MEAN) / NORM_STD
    return output


def aggregate_words(word_pred, word_target):
    word_ids = word_target[:, 3]
    result = []
    for word_id in sorted(set(int(value) for value in word_ids if value >= 0)):
        positions = np.where(word_ids == word_id)[0]
        pred = word_pred[positions].mean(axis=0) * 5.0
        result.append(
            {
                "word_id": word_id,
                "pred_accuracy": float(pred[0]),
                "pred_stress": float(pred[1]),
                "pred_total": float(pred[2]),
            }
        )
    return result


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args, device)
    manifest_rows = [
        json.loads(line)
        for line in args.prefix_audio_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transcripts = load_transcripts(args.transcript_tsv)
    available_ids = [
        line.strip()
        for line in (args.seq_data_dir / "te_utt_ids.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    id_to_index = {utt_id: index for index, utt_id in enumerate(available_ids)}
    feat_all = np.load(args.seq_data_dir / "te_feat.npy", mmap_mode="r")
    phn_all = np.load(args.seq_data_dir / "te_label_phn.npy", mmap_mode="r")
    word_all = np.load(args.seq_data_dir / "te_label_word.npy", mmap_mode="r")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for batch_start in range(0, len(manifest_rows), args.batch_size):
            batch_rows = manifest_rows[batch_start : batch_start + args.batch_size]
            valid_rows = [row for row in batch_rows if row["utt_id"] in id_to_index]
            predictions = {}
            elapsed_per_row = 0.0
            if valid_rows:
                indices = [id_to_index[row["utt_id"]] for row in valid_rows]
                feat = np.stack(
                    [normalize_features(np.asarray(feat_all[index])) for index in indices]
                )
                phn = np.stack(
                    [np.asarray(phn_all[index, :, 0], dtype=np.int64) for index in indices]
                )
                x = torch.from_numpy(feat).to(device)
                p = torch.from_numpy(phn).to(device)
                started = time.perf_counter()
                with torch.no_grad():
                    outputs = model(x, p)
                elapsed_per_row = (time.perf_counter() - started) / len(valid_rows)
                utt_pred = torch.cat(outputs[:5], dim=1).cpu().numpy() * 5.0
                word_pred = torch.cat(outputs[6:9], dim=2).cpu().numpy()
                for local_index, row in enumerate(valid_rows):
                    seq_index = indices[local_index]
                    predictions[row["utt_id"]] = {
                        "scores": {
                            name: float(utt_pred[local_index, score_index])
                            for score_index, name in enumerate(SCORE_NAMES)
                        },
                        "word_scores": aggregate_words(
                            word_pred[local_index],
                            np.asarray(word_all[seq_index]),
                        ),
                    }

            for row in batch_rows:
                eval_id = row["utt_id"]
                record = dict(row)
                record["utt_id"] = row["source_utt_id"]
                record.update(
                    {
                        "eval_id": eval_id,
                        "model": "original_gopt",
                        "mode": "whisper_text_prefix_gop",
                        "asr_model": args.asr_model_name,
                        "asr_text": transcripts.get(eval_id, ""),
                        "batch_size": args.batch_size,
                        "timing_mode": (
                            "online_single_prefix"
                            if args.batch_size == 1
                            else "batch_amortized"
                        ),
                        "process_time_sec": elapsed_per_row,
                    }
                )
                prediction = predictions.get(eval_id)
                if prediction is None:
                    record["status"] = "missing_gop_features"
                else:
                    record["status"] = "ok"
                    record.update(prediction)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            print(
                f"[gopt-whisper-prefix] {min(batch_start + len(batch_rows), len(manifest_rows))}/{len(manifest_rows)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
