import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


SCORE_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate StreamingGOPT on the existing streaming test chunks."
    )
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--main-context-tokens", type=int, default=8)
    parser.add_argument("--right-context-tokens", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model(model_dir, repo_src, device):
    import sys

    sys.path.insert(0, str(repo_src))
    from models import StreamingGOPT, StreamingGOPTNoPhn
    from train_streaming_charsiu import load_model_state

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    train_args = config["args"]
    model_class = (
        StreamingGOPT
        if train_args["model"] == "streaming_gopt"
        else StreamingGOPTNoPhn
    )
    model = model_class(
        embed_dim=train_args["embed_dim"],
        num_heads=train_args["heads"],
        depth=train_args["depth"],
        input_dim=config["input_dim"],
        seq_len=config["seq_len"],
        phn_num=config["phn_num"],
    )
    state = torch.load(model_dir / "best_audio_model.pth", map_location="cpu")
    load_model_state(model, state)
    return model.to(device).eval(), config


def aggregate_words(word_pred, word_target):
    word_ids = word_target[:, 4]
    rows = []
    for word_id in sorted(set(int(value) for value in word_ids if value >= 0)):
        positions = np.where(word_ids == word_id)[0]
        pred = word_pred[positions].mean(axis=0)
        target = word_target[positions, :4].mean(axis=0)
        rows.append(
            {
                "word_id": word_id,
                "pred_accuracy": float(pred[0] * 5.0),
                "pred_stress": float(pred[1] * 5.0),
                "pred_total": float(pred[2] * 5.0),
                "pred_asr_accuracy": float(pred[3]),
                "target_accuracy": float(target[0]),
                "target_stress": float(target[1]),
                "target_total": float(target[2]),
                "target_asr_accuracy": float(target[3]),
            }
        )
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_model(args.model_dir, args.repo_src, device)
    metadata = json.loads((args.data_root / "metadata.json").read_text(encoding="utf-8"))
    norm_mean = float(metadata["train_norm_mean"])
    norm_std = float(metadata["train_norm_std"])
    archive = np.load(args.data_root / "test_chunks.npz", mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.prefix_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            source_indices = [int(row["manifest_row_index"]) for row in batch_rows]
            feat = np.asarray(archive["feat"][source_indices], dtype=np.float32).copy()
            phn_id = np.asarray(archive["phn_id"][source_indices], dtype=np.int64)
            valid = phn_id >= 0
            feat[valid] = (feat[valid] - norm_mean) / norm_std

            x = torch.from_numpy(feat).to(device)
            p = torch.from_numpy(phn_id).to(device)
            started = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    x,
                    p,
                    main_context_tokens=args.main_context_tokens,
                    right_context_tokens=args.right_context_tokens,
                )
            elapsed_per_row = (time.perf_counter() - started) / len(batch_rows)
            utt_pred = torch.cat(outputs[:5], dim=1).cpu().numpy() * 5.0
            word_pred = torch.cat(outputs[6:10], dim=2).cpu().numpy()

            for local_index, row in enumerate(batch_rows):
                source_index = source_indices[local_index]
                record = dict(row)
                record.update(
                    {
                        "model": "streaming_gopt_v6",
                        "mode": "native_streaming_chunk",
                        "status": "ok",
                        "process_time_sec": elapsed_per_row,
                        "batch_size": args.batch_size,
                        "timing_mode": (
                            "online_single_prefix"
                            if args.batch_size == 1
                            else "batch_amortized"
                        ),
                        "main_context_tokens": args.main_context_tokens,
                        "right_context_tokens": args.right_context_tokens,
                        "scores": {
                            name: float(utt_pred[local_index, index])
                            for index, name in enumerate(SCORE_NAMES)
                        },
                        "word_scores": aggregate_words(
                            word_pred[local_index],
                            np.asarray(archive["word_label"][source_index]),
                        ),
                    }
                )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            print(
                f"[streaming-gopt] {min(batch_start + len(batch_rows), len(rows))}/{len(rows)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
