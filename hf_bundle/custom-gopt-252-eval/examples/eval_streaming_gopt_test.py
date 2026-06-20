import argparse
import json
import sys
from pathlib import Path

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the bundled Streaming GOPT checkpoint on a prepared val/test split."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing *_chunks.npz and metadata.json.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing best_audio_model.pth and config.json.")
    parser.add_argument("--repo-src", type=Path, required=True, help="Path to custom-gopt/src.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu. Defaults to cuda if available.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--main-context-tokens", type=int, default=None, help="Override evaluation main context tokens.")
    parser.add_argument("--right-context-tokens", type=int, default=None, help="Override evaluation right context tokens.")
    return parser.parse_args()


def load_summary(data_dir, model_dir, repo_src, split, device_name, main_context_tokens=None, right_context_tokens=None):
    sys.path.insert(0, str(repo_src))

    from models import StreamingGOPT, StreamingGOPTNoPhn
    from train_streaming_charsiu import StreamingChunkDataset, load_model_state, make_loader, validate

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    args_dict = cfg["args"]

    class Args:
        pass

    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    args.main_context_token_choices = args_dict.get("main_context_token_choices") or [
        int(item.strip()) for item in str(args_dict["main_context_tokens"]).split(",") if item.strip()
    ]
    args.right_context_token_choices = args_dict.get("right_context_token_choices") or [
        int(item.strip()) for item in str(args_dict["right_context_tokens"]).split(",") if item.strip()
    ]

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if getattr(args, "tf32", False) and device.type == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = StreamingChunkDataset(split, data_dir, metadata, final_only=(split != "train"))
    loader = make_loader(dataset, len(dataset), False, int(getattr(args, "num_workers", 0)))

    model_cls = StreamingGOPT if args_dict["model"] == "streaming_gopt" else StreamingGOPTNoPhn
    model = model_cls(
        embed_dim=int(args_dict["embed_dim"]),
        num_heads=int(args_dict["heads"]),
        depth=int(args_dict["depth"]),
        input_dim=int(cfg["input_dim"]),
        seq_len=int(cfg["seq_len"]),
        phn_num=int(cfg["phn_num"]),
    )

    state = torch.load(model_dir / "best_audio_model.pth", map_location=device)
    load_model_state(model, state)
    model = model.to(device)
    model.eval()

    eval_main_context = int(main_context_tokens if main_context_tokens is not None else max(args.main_context_token_choices))
    eval_right_context = int(right_context_tokens if right_context_tokens is not None else max(args.right_context_token_choices))

    mse, corr, utt_mse, utt_corr, word_mse, word_corr = validate(
        model,
        loader,
        args,
        -1,
        device,
        eval_main_context,
        eval_right_context,
    )

    return {
        "split": split,
        "device": str(device),
        "main_context_tokens": eval_main_context,
        "right_context_tokens": eval_right_context,
        "phone_mse": float(mse),
        "phone_pcc": float(corr),
        "utt_mse": [float(x) for x in utt_mse],
        "utt_pcc": [float(x) for x in utt_corr],
        "word_mse": [float(x) for x in word_mse],
        "word_pcc": [float(x) for x in word_corr],
    }


def main():
    args = get_args()
    summary = load_summary(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        repo_src=args.repo_src,
        split=args.split,
        device_name=args.device,
        main_context_tokens=args.main_context_tokens,
        right_context_tokens=args.right_context_tokens,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
