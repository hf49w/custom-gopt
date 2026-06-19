import argparse
from pathlib import Path

from huggingface_hub import HfApi


def get_args():
    parser = argparse.ArgumentParser(description="Upload the custom-gopt HF bundle.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "hf_bundle" / "custom-gopt-252-eval",
    )
    parser.add_argument("--repo-id", default="faeea/custom-gopt-252-eval")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument(
        "--commit-message",
        default="Update Streaming GOPT checkpoint to v6 ASR-confidence model",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not args.folder.exists():
        raise FileNotFoundError(args.folder)

    api = HfApi()
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
    )
    print(f"uploaded {args.folder} to {args.repo_id}")


if __name__ == "__main__":
    main()
