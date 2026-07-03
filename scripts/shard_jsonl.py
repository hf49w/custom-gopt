import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Write one shard of a JSONL file.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    total = 0
    kept = 0
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if not line.strip():
                continue
            total += 1
            if idx % args.num_shards != args.shard_index:
                continue
            dst.write(line)
            kept += 1
    print(json.dumps({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_rows": total,
        "kept_rows": kept,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
