import json
from pathlib import Path

paths = [
    Path("downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_closed_oracle_prefix_gt_time/predictions.jsonl"),
    Path("downloads/custom-gopt-252/eval/pcn_v2_same_manifest_streaming/gopt_open_whisper_base/predictions.phone_word_targets.jsonl"),
    Path("downloads/custom-gopt-252/eval/same_manifest_streaming/multipa/multipa.shard_0.jsonl"),
]

for path in paths:
    print("====", path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            print("keys", sorted(row.keys()))
            print("status", row.get("status"), "is_final", row.get("is_final"), "utt_id", row.get("utt_id"), "source", row.get("source_utt_id"))
            print("scores", row.get("scores"))
            print("target_scores", row.get("target_scores"))
            print("confidence", row.get("mean_asr_confidence"), row.get("matched_ratio"), row.get("coverage_ratio"), row.get("progress"))
            print("word first", (row.get("word_scores") or [None])[0])
            print("phone first", (row.get("phone_scores") or [None])[0])
            break
