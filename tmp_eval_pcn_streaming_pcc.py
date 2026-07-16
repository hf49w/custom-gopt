import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/DATA_2/guest/custom-gopt")
DATA_DIR = ROOT / "data/streaming_pcn_gopt_v2_stateful"
EXP_DIR = ROOT / "exp/streaming-pcn-gopt-v2-stateful-teacher-state"

sys.path.insert(0, str(ROOT / "src"))

from models import PCNStreamingScorer  # noqa: E402
from train_streaming_pcn import (  # noqa: E402
    PCNUtteranceDataset,
    load_state,
    move_batch,
    pcn_utterance_collate,
    pcc,
    reset_state_where_needed,
    restore_invalid_state,
    slice_chunk,
    valid_slot_mask,
)


PHONE_NAMES = ["accuracy"]
WORD_NAMES = ["accuracy", "stress", "total"]
UTT_NAMES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def as_float_array(values):
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def metric_block(pred_values, target_values, names):
    pred = as_float_array(pred_values)
    target = as_float_array(target_values)
    if pred.ndim == 1 and len(names) == 1:
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
    out = {"n": int(pred.shape[0]) if pred.size else 0, "pcc": {}, "mae": {}}
    for idx, name in enumerate(names):
        if pred.size == 0:
            out["pcc"][name] = None
            out["mae"][name] = None
            continue
        x = pred[:, idx]
        y = target[:, idx]
        out["pcc"][name] = pcc(x, y)
        out["mae"][name] = float(np.mean(np.abs(x - y)))
    return out


def mean_or_none(values):
    arr = as_float_array(values)
    return float(np.mean(arr)) if arr.size else None


def main():
    config = json.loads((EXP_DIR / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    args = config["args"]
    prosody_mean = np.asarray(config["prosody_norm_mean"], dtype=np.float32)
    prosody_std = np.asarray(config["prosody_norm_std"], dtype=np.float32)

    train_raw = np.load(DATA_DIR / "train_chunks.npz")
    has_teacher_state = "teacher_state_embedding" in train_raw.files
    teacher_state_dim = int(train_raw["teacher_state_embedding"].shape[-1]) if has_teacher_state else 128

    model = PCNStreamingScorer(
        phone_dim=int(metadata["phone_dim"]),
        seq_len=int(metadata["seq_len"]),
        prosody_dim=int(config["prosody_dim"]),
        embed_dim=int(args["embed_dim"]),
        num_heads=int(args["heads"]),
        depth=int(args["depth"]),
        gru_dim=int(args["gru_dim"]),
        main_context_tokens=int(args["main_context_tokens"]),
        use_state_projection=bool(has_teacher_state and args.get("loss_w_state_projection", 0) > 0),
        teacher_state_dim=teacher_state_dim,
    )
    state_dict = torch.load(EXP_DIR / "models/best_audio_model.pth", map_location="cpu")
    load_state(model, state_dict)

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = PCNUtteranceDataset("test", DATA_DIR, prosody_mean, prosody_std)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=pcn_utterance_collate,
        pin_memory=torch.cuda.is_available(),
    )

    all_utt_pred = []
    all_utt_target = []
    all_coverage = []
    final_utt_pred = []
    final_utt_target = []
    final_coverage = []

    all_word_pred = []
    all_word_target = []
    final_word_pred = []
    final_word_target = []

    all_phone_pred = []
    all_phone_target = []
    final_phone_pred = []
    final_phone_target = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            state = None
            max_chunks = batch["chunk_valid_mask"].shape[1]
            for chunk_idx in range(max_chunks):
                cur_valid = batch["chunk_valid_mask"][:, chunk_idx]
                if cur_valid.sum().item() <= 0:
                    continue

                chunk = slice_chunk(batch, chunk_idx)
                state = reset_state_where_needed(state, chunk["state_reset"])
                out = model(
                    cn_post=chunk["cn_post"],
                    cn_stats=chunk["cn_stats"],
                    acoustic_post=chunk["acoustic_post"],
                    acoustic_stats=chunk["acoustic_stats"],
                    prosody=chunk["prosody"],
                    visible_len=chunk["visible_len"],
                    cumulative_commit_mask=chunk["cumulative_commit_mask"],
                    new_commit_mask=chunk["new_commit_mask"],
                    word_ids=chunk["pcn_word_id"],
                    prev_state=state,
                    detach_next_state=True,
                )
                state = restore_invalid_state(out["next_state"], state, cur_valid)

                valid_rows = cur_valid > 0
                all_utt_pred.extend(out["utt_scores"][valid_rows].detach().cpu().tolist())
                all_utt_target.extend(chunk["utt_target"][valid_rows].detach().cpu().tolist())
                all_coverage.extend(chunk["coverage_ratio"][valid_rows].detach().cpu().tolist())

                final_rows = (chunk["is_final"] > 0) & valid_rows
                if final_rows.any():
                    final_utt_pred.extend(out["utt_scores"][final_rows].detach().cpu().tolist())
                    final_utt_target.extend(chunk["utt_target"][final_rows].detach().cpu().tolist())
                    final_coverage.extend(chunk["coverage_ratio"][final_rows].detach().cpu().tolist())

                slot_valid = valid_slot_mask(chunk) * cur_valid.unsqueeze(-1)
                supervise = chunk["soft_label_weight"] * chunk["cumulative_commit_mask"] * slot_valid
                if supervise.sum().item() > 0:
                    mask = supervise > 0
                    all_phone_pred.extend(out["phone_score"].squeeze(-1)[mask].detach().cpu().tolist())
                    all_phone_target.extend(chunk["phone_score_target"][mask].detach().cpu().tolist())
                    all_word_pred.extend(out["word_scores"][mask].detach().cpu().tolist())
                    all_word_target.extend(chunk["word_score_target"][mask].detach().cpu().tolist())

                    final_slot_mask = mask & final_rows.unsqueeze(-1)
                    if final_slot_mask.sum().item() > 0:
                        final_phone_pred.extend(out["phone_score"].squeeze(-1)[final_slot_mask].detach().cpu().tolist())
                        final_phone_target.extend(chunk["phone_score_target"][final_slot_mask].detach().cpu().tolist())
                        final_word_pred.extend(out["word_scores"][final_slot_mask].detach().cpu().tolist())
                        final_word_target.extend(chunk["word_score_target"][final_slot_mask].detach().cpu().tolist())

    all_utt_pred_arr = as_float_array(all_utt_pred)
    all_utt_target_arr = as_float_array(all_utt_target)
    all_coverage_arr = as_float_array(all_coverage)

    result = {
        "checkpoint": str(EXP_DIR / "models/best_audio_model.pth"),
        "data_dir": str(DATA_DIR),
        "scale": "targets and predictions are normalized to 0-1; PCC is scale-invariant",
        "all_streaming_chunks": {
            "utterance": metric_block(all_utt_pred, all_utt_target, UTT_NAMES),
            "word": metric_block(all_word_pred, all_word_target, WORD_NAMES),
            "phone": metric_block(all_phone_pred, all_phone_target, PHONE_NAMES),
            "mean_coverage_ratio": mean_or_none(all_coverage),
        },
        "final_chunks_full_utterance": {
            "utterance": metric_block(final_utt_pred, final_utt_target, UTT_NAMES),
            "word": metric_block(final_word_pred, final_word_target, WORD_NAMES),
            "phone": metric_block(final_phone_pred, final_phone_target, PHONE_NAMES),
            "mean_coverage_ratio": mean_or_none(final_coverage),
        },
        "streaming_utterance_by_coverage_threshold": {},
    }

    for threshold in [0.25, 0.5, 0.75, 0.9]:
        mask = all_coverage_arr >= threshold
        result["streaming_utterance_by_coverage_threshold"][str(threshold)] = {
            "n": int(mask.sum()),
            "mean_coverage_ratio": float(np.mean(all_coverage_arr[mask])) if mask.any() else None,
            "utterance": metric_block(all_utt_pred_arr[mask], all_utt_target_arr[mask], UTT_NAMES),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
