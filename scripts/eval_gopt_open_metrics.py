import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch


REFERENCE_METRICS = {
    "multipa_gopt_open": {
        "utt_accuracy_pcc": 0.528,
        "utt_fluency_pcc": 0.527,
        "utt_prosodic_pcc": 0.545,
        "utt_total_pcc": 0.528,
        "word_accuracy_pcc": 0.273,
        "word_stress_pcc": 0.067,
        "word_total_pcc": 0.265,
    },
    "multipa": {
        "utt_accuracy_pcc": 0.705,
        "utt_fluency_pcc": 0.772,
        "utt_prosodic_pcc": 0.764,
        "utt_total_pcc": 0.730,
        "word_accuracy_pcc": 0.427,
        "word_stress_pcc": 0.239,
        "word_total_pcc": 0.436,
    },
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a GOPT-open prediction file with the MultiPA open-response protocol."
    )
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gt-alignment-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--multipa-repo-root", type=Path, default=None)
    parser.add_argument("--aligner", type=str, default="charsiu/en_w2v2_fc_10ms")
    parser.add_argument("--align-device", type=str, default=None)
    parser.add_argument("--ensure-gt-alignments", action="store_true")
    parser.add_argument(
        "--compare-reference",
        type=str,
        default="multipa_gopt_open",
        choices=["multipa_gopt_open", "multipa"],
    )
    return parser.parse_args()


def read_wav_scp(dataset_root: Path, split: str):
    wav_scp = dataset_root / split / "wav.scp"
    entries = []
    with wav_scp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path = line.split("\t", maxsplit=1)
            entries.append((utt_id, Path(dataset_root / wav_path if not Path(wav_path).is_absolute() else wav_path)))
    return entries


def build_test_data(dataset_root: Path):
    with (dataset_root / "resource" / "scores.json").open("r", encoding="utf-8") as f:
        scores = json.load(f)
    test_data = {}
    for utt_id, _ in read_wav_scp(dataset_root, "test"):
        test_data[utt_id] = scores[utt_id]
    return test_data


def parse_prediction_file(path: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict], int]:
    invalid = 0
    result_word: Dict[str, Dict] = {}
    result_uttr: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    for sample in rows:
        parts = sample.split(";")
        wavidx = parts[0].replace(".wav", "")
        valid = parts[5].split(":", maxsplit=1)[1]

        if valid == "F":
            invalid += 1
            accuracy = 1.0
            fluency = 0.0
            prosodic = 0.0
            total = 0.0
            result_word[wavidx] = {
                "word_accuracy": 0,
                "word_stress": 5,
                "word_total": 1,
                "text": "",
                "alignment": None,
            }
        else:
            accuracy = float(parts[1].split(":", maxsplit=1)[1])
            fluency = float(parts[2].split(":", maxsplit=1)[1])
            prosodic = float(parts[3].split(":", maxsplit=1)[1])
            total = float(parts[4].split(":", maxsplit=1)[1])
            w_a = ast.literal_eval(parts[8].split(":", maxsplit=1)[1])
            w_s = ast.literal_eval(parts[9].split(":", maxsplit=1)[1])
            w_t = ast.literal_eval(parts[10].split(":", maxsplit=1)[1])
            if isinstance(w_a, float):
                w_a = [w_a]
                w_s = [w_s]
                w_t = [w_t]
            alignment = ast.literal_eval(parts[-1].split(":", maxsplit=1)[1])
            result_word[wavidx] = {
                "word_accuracy": [10 if x > 10 else x for x in w_a],
                "word_stress": [10 if x > 10 else x for x in w_s],
                "word_total": [10 if x > 10 else x for x in w_t],
                "text": [word[-1] for word in alignment],
                "alignment": alignment,
            }

        result_uttr[wavidx] = {
            "accuracy": accuracy,
            "fluency": fluency,
            "prosodic": prosodic,
            "total": total,
        }
    return result_word, result_uttr, invalid


def pad_mismatch_sequence(gt_words, pred_words, pred_acc, pred_stress, pred_total):
    padded_acc, padded_stress, padded_total = [], [], []
    asr_w_idx = 0
    for gt_word in gt_words:
        if asr_w_idx >= len(pred_words):
            padded_acc.append(pred_acc[asr_w_idx - 1])
            padded_stress.append(pred_stress[asr_w_idx - 1])
            padded_total.append(pred_total[asr_w_idx - 1])
            break
        if gt_word == pred_words[asr_w_idx]:
            padded_acc.append(pred_acc[asr_w_idx])
            padded_stress.append(pred_stress[asr_w_idx])
            padded_total.append(pred_total[asr_w_idx])
            asr_w_idx += 1
        else:
            padded_acc.append(pred_acc[asr_w_idx - 1])
            padded_stress.append(pred_stress[asr_w_idx - 1])
            padded_total.append(pred_total[asr_w_idx - 1])
    return padded_acc, padded_stress, padded_total


def align_two_sentences(result_gt, result_asr):
    asr_wordidx_list = []
    for gt_value in result_gt:
        gt_start = gt_value[0]
        gt_end = gt_value[1]
        asr_wordidx = []
        for asr_idx, asr_value in enumerate(result_asr):
            asr_start = asr_value[0]
            asr_end = asr_value[1]
            if gt_end <= asr_start:
                break
            if gt_start >= asr_end:
                continue
            if max(gt_start, asr_start) <= min(gt_end, asr_end):
                asr_wordidx.append(asr_idx)
        asr_wordidx_list.append(asr_wordidx)
    return asr_wordidx_list


def pearson_corr(pred: List[float], gt: List[float]) -> Optional[float]:
    if len(pred) < 2 or len(gt) < 2:
        return None
    if len(set(pred)) < 2 or len(set(gt)) < 2:
        return None
    try:
        corr, _ = scipy.stats.pearsonr(pred, gt)
    except ValueError:
        return None
    return float(corr)


def evaluate_predictions(prediction_path: Path, test_data: Dict[str, Dict], gt_alignment_dir: Path):
    result_word, result_uttr, invalid = parse_prediction_file(prediction_path)
    wav_idx_word = list(result_word.keys())
    wav_idx_uttr = list(result_uttr.keys())

    gt_A, gt_F, gt_P, gt_T = [], [], [], []
    pred_A, pred_F, pred_P, pred_T = [], [], [], []
    for wavidx in wav_idx_uttr:
        gt_A.append(test_data[wavidx]["accuracy"])
        pred_A.append(result_uttr[wavidx]["accuracy"])
        gt_F.append(test_data[wavidx]["fluency"])
        pred_F.append(result_uttr[wavidx]["fluency"])
        gt_P.append(test_data[wavidx]["prosodic"])
        pred_P.append(result_uttr[wavidx]["prosodic"])
        gt_T.append(test_data[wavidx]["total"])
        pred_T.append(result_uttr[wavidx]["total"])

    gt_w_acc, gt_w_stress, gt_w_total = [], [], []
    pred_w_acc, pred_w_stress, pred_w_total = [], [], []
    count_sen = 0
    skipped_word_eval_missing_gt_alignment = []

    for wavidx in wav_idx_word:
        the_gt_w_acc, the_gt_w_stress, the_gt_w_total, the_gt_w_text = [], [], [], []
        for word in test_data[wavidx]["words"]:
            the_gt_w_acc.append(int(word["accuracy"]))
            the_gt_w_stress.append(int(word["stress"]))
            the_gt_w_total.append(int(word["total"]))
            the_gt_w_text.append(word["text"].lower())

        the_pred_w_acc = result_word[wavidx]["word_accuracy"]
        the_pred_w_stress = result_word[wavidx]["word_stress"]
        the_pred_w_total = result_word[wavidx]["word_total"]

        if result_word[wavidx]["alignment"] is None:
            gt_len = len(the_gt_w_text)
            the_pred_w_acc = [the_pred_w_acc for _ in range(gt_len)]
            the_pred_w_stress = [the_pred_w_stress for _ in range(gt_len)]
            the_pred_w_total = [the_pred_w_total for _ in range(gt_len)]
            gt_w_acc.extend(the_gt_w_acc)
            gt_w_stress.extend(the_gt_w_stress)
            gt_w_total.extend(the_gt_w_total)
            pred_w_acc.extend(the_pred_w_acc)
            pred_w_stress.extend(the_pred_w_stress)
            pred_w_total.extend(the_pred_w_total)
            continue

        pred_sen = " ".join(result_word[wavidx]["text"])
        gt_sen = " ".join(the_gt_w_text)
        if pred_sen != gt_sen:
            gt_alignment_path = gt_alignment_dir / f"{wavidx}.pt"
            if not gt_alignment_path.exists():
                skipped_word_eval_missing_gt_alignment.append(wavidx)
                continue
            gt_alignment = torch.load(gt_alignment_path)
            align_result = align_two_sentences(result_word[wavidx]["alignment"], gt_alignment)
            the_gt_w_acc = np.asarray(the_gt_w_acc)
            the_gt_w_stress = np.asarray(the_gt_w_stress)
            the_gt_w_total = np.asarray(the_gt_w_total)
            align_gt_w_acc, align_gt_w_stress, align_gt_w_total = [], [], []
            for widxs in align_result:
                if len(widxs) != 0:
                    align_gt_w_acc.append(float(np.mean(the_gt_w_acc[widxs])))
                    align_gt_w_stress.append(float(np.mean(the_gt_w_stress[widxs])))
                    align_gt_w_total.append(float(np.mean(the_gt_w_total[widxs])))
                else:
                    align_gt_w_acc.append(0.0)
                    align_gt_w_stress.append(5.0)
                    align_gt_w_total.append(1.0)
            gt_w_acc.extend(align_gt_w_acc)
            gt_w_stress.extend(align_gt_w_stress)
            gt_w_total.extend(align_gt_w_total)
            pred_w_acc.extend(the_pred_w_acc)
            pred_w_stress.extend(the_pred_w_stress)
            pred_w_total.extend(the_pred_w_total)
        else:
            if len(the_gt_w_text) != len(result_word[wavidx]["text"]):
                the_pred_w_acc, the_pred_w_stress, the_pred_w_total = pad_mismatch_sequence(
                    the_gt_w_text, result_word[wavidx]["text"], the_pred_w_acc, the_pred_w_stress, the_pred_w_total
                )
            gt_w_acc.extend(the_gt_w_acc)
            gt_w_stress.extend(the_gt_w_stress)
            gt_w_total.extend(the_gt_w_total)
            pred_w_acc.extend(the_pred_w_acc)
            pred_w_stress.extend(the_pred_w_stress)
            pred_w_total.extend(the_pred_w_total)
        count_sen += 1

    return {
        "utterance_count": len(pred_A),
        "invalid_sentence_count": invalid,
        "word_eval_sentence_count": count_sen,
        "word_eval_word_count": len(pred_w_acc),
        "skipped_word_eval_missing_gt_alignment_count": len(skipped_word_eval_missing_gt_alignment),
        "skipped_word_eval_missing_gt_alignment_examples": skipped_word_eval_missing_gt_alignment[:20],
        "utterance_pcc": {
            "accuracy": pearson_corr(pred_A, gt_A),
            "fluency": pearson_corr(pred_F, gt_F),
            "prosodic": pearson_corr(pred_P, gt_P),
            "total": pearson_corr(pred_T, gt_T),
        },
        "word_pcc": {
            "accuracy": pearson_corr(pred_w_acc, gt_w_acc),
            "stress": pearson_corr(pred_w_stress, gt_w_stress),
            "total": pearson_corr(pred_w_total, gt_w_total),
        },
    }


def load_charsiu(multipa_repo_root: Path, aligner_name: str, align_device: Optional[str]):
    sys.path.insert(0, str(multipa_repo_root))
    from Charsiu import charsiu_forced_aligner
    from utils_assessment import get_match_index

    kwargs = {"aligner": aligner_name}
    if align_device:
        kwargs["device"] = align_device
    return charsiu_forced_aligner(**kwargs), get_match_index


def ensure_gt_alignments(args, test_data: Dict[str, Dict]):
    if args.multipa_repo_root is None:
        raise ValueError("--multipa-repo-root is required when --ensure-gt-alignments is set.")
    aligner, get_match_index = load_charsiu(args.multipa_repo_root, args.aligner, args.align_device)
    prediction_word, _, _ = parse_prediction_file(args.prediction_path)
    wav_map = dict(read_wav_scp(args.dataset_root, "test"))
    args.gt_alignment_dir.mkdir(parents=True, exist_ok=True)

    for utt_id in prediction_word.keys():
        out_path = args.gt_alignment_dir / f"{utt_id}.pt"
        if out_path.exists():
            continue
        wav_path = wav_map[utt_id]
        gt_text = " ".join(word["text"].lower() for word in test_data[utt_id]["words"])
        pred_phones, pred_words, words, pred_prob, phone_ids, word_phone_map = aligner.align(audio=str(wav_path), text=gt_text)
        selected_idx = get_match_index(pred_words, words)
        pred_words = np.asarray(pred_words)
        pred_words = pred_words[selected_idx]
        torch.save(pred_words, out_path)


def main():
    args = get_args()
    test_data = build_test_data(args.dataset_root)
    if args.ensure_gt_alignments:
        ensure_gt_alignments(args, test_data)
    metrics = evaluate_predictions(args.prediction_path, test_data, args.gt_alignment_dir)
    reference = REFERENCE_METRICS[args.compare_reference]
    summary = {
        "prediction_path": str(args.prediction_path),
        "dataset_root": str(args.dataset_root),
        "gt_alignment_dir": str(args.gt_alignment_dir),
        "metrics": metrics,
        "reference_name": args.compare_reference,
        "reference_metrics": reference,
        "delta_vs_reference": {
            "utt_accuracy_pcc": None if metrics["utterance_pcc"]["accuracy"] is None else metrics["utterance_pcc"]["accuracy"] - reference["utt_accuracy_pcc"],
            "utt_fluency_pcc": None if metrics["utterance_pcc"]["fluency"] is None else metrics["utterance_pcc"]["fluency"] - reference["utt_fluency_pcc"],
            "utt_prosodic_pcc": None if metrics["utterance_pcc"]["prosodic"] is None else metrics["utterance_pcc"]["prosodic"] - reference["utt_prosodic_pcc"],
            "utt_total_pcc": None if metrics["utterance_pcc"]["total"] is None else metrics["utterance_pcc"]["total"] - reference["utt_total_pcc"],
            "word_accuracy_pcc": None if metrics["word_pcc"]["accuracy"] is None else metrics["word_pcc"]["accuracy"] - reference["word_accuracy_pcc"],
            "word_stress_pcc": None if metrics["word_pcc"]["stress"] is None else metrics["word_pcc"]["stress"] - reference["word_stress_pcc"],
            "word_total_pcc": None if metrics["word_pcc"]["total"] is None else metrics["word_pcc"]["total"] - reference["word_total_pcc"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
