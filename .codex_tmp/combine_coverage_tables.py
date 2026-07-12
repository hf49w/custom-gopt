import csv
from pathlib import Path

root = Path("/DATA_2/guest/custom-gopt/exp/pcn_extra_20260704_2130")
ab_path = root / "coverage_pcc_phone_word_sentence.csv"
baseline_path = root / "baseline_coverage_pcc_phone_word_sentence.csv"
out_path = root / "coverage_pcc_phone_word_sentence_all_models.csv"

columns = ["sent_acc", "sent_comp", "sent_flu", "sent_pros", "sent_total", "word_acc", "word_stress", "word_total", "phone"]
metric_map = {
    ("sentence", "accuracy"): "sent_acc",
    ("sentence", "completeness"): "sent_comp",
    ("sentence", "fluency"): "sent_flu",
    ("sentence", "prosody"): "sent_pros",
    ("sentence", "prosodic"): "sent_pros",
    ("sentence", "total"): "sent_total",
    ("word", "accuracy"): "word_acc",
    ("word", "stress"): "word_stress",
    ("word", "total"): "word_total",
    ("phone", "phone"): "phone",
    ("phone", "accuracy"): "phone",
}

rows = {}
with ab_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        model = row.get("experiment") or row.get("model")
        coverage = int(row["coverage"])
        key = (model, coverage)
        rows.setdefault(key, {"model": model, "coverage": coverage})
        col = metric_map.get((row["level"], row["metric"]))
        if col:
            rows[key][col] = row["pcc"]

with baseline_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        model = row["model"]
        coverage = int(row["coverage"])
        key = (model, coverage)
        rows.setdefault(key, {"model": model, "coverage": coverage})
        for col in columns:
            rows[key][col] = row.get(col, "")

order = ["A_loss_dimmask", "B_relaxed_softlabel", "gopt_original", "gopt_open_base", "gopt_open_medium", "multipa"]
order_index = {name: idx for idx, name in enumerate(order)}
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["model", "coverage"] + columns)
    writer.writeheader()
    for _, row in sorted(rows.items(), key=lambda item: (order_index.get(item[0][0], 999), -item[0][1])):
        writer.writerow({field: row.get(field, "") for field in ["model", "coverage"] + columns})
print(out_path)
