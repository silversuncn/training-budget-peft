#!/usr/bin/env python3
"""Aggregate per-run metrics.json into results.csv"""
import csv
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "final_runs"
OUTPUT_CSV = ARTIFACTS / "results.csv"
OUTPUT_JSON = ARTIFACTS / "results.json"

FIELDS = [
    "method", "model_name", "task_name", "train_subset_size",
    "num_train_epochs", "seed", "accuracy", "learning_rate",
    "batch_size", "max_length", "lora_r", "lora_alpha",
    "warmup_ratio", "bf16", "elapsed_seconds",
    "total_parameters", "trainable_parameters", "trainable_percentage",
]


def main():
    rows = []
    for metrics_file in sorted(ARTIFACTS.glob("*/metrics.json")):
        try:
            data = json.loads(metrics_file.read_text())
            row = {f: data.get(f) for f in FIELDS}
            rows.append(row)
        except Exception as e:
            print(f"WARN: skipping {metrics_file}: {e}")

    rows.sort(key=lambda r: (r["model_name"], r["task_name"], r["method"],
                             r["num_train_epochs"], r["train_subset_size"], r["seed"]))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"Aggregated {len(rows)} runs -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
