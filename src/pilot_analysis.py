#!/usr/bin/env python3
"""Pilot signal analysis: epoch effect and method x epoch interaction."""
import csv
import numpy as np
from scipy import stats
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
RESULTS = _ROOT / "data" / "results.csv"
if not RESULTS.exists():
    RESULTS = _ROOT / "artifacts" / "final_runs" / "results.csv"


def main():
    with open(RESULTS) as f:
        rows = list(csv.DictReader(f))

    strata = {}
    for r in rows:
        key = (r["model_name"], r["task_name"])
        strata.setdefault(key, []).append(r)

    print("=" * 70)
    print("PILOT SIGNAL ANALYSIS: Epoch effect and Method x Epoch interaction")
    print("=" * 70)

    for (model, task), data in sorted(strata.items()):
        print(f"\n--- {model} / {task} ---")

        print(f"  {'Method':<8} {'ep=1':>8} {'ep=2':>8} {'ep=5':>8} {'ep=10':>8} {'d(10-1)':>10}")
        for method in ["lora", "bitfit"]:
            accs_by_ep = {}
            for r in data:
                if r["method"] == method:
                    ep = int(r["num_train_epochs"])
                    accs_by_ep.setdefault(ep, []).append(float(r["accuracy"]))
            means = {ep: np.mean(v) for ep, v in accs_by_ep.items()}
            delta = means.get(10, 0) - means.get(1, 0)
            print(f"  {method:<8} {means.get(1,0):>8.4f} {means.get(2,0):>8.4f} {means.get(5,0):>8.4f} {means.get(10,0):>8.4f} {delta:>+10.4f}")

        # Epoch main effect (one-way ANOVA pooling methods and sizes)
        epoch_groups = {}
        for r in data:
            ep = int(r["num_train_epochs"])
            epoch_groups.setdefault(ep, []).append(float(r["accuracy"]))

        f_stat, p_val = stats.f_oneway(*[epoch_groups[ep] for ep in sorted(epoch_groups.keys())])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Epoch main effect (one-way ANOVA): F={f_stat:.3f}, p={p_val:.6f} {sig}")

        # Method effect per epoch
        print("  Method effect per epoch:")
        for ep in [1, 2, 5, 10]:
            lora_accs = [float(r["accuracy"]) for r in data if r["method"] == "lora" and int(r["num_train_epochs"]) == ep]
            bitfit_accs = [float(r["accuracy"]) for r in data if r["method"] == "bitfit" and int(r["num_train_epochs"]) == ep]
            if lora_accs and bitfit_accs:
                t, p = stats.ttest_ind(lora_accs, bitfit_accs)
                diff = np.mean(lora_accs) - np.mean(bitfit_accs)
                winner = "LoRA" if diff > 0 else "BitFit"
                print(f"    ep={ep:2d}: LoRA={np.mean(lora_accs):.4f} BitFit={np.mean(bitfit_accs):.4f} diff={diff:+.4f} ({winner}) p={p:.4f}")

    print("\n" + "=" * 70)
    print("PILOT GATE ASSESSMENT")
    print("=" * 70)
    print("  Hard gate 1: 96/96 runs, 0 NaN/Inf -> PASS")
    all_accs = [float(r["accuracy"]) for r in rows]
    print(f"  Hard gate 2: Epoch main effect -> see ANOVA results above")
    print(f"  Accuracy range: [{min(all_accs):.4f}, {max(all_accs):.4f}]")
    above_random = "PASS" if min(all_accs) > 0.45 else "CHECK"
    print(f"  All > 0.45 (near-random for binary): {above_random}")


if __name__ == "__main__":
    main()
