#!/usr/bin/env python3
"""
Training Budget Sensitivity — Statistical Analysis (v2)
Fixes:
  1. Proper two-way ANOVA with interaction: accuracy ~ C(method)*C(epoch) + C(sample_size)
  2. NaN-safe pairwise tests with degenerate case handling
  3. Corrected majority baselines from GLUE validation splits
"""
import csv
import json
import warnings
import numpy as np
from itertools import combinations
from pathlib import Path
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "data" / "results.csv"
if not RESULTS_CSV.exists():
    RESULTS_CSV = ROOT / "artifacts" / "final_runs" / "results.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_JSON = OUTPUT_DIR / "statistical_analysis.json"

PEFT_METHODS = ["lora", "bitfit", "ia3"]
ALL_METHODS = ["full_ft", "lora", "bitfit", "ia3"]
EPOCHS = [1, 2, 5, 10]
SAMPLE_SIZES = [80, 320, 1280]
SEEDS = [11, 17, 23]
MODELS = ["bert-base-uncased", "roberta-base"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]

# Majority baselines from GLUE validation splits
# SST-2 val: 872 samples, 444 positive (0.5092)
# MRPC val: 408 samples, 279 positive/equivalent (0.6838)
# QNLI val: 5463 samples, 2761 entailment (0.5054)
# RTE val: 277 samples, 146 entailment (0.5271)
MAJORITY_BASELINES_VAL = {"sst2": 0.5092, "mrpc": 0.6838, "qnli": 0.5054, "rte": 0.5271}
UNIFORM_RANDOM = 0.5


def load_data():
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["accuracy"] = df["accuracy"].astype(float)
    df["num_train_epochs"] = df["num_train_epochs"].astype(int)
    df["train_subset_size"] = df["train_subset_size"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df


def run_anova_per_stratum(df):
    """Proper OLS ANOVA: accuracy ~ C(method)*C(epoch) + C(sample_size), PEFT only."""
    results = {}
    for model in MODELS:
        for task in TASKS:
            stratum_key = f"{model}__{task}"
            sub = df[(df["model_name"] == model) & (df["task_name"] == task) &
                     (df["method"].isin(PEFT_METHODS))].copy()

            if len(sub) < 10:
                results[stratum_key] = {"error": "insufficient data"}
                continue

            sub["method"] = sub["method"].astype(str)
            sub["epoch"] = sub["num_train_epochs"].astype(str)
            sub["size"] = sub["train_subset_size"].astype(str)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_ols = ols("accuracy ~ C(method) * C(epoch) + C(size)", data=sub).fit()
                    anova_table = anova_lm(model_ols, typ=2)

                    res = {}
                    for term in anova_table.index:
                        if term == "Residual":
                            res["Residual"] = {"df": int(anova_table.loc[term, "df"]),
                                               "sum_sq": round(float(anova_table.loc[term, "sum_sq"]), 8)}
                        else:
                            clean_term = term.replace("C(method)", "method").replace("C(epoch)", "epoch").replace("C(size)", "sample_size").replace(":", "x")
                            res[clean_term] = {
                                "df": int(anova_table.loc[term, "df"]),
                                "sum_sq": round(float(anova_table.loc[term, "sum_sq"]), 8),
                                "F": round(float(anova_table.loc[term, "F"]), 4),
                                "p": round(float(anova_table.loc[term, "PR(>F)"]), 8),
                            }
                    results[stratum_key] = res
                except Exception as e:
                    results[stratum_key] = {"error": str(e)}

    return results


def get_accuracy(df, method, model, task, epoch, size, seed):
    row = df[(df["method"] == method) & (df["model_name"] == model) &
             (df["task_name"] == task) & (df["num_train_epochs"] == epoch) &
             (df["train_subset_size"] == size) & (df["seed"] == seed)]
    if len(row) == 1:
        return float(row["accuracy"].iloc[0])
    return None


def paired_bootstrap_ci(diffs, n_boot=10000, ci=0.95, seed=42):
    if len(diffs) == 0 or np.std(diffs) == 0:
        m = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def cohens_dz(diffs):
    if len(diffs) < 2:
        return 0.0
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diffs) / sd)


def bh_correction(pvalues):
    """BH-FDR correction. Input must not contain NaN."""
    n = len(pvalues)
    if n == 0:
        return []
    pv = np.array(pvalues, dtype=float)
    assert not np.any(np.isnan(pv)), "BH input contains NaN"
    sorted_idx = np.argsort(pv)
    sorted_p = pv[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[n - 1] = sorted_p[n - 1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


def run_pairwise_method_tests(df):
    """Seed-paired t-tests for PEFT method pairs, per stratum per epoch."""
    all_tests = []
    for model in MODELS:
        for task in TASKS:
            for epoch in EPOCHS:
                for m1, m2 in combinations(PEFT_METHODS, 2):
                    diffs = []
                    for size in SAMPLE_SIZES:
                        for seed in SEEDS:
                            a1 = get_accuracy(df, m1, model, task, epoch, size, seed)
                            a2 = get_accuracy(df, m2, model, task, epoch, size, seed)
                            if a1 is not None and a2 is not None:
                                diffs.append(a1 - a2)

                    diffs = np.array(diffs)
                    n_pairs = len(diffs)

                    if n_pairs < 2:
                        all_tests.append({
                            "family": "method_pairwise",
                            "model": model, "task": task, "epoch": epoch,
                            "method_1": m1, "method_2": m2,
                            "status": "degenerate_skipped",
                            "reason": "insufficient pairs",
                            "n_pairs": n_pairs,
                            "mean_diff": 0.0, "t_stat": 0.0, "p_value": 1.0,
                            "cohens_dz": 0.0, "ci_95_lo": 0.0, "ci_95_hi": 0.0,
                        })
                        continue

                    if np.std(diffs, ddof=1) == 0:
                        all_tests.append({
                            "family": "method_pairwise",
                            "model": model, "task": task, "epoch": epoch,
                            "method_1": m1, "method_2": m2,
                            "status": "degenerate_zero_variance",
                            "n_pairs": n_pairs,
                            "mean_diff": round(float(np.mean(diffs)), 6),
                            "t_stat": 0.0, "p_value": 1.0,
                            "cohens_dz": 0.0, "ci_95_lo": round(float(np.mean(diffs)), 6),
                            "ci_95_hi": round(float(np.mean(diffs)), 6),
                        })
                        continue

                    t, p = stats.ttest_1samp(diffs, 0)
                    d = cohens_dz(diffs)
                    ci_lo, ci_hi = paired_bootstrap_ci(diffs)
                    all_tests.append({
                        "family": "method_pairwise",
                        "model": model, "task": task, "epoch": epoch,
                        "method_1": m1, "method_2": m2,
                        "status": "valid",
                        "n_pairs": n_pairs,
                        "mean_diff": round(float(np.mean(diffs)), 6),
                        "t_stat": round(float(t), 4),
                        "p_value": round(float(p), 6),
                        "cohens_dz": round(d, 4),
                        "ci_95_lo": round(ci_lo, 6),
                        "ci_95_hi": round(ci_hi, 6),
                    })

    # BH correction on valid tests only
    valid_idx = [i for i, t in enumerate(all_tests) if t.get("status") == "valid"]
    valid_pvals = [all_tests[i]["p_value"] for i in valid_idx]

    if valid_pvals:
        adjusted = bh_correction(valid_pvals)
        for k, i in enumerate(valid_idx):
            all_tests[i]["p_adjusted"] = round(adjusted[k], 6)
            all_tests[i]["significant_bh"] = adjusted[k] < 0.05

    # Mark degenerate tests
    for t in all_tests:
        if t.get("status") != "valid":
            t["p_adjusted"] = 1.0
            t["significant_bh"] = False

    return all_tests


def run_pairwise_epoch_tests(df):
    """Seed-paired t-tests for epoch pairs, per stratum per PEFT method."""
    all_tests = []
    for model in MODELS:
        for task in TASKS:
            for method in PEFT_METHODS:
                for ep1, ep2 in combinations(EPOCHS, 2):
                    diffs = []
                    for size in SAMPLE_SIZES:
                        for seed in SEEDS:
                            a1 = get_accuracy(df, method, model, task, ep1, size, seed)
                            a2 = get_accuracy(df, method, model, task, ep2, size, seed)
                            if a1 is not None and a2 is not None:
                                diffs.append(a2 - a1)

                    diffs = np.array(diffs)
                    n_pairs = len(diffs)

                    if n_pairs < 2:
                        all_tests.append({
                            "family": "epoch_pairwise",
                            "model": model, "task": task, "method": method,
                            "epoch_1": ep1, "epoch_2": ep2,
                            "status": "degenerate_skipped",
                            "reason": "insufficient pairs",
                            "n_pairs": n_pairs,
                            "mean_diff": 0.0, "t_stat": 0.0, "p_value": 1.0,
                            "cohens_dz": 0.0, "ci_95_lo": 0.0, "ci_95_hi": 0.0,
                        })
                        continue

                    if np.std(diffs, ddof=1) == 0:
                        all_tests.append({
                            "family": "epoch_pairwise",
                            "model": model, "task": task, "method": method,
                            "epoch_1": ep1, "epoch_2": ep2,
                            "status": "degenerate_zero_variance",
                            "n_pairs": n_pairs,
                            "mean_diff": round(float(np.mean(diffs)), 6),
                            "t_stat": 0.0, "p_value": 1.0,
                            "cohens_dz": 0.0, "ci_95_lo": round(float(np.mean(diffs)), 6),
                            "ci_95_hi": round(float(np.mean(diffs)), 6),
                        })
                        continue

                    t, p = stats.ttest_1samp(diffs, 0)
                    d = cohens_dz(diffs)
                    ci_lo, ci_hi = paired_bootstrap_ci(diffs)
                    all_tests.append({
                        "family": "epoch_pairwise",
                        "model": model, "task": task, "method": method,
                        "epoch_1": ep1, "epoch_2": ep2,
                        "status": "valid",
                        "n_pairs": n_pairs,
                        "mean_diff": round(float(np.mean(diffs)), 6),
                        "t_stat": round(float(t), 4),
                        "p_value": round(float(p), 6),
                        "cohens_dz": round(d, 4),
                        "ci_95_lo": round(ci_lo, 6),
                        "ci_95_hi": round(ci_hi, 6),
                    })

    valid_idx = [i for i, t in enumerate(all_tests) if t.get("status") == "valid"]
    valid_pvals = [all_tests[i]["p_value"] for i in valid_idx]

    if valid_pvals:
        adjusted = bh_correction(valid_pvals)
        for k, i in enumerate(valid_idx):
            all_tests[i]["p_adjusted"] = round(adjusted[k], 6)
            all_tests[i]["significant_bh"] = adjusted[k] < 0.05

    for t in all_tests:
        if t.get("status") != "valid":
            t["p_adjusted"] = 1.0
            t["significant_bh"] = False

    return all_tests


def compute_descriptive_stats(df):
    """Mean accuracy per method x model x task x epoch."""
    desc = {}
    for model in MODELS:
        for task in TASKS:
            for method in ALL_METHODS:
                for epoch in EPOCHS:
                    sub = df[(df["method"] == method) & (df["model_name"] == model) &
                             (df["task_name"] == task) & (df["num_train_epochs"] == epoch)]
                    if len(sub) > 0:
                        accs = sub["accuracy"].values
                        key = f"{method}__{model}__{task}__ep{epoch}"
                        desc[key] = {
                            "mean": round(float(np.mean(accs)), 6),
                            "std": round(float(np.std(accs, ddof=1)), 6) if len(accs) > 1 else 0.0,
                            "min": round(float(np.min(accs)), 6),
                            "max": round(float(np.max(accs)), 6),
                            "n": int(len(accs)),
                        }
    return desc


def boundary_condition_analysis(df):
    """Identify below-baseline runs with corrected majority baselines."""
    boundary_uniform = df[df["accuracy"] < UNIFORM_RANDOM].copy()
    boundary_majority = {}
    for task in TASKS:
        maj = MAJORITY_BASELINES_VAL[task]
        below = df[(df["task_name"] == task) & (df["accuracy"] < maj)]
        boundary_majority[task] = {
            "majority_baseline_val": maj,
            "n_below": int(len(below)),
            "by_method": {m: int(len(below[below["method"] == m])) for m in ALL_METHODS},
            "by_epoch": {str(e): int(len(below[below["num_train_epochs"] == e])) for e in EPOCHS},
        }

    return {
        "uniform_random_baseline": UNIFORM_RANDOM,
        "majority_baselines_validation_split": MAJORITY_BASELINES_VAL,
        "baseline_source": "Computed from GLUE validation split label distributions",
        "total_below_uniform_random": int(len(boundary_uniform)),
        "below_uniform_by_method": {m: int(len(boundary_uniform[boundary_uniform["method"] == m])) for m in ALL_METHODS},
        "below_uniform_by_task": {t: int(len(boundary_uniform[boundary_uniform["task_name"] == t])) for t in TASKS},
        "below_uniform_by_epoch": {str(e): int(len(boundary_uniform[boundary_uniform["num_train_epochs"] == e])) for e in EPOCHS},
        "below_majority_per_task": boundary_majority,
        "interpretation": "Below-random runs concentrate in full_ft at low epochs (1-2) with small samples (80). This represents undertraining boundary conditions, not systematic failure. MRPC 0.31-0.32 accuracy indicates all-negative-class prediction under severe undertraining, not a random baseline.",
    }


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows loaded")

    print("\nRunning proper ANOVA per stratum...")
    print("  Model: accuracy ~ C(method) * C(epoch) + C(sample_size)")
    anova = run_anova_per_stratum(df)

    print("\nRunning method pairwise tests (BH-FDR family 1)...")
    method_tests = run_pairwise_method_tests(df)
    valid_m = [t for t in method_tests if t["status"] == "valid"]
    degen_m = [t for t in method_tests if t["status"] != "valid"]
    sig_m = sum(1 for t in valid_m if t.get("significant_bh", False))
    nan_m = sum(1 for t in method_tests if np.isnan(t.get("p_adjusted", 0)))
    print(f"  Valid: {len(valid_m)}, Degenerate: {len(degen_m)}, BH-significant: {sig_m}, NaN in p_adjusted: {nan_m}")

    print("\nRunning epoch pairwise tests (BH-FDR family 2)...")
    epoch_tests = run_pairwise_epoch_tests(df)
    valid_e = [t for t in epoch_tests if t["status"] == "valid"]
    degen_e = [t for t in epoch_tests if t["status"] != "valid"]
    sig_e = sum(1 for t in valid_e if t.get("significant_bh", False))
    nan_e = sum(1 for t in epoch_tests if np.isnan(t.get("p_adjusted", 0)))
    print(f"  Valid: {len(valid_e)}, Degenerate: {len(degen_e)}, BH-significant: {sig_e}, NaN in p_adjusted: {nan_e}")

    print("\nComputing descriptive statistics...")
    descriptive = compute_descriptive_stats(df)

    print("\nBoundary condition analysis...")
    boundary = boundary_condition_analysis(df)
    print(f"  {boundary['total_below_uniform_random']} runs below uniform random (0.5)")

    # Assemble output
    output = {
        "metadata": {
            "total_runs": int(len(df)),
            "methods": ALL_METHODS,
            "peft_methods": PEFT_METHODS,
            "models": MODELS,
            "tasks": TASKS,
            "epochs": EPOCHS,
            "sample_sizes": SAMPLE_SIZES,
            "seeds": SEEDS,
            "full_ft_role": "reference_baseline_only_not_in_peft_anova",
            "primary_endpoint": "final_epoch_validation_accuracy",
            "anova_model": "accuracy ~ C(method) * C(epoch) + C(sample_size)",
            "anova_type": "Type II",
        },
        "anova_per_stratum": anova,
        "method_pairwise_tests": method_tests,
        "epoch_pairwise_tests": epoch_tests,
        "pairwise_summary": {
            "method_family": {"total": len(method_tests), "valid": len(valid_m), "degenerate": len(degen_m), "bh_significant": sig_m, "nan_count": nan_m},
            "epoch_family": {"total": len(epoch_tests), "valid": len(valid_e), "degenerate": len(degen_e), "bh_significant": sig_e, "nan_count": nan_e},
        },
        "descriptive_stats": descriptive,
        "boundary_conditions": boundary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {STATS_JSON}")

    # Print ANOVA summary
    print("\n" + "=" * 70)
    print("ANOVA SUMMARY (Type II): accuracy ~ C(method)*C(epoch) + C(sample_size)")
    print("=" * 70)
    for stratum, res in sorted(anova.items()):
        print(f"\n  {stratum}:")
        if "error" in res:
            print(f"    ERROR: {res['error']}")
            continue
        for term, vals in res.items():
            if term == "Residual":
                continue
            sig = ""
            p = vals.get("p", 1.0)
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            print(f"    {term:20s}  df={vals['df']:2d}  F={vals['F']:8.3f}  p={p:.6f} {sig}")


if __name__ == "__main__":
    main()
