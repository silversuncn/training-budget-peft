#!/usr/bin/env python3
"""Diagnose BH-FDR significance issue."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_stats_path = ROOT / "output" / "statistical_analysis.json"
if not _stats_path.exists():
    _stats_path = ROOT / "artifacts" / "statistical_analysis.json"
if not _stats_path.exists():
    raise FileNotFoundError(
        "statistical_analysis.json not found. Run `python src/statistical_analysis.py` first."
    )
with open(_stats_path) as f:
    data = json.load(f)

mp = data["method_pairwise_tests"]
ep = data["epoch_pairwise_tests"]

mp_pvals = sorted([t["p_value"] for t in mp])
ep_pvals = sorted([t["p_value"] for t in ep])

print("=== Method Pairwise (Family 1) ===")
print(f"  Total tests: {len(mp)}")
print(f"  Raw p < 0.05: {sum(1 for p in mp_pvals if p < 0.05)}")
print(f"  Raw p < 0.10: {sum(1 for p in mp_pvals if p < 0.10)}")
print(f"  Smallest 10 p-values: {mp_pvals[:10]}")
n_pairs = mp[0]["n_pairs"]
print(f"  n_pairs per test: {n_pairs}")

print(f"\n=== Epoch Pairwise (Family 2) ===")
print(f"  Total tests: {len(ep)}")
print(f"  Raw p < 0.05: {sum(1 for p in ep_pvals if p < 0.05)}")
print(f"  Raw p < 0.10: {sum(1 for p in ep_pvals if p < 0.10)}")
print(f"  Smallest 10 p-values: {ep_pvals[:10]}")
n_pairs_ep = ep[0]["n_pairs"]
print(f"  n_pairs per test: {n_pairs_ep}")

print(f"\n=== Strongest epoch pairwise effects (top 15 by raw p) ===")
for t in sorted(ep, key=lambda x: x["p_value"])[:15]:
    m = t["method"]
    mo = t["model"].replace("bert-base-uncased", "BERT").replace("roberta-base", "RoBERTa")
    ta = t["task"]
    print(f"  {m:6s} {mo:8s} {ta:5s} ep{t['epoch_1']}->{t['epoch_2']:2d} "
          f"diff={t['mean_diff']:+.4f} d={t['cohens_dz']:+.3f} p={t['p_value']:.4f}")

print(f"\n=== Strongest method pairwise effects (top 15 by raw p) ===")
for t in sorted(mp, key=lambda x: x["p_value"])[:15]:
    m1 = t["method_1"]
    m2 = t["method_2"]
    mo = t["model"].replace("bert-base-uncased", "BERT").replace("roberta-base", "RoBERTa")
    ta = t["task"]
    print(f"  {m1:6s} vs {m2:6s} {mo:8s} {ta:5s} ep={t['epoch']:2d} "
          f"diff={t['mean_diff']:+.4f} d={t['cohens_dz']:+.3f} p={t['p_value']:.4f}")

print(f"\n=== Diagnosis ===")
print(f"  n_pairs = {n_pairs} (3 sizes x 3 seeds = 9)")
print(f"  df = {n_pairs - 1} per test")
print(f"  With df=8, t-test has low power for small-to-medium effects")
print(f"  ANOVA detects signal (7/8 strata significant) because it pools all observations")
print(f"  Pairwise tests split into many small cells, losing power")
print(f"  This is a known limitation of the 3-seed x 3-size design")
