#!/usr/bin/env python3
"""
Training Budget Sensitivity — Figure Generation
Generates publication-quality figures for the paper.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "data" / "results.csv"
if not RESULTS_CSV.exists():
    RESULTS_CSV = ROOT / "artifacts" / "final_runs" / "results.csv"
FIGURES_DIR = ROOT / "figures"

PEFT_METHODS = ["lora", "bitfit", "ia3"]
ALL_METHODS = ["full_ft", "lora", "bitfit", "ia3"]
EPOCHS = [1, 2, 5, 10]
MODELS = ["bert-base-uncased", "roberta-base"]
TASKS = ["sst2", "mrpc", "qnli", "rte"]
SAMPLE_SIZES = [80, 320, 1280]

METHOD_LABELS = {"full_ft": "Full FT (ref)", "lora": "LoRA", "bitfit": "BitFit", "ia3": "(IA)³"}
METHOD_COLORS = {"full_ft": "#888888", "lora": "#1f77b4", "bitfit": "#ff7f0e", "ia3": "#2ca02c"}
METHOD_MARKERS = {"full_ft": "s", "lora": "o", "bitfit": "^", "ia3": "D"}
TASK_LABELS = {"sst2": "SST-2", "mrpc": "MRPC", "qnli": "QNLI", "rte": "RTE"}
MODEL_LABELS = {"bert-base-uncased": "BERT", "roberta-base": "RoBERTa"}


def load_data():
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def get_mean_acc(rows, method, model, task, epoch):
    accs = [float(r["accuracy"]) for r in rows
            if r["method"] == method and r["model_name"] == model
            and r["task_name"] == task and int(r["num_train_epochs"]) == epoch]
    return np.mean(accs) if accs else None


def get_std_acc(rows, method, model, task, epoch):
    accs = [float(r["accuracy"]) for r in rows
            if r["method"] == method and r["model_name"] == model
            and r["task_name"] == task and int(r["num_train_epochs"]) == epoch]
    return np.std(accs, ddof=1) if len(accs) > 1 else 0.0


def fig1_interaction_plot(rows):
    """Method x Epoch interaction plot (2x4 grid: models x tasks)."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    fig.suptitle("Method × Epoch Interaction: Accuracy vs Training Epochs", fontsize=13, y=0.98)

    for i, model in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for method in ALL_METHODS:
                means = [get_mean_acc(rows, method, model, task, ep) for ep in EPOCHS]
                stds = [get_std_acc(rows, method, model, task, ep) for ep in EPOCHS]
                ls = "--" if method == "full_ft" else "-"
                ax.errorbar(EPOCHS, means, yerr=stds, label=METHOD_LABELS[method],
                           color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                           linestyle=ls, markersize=5, capsize=3, linewidth=1.5)
            ax.set_xlabel("Epochs" if i == 1 else "")
            ax.set_ylabel("Accuracy" if j == 0 else "")
            ax.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[task]}", fontsize=10)
            ax.set_xticks(EPOCHS)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig1_method_epoch_interaction.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig2_sensitivity_heatmap(rows):
    """Epoch sensitivity heatmap: delta(acc) from epoch=10 to epoch=1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5),
                             gridspec_kw={"wspace": 0.3, "right": 0.88})
    fig.suptitle("Epoch Sensitivity: Accuracy Gain (epoch=10 vs epoch=1)", fontsize=12, y=0.98)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        data = np.zeros((len(PEFT_METHODS), len(TASKS)))
        for i, method in enumerate(PEFT_METHODS):
            for j, task in enumerate(TASKS):
                acc_10 = get_mean_acc(rows, method, model, task, 10)
                acc_1 = get_mean_acc(rows, method, model, task, 1)
                if acc_10 is not None and acc_1 is not None:
                    data[i, j] = acc_10 - acc_1

        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.4)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
        ax.set_yticks(range(len(PEFT_METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in PEFT_METHODS])
        ax.set_title(MODEL_LABELS[model], fontsize=11)

        for i in range(len(PEFT_METHODS)):
            for j in range(len(TASKS)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9,
                       color="black" if data[i,j] < 0.3 else "white")

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Δ Accuracy (ep10 − ep1)")
    out = FIGURES_DIR / "fig2_epoch_sensitivity_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig3_sample_size_epoch_interaction(rows):
    """Sample size x epoch interaction for LoRA (best PEFT method)."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    fig.suptitle("LoRA: Sample Size × Epoch Interaction", fontsize=13, y=0.98)

    size_colors = {80: "#d62728", 320: "#9467bd", 1280: "#17becf"}

    for i, model in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for size in SAMPLE_SIZES:
                means = []
                stds = []
                for ep in EPOCHS:
                    accs = [float(r["accuracy"]) for r in rows
                            if r["method"] == "lora" and r["model_name"] == model
                            and r["task_name"] == task and int(r["num_train_epochs"]) == ep
                            and int(r["train_subset_size"]) == size]
                    means.append(np.mean(accs) if accs else 0)
                    stds.append(np.std(accs, ddof=1) if len(accs) > 1 else 0)
                ax.errorbar(EPOCHS, means, yerr=stds, label=f"n={size}",
                           color=size_colors[size], marker="o", capsize=3, linewidth=1.5)
            ax.set_xlabel("Epochs" if i == 1 else "")
            ax.set_ylabel("Accuracy" if j == 0 else "")
            ax.set_title(f"{MODEL_LABELS[model]} / {TASK_LABELS[task]}", fontsize=10)
            ax.set_xticks(EPOCHS)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "fig3_lora_size_epoch_interaction.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} rows")

    print("\nGenerating figures...")
    fig1_interaction_plot(rows)
    fig2_sensitivity_heatmap(rows)
    fig3_sample_size_epoch_interaction(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
