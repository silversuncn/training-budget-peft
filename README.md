# Training Budget Sensitivity of LoRA and Lightweight PEFT Methods

This repository contains the experiment code and data for the paper:

> **Training Budget Sensitivity of LoRA and Lightweight PEFT Methods under Low-Resource Text Classification**
> Yaowen Sun, Xin Zhang, Jianting Gao

## Overview

We study how sensitive lightweight PEFT methods (LoRA, BitFit, (IA)³) are to training epoch budget under low-resource text classification. The experiments cover:

- **Models**: BERT-base-uncased, RoBERTa-base
- **Tasks**: SST-2, MRPC, QNLI, RTE (GLUE)
- **Sample sizes**: 80, 320, 1280
- **Epoch budgets**: 1, 2, 5, 10
- **Seeds**: 11, 17, 23
- **Total runs**: 1152 (including full fine-tuning reference)

## Repository Structure

```
├── src/
│   ├── constants.py            # Supported methods, models, tasks
│   ├── grid_runner.py          # Grid search launcher
│   ├── aggregate_results.py    # Aggregate per-run metrics into results.csv
│   ├── statistical_analysis.py # Type II ANOVA + BH-corrected pairwise tests
│   ├── generate_figures.py     # Publication figure generation
│   ├── pilot_analysis.py       # Pilot signal analysis
│   ├── diagnose_stats.py       # Diagnostic utilities
│   └── preflight.py            # Environment check
├── data/
│   └── results.csv             # Aggregated results (1152 rows)
├── figures/                    # Publication figures (PDF)
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Transformers 4.36+
- PEFT 0.7+
- scipy, statsmodels, pandas, numpy, matplotlib, seaborn

## Usage

```bash
# Run statistical analysis on pre-computed results
python src/statistical_analysis.py

# Generate publication figures
python src/generate_figures.py

# Pilot signal check
python src/pilot_analysis.py
```

The `data/results.csv` file contains all 1152 experiment runs. Each row records the method, model, task, sample size, epoch budget, seed, and validation accuracy.

## Key Findings

- Epoch budget is significant in all 8 model–task strata (PEFT-only ANOVA)
- Method–epoch interaction is significant in 4/8 strata
- LoRA shows the largest epoch sensitivity; BitFit and (IA)³ are more stable but with lower ceilings
- Corrected pairwise: 22/133 epoch comparisons survive BH-FDR; only 1/88 method comparisons survives

## Citation

```bibtex
@article{sun2026training,
  title={Training Budget Sensitivity of LoRA and Lightweight PEFT Methods under Low-Resource Text Classification},
  author={Sun, Yaowen and Zhang, Xin and Gao, Jianting},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
