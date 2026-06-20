"""Training Budget Sensitivity — Constants"""
from __future__ import annotations

SUPPORTED_METHODS = ["full_ft", "lora", "bitfit", "ia3"]

SUPPORTED_MODELS = ["bert-base-uncased", "roberta-base"]

SUPPORTED_TASKS = ["sst2", "mrpc", "qnli", "rte"]

SAMPLE_SIZES = [80, 320, 1280]

SEEDS = [11, 17, 23]

EPOCHS = [1, 2, 5, 10]

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

SAMPLE_LIMITS = {"sst2": 67349, "mrpc": 3668, "qnli": 104743, "rte": 2490}

DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0

PEFT_LR = 2e-4
FULL_FT_LR = 2e-5
WARMUP_RATIO = 0.1
