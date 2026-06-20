#!/usr/bin/env python3
"""
Training Budget Sensitivity — Grid Runner
Single-file grid runner with manual training loop, offline mode, and progress logging.
"""
from __future__ import annotations
import os
os.environ.update({
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHINDUCTOR_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "CUDA_MODULE_LOADING": "LAZY",
})
import argparse
import copy
import fcntl
import gc
import json
import time
import traceback
from itertools import product
from pathlib import Path

import torch
torch._dynamo.config.disable = True
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, IA3Config, TaskType
from sklearn.model_selection import StratifiedShuffleSplit

from constants import (
    SUPPORTED_METHODS, SUPPORTED_MODELS, SUPPORTED_TASKS,
    SAMPLE_SIZES, SEEDS, EPOCHS, TASK_TO_KEYS, SAMPLE_LIMITS,
    DEFAULT_MAX_LENGTH, DEFAULT_BATCH_SIZE,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
    PEFT_LR, FULL_FT_LR, WARMUP_RATIO,
)

ROOT = Path(__file__).resolve().parents[1]
PLOG = ROOT / "progress.log"
LLOG = ROOT / "pipeline.log"
ARTIFACTS = ROOT / "artifacts" / "final_runs"


def log(m, l="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{l}] {ts} {m}\n"
    print(s.strip(), flush=True)
    with open(LLOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(s)
        fcntl.flock(f, fcntl.LOCK_UN)


def logp(method, model, task, n, ep, seed, acc, status, elapsed):
    ts = time.strftime("%H:%M:%S")
    s = (f"{ts} | {status:4s} | {method:8s} | {model:20s} | "
         f"{task:5s} | n={n:5d} | ep={ep:2d} | s={seed:2d} | acc={acc:.4f} | {elapsed:.0f}s\n")
    with open(PLOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(s)
        fcntl.flock(f, fcntl.LOCK_UN)


def make_run_dir(method, model_name, task, n, epochs, seed):
    safe_model = model_name.replace("/", "-")
    name = f"{method}__{safe_model}__{task}__n{n}__ep{epochs}__s{seed}"
    return ARTIFACTS / name


def prepare_bitfit(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.endswith("bias"):
            param.requires_grad = True
    for name, param in model.named_parameters():
        if any(kw in name for kw in ("classifier", "score", "pre_classifier")):
            param.requires_grad = True
    return model


def prepare_lora(model):
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
    )
    return get_peft_model(model, config)


def prepare_ia3(model):
    config = IA3Config(
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "value", "intermediate.dense"],
        feedforward_modules=["intermediate.dense"],
    )
    return get_peft_model(model, config)


def prepare_model(method, base_model):
    model = copy.deepcopy(base_model)
    if method == "full_ft":
        return model
    elif method == "lora":
        return prepare_lora(model)
    elif method == "bitfit":
        return prepare_bitfit(model)
    elif method == "ia3":
        return prepare_ia3(model)
    else:
        raise ValueError(f"Unknown method: {method}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percentage": round(100.0 * trainable / total, 4)}


def get_peft_info(method, model):
    """Extract PEFT config details for IA3 judgment criteria."""
    info = {"peft_type": method}
    if hasattr(model, "peft_config"):
        for adapter_name, cfg in model.peft_config.items():
            info["adapter_name"] = adapter_name
            if hasattr(cfg, "target_modules"):
                info["target_modules"] = list(cfg.target_modules) if cfg.target_modules else []
            if hasattr(cfg, "feedforward_modules"):
                info["feedforward_modules"] = list(cfg.feedforward_modules) if cfg.feedforward_modules else []
            break
    return info


def tokenize_dataset(raw_dataset, tokenizer, task):
    s1_key, s2_key = TASK_TO_KEYS[task]
    def tok_fn(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], truncation=True, max_length=DEFAULT_MAX_LENGTH, padding="max_length")
        return tokenizer(examples[s1_key], examples[s2_key], truncation=True, max_length=DEFAULT_MAX_LENGTH, padding="max_length")
    ds = raw_dataset.map(tok_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    ds.set_format("torch", columns=cols)
    return ds


def subsample(dataset, n, seed):
    actual_n = min(n, len(dataset))
    labels = np.array(dataset["labels"])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=actual_n, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, idx.tolist()), actual_n


def train_eval(model, train_loader, val_loader, epochs, lr, device):
    total_steps = len(train_loader) * epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.append(logits.argmax(-1).cpu())
            labels.append(batch["labels"].cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    accuracy = float((preds == labels).mean())
    return accuracy


def run_single(method, model_name, task, n, epochs, seed, base_model, tokenizer, train_ds_full, val_ds, device):
    run_dir = make_run_dir(method, model_name, task, n, epochs, seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    train_subset, actual_n = subsample(train_ds_full, n, seed)
    train_loader = DataLoader(train_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DEFAULT_BATCH_SIZE * 2, shuffle=False)

    model = prepare_model(method, base_model)
    model.to(device)
    param_stats = count_params(model)
    peft_info = get_peft_info(method, model) if method in ("lora", "ia3") else {"peft_type": method}

    lr = FULL_FT_LR if method == "full_ft" else PEFT_LR
    t0 = time.time()
    accuracy = train_eval(model, train_loader, val_loader, epochs, lr, device)
    elapsed = time.time() - t0

    result = {
        "method": method,
        "model_name": model_name,
        "task_name": task,
        "train_subset_size": actual_n,
        "num_train_epochs": epochs,
        "seed": seed,
        "accuracy": accuracy,
        "learning_rate": lr,
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_length": DEFAULT_MAX_LENGTH,
        "lora_r": DEFAULT_LORA_R if method == "lora" else None,
        "lora_alpha": DEFAULT_LORA_ALPHA if method == "lora" else None,
        "warmup_ratio": WARMUP_RATIO,
        "bf16": True,
        "elapsed_seconds": round(elapsed, 1),
        **param_stats,
        **peft_info,
    }

    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (run_dir / "config.json").write_text(json.dumps(result, indent=2))

    del model
    torch.cuda.empty_cache()
    return accuracy, elapsed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=SUPPORTED_METHODS)
    p.add_argument("--models", nargs="+", default=SUPPORTED_MODELS)
    p.add_argument("--tasks", nargs="+", default=SUPPORTED_TASKS)
    p.add_argument("--sample_sizes", nargs="+", type=int, default=SAMPLE_SIZES)
    p.add_argument("--epochs", nargs="+", type=int, default=EPOCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Grid: {len(args.methods)} methods x {len(args.models)} models x {len(args.tasks)} tasks x {len(args.sample_sizes)} sizes x {len(args.epochs)} epochs x {len(args.seeds)} seeds")
    total = len(args.methods) * len(args.models) * len(args.tasks) * len(args.sample_sizes) * len(args.epochs) * len(args.seeds)
    log(f"Total runs: {total}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    done, ok, fail, skip = 0, 0, 0, 0

    for model_name in args.models:
        log(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, local_files_only=True, torch_dtype=torch.bfloat16
        )

        for task in args.tasks:
            log(f"Loading dataset: glue/{task}")
            raw = load_dataset("glue", task)
            train_ds_full = tokenize_dataset(raw["train"], tokenizer, task)
            val_ds = tokenize_dataset(raw["validation"], tokenizer, task)

            for n, ep, seed, method in product(args.sample_sizes, args.epochs, args.seeds, args.methods):
                done += 1
                try:
                    result = run_single(method, model_name, task, n, ep, seed, base_model, tokenizer, train_ds_full, val_ds, device)
                    if result is None:
                        skip += 1
                        continue
                    acc, elapsed = result
                    logp(method, model_name, task, n, ep, seed, acc, "OK", elapsed)
                    ok += 1
                except Exception as e:
                    logp(method, model_name, task, n, ep, seed, 0.0, "FAIL", 0)
                    log(f"FAIL [{done}/{total}]: {method}/{model_name}/{task}/n={n}/ep={ep}/s={seed}: {e}", "ERROR")
                    traceback.print_exc()
                    fail += 1

                if done % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            del train_ds_full, val_ds
            gc.collect()

        del base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    log(f"DONE: {ok} OK, {fail} FAIL, {skip} SKIP out of {total} total")


if __name__ == "__main__":
    main()
