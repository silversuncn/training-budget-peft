#!/usr/bin/env python3
"""Environment preflight check — models, datasets, GPU, offline mode."""
import sys
import os

def check(label, condition, detail=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {label}" + (f" ({detail})" if detail else ""))
    return condition

def main():
    all_ok = True
    print("=" * 60)
    print("Environment Preflight Check")
    print("=" * 60)

    # 1. Python version
    print("\n[1] Python")
    all_ok &= check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])

    # 2. Core packages
    print("\n[2] Core packages")
    pkgs = {}
    for pkg in ["torch", "transformers", "peft", "datasets", "sklearn", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            pkgs[pkg] = ver
            all_ok &= check(f"{pkg}", True, ver)
        except ImportError:
            all_ok &= check(f"{pkg}", False, "NOT INSTALLED")

    # 3. CUDA / GPU
    print("\n[3] GPU")
    import torch
    cuda_ok = torch.cuda.is_available()
    all_ok &= check("CUDA available", cuda_ok)
    if cuda_ok:
        check("Device", True, torch.cuda.get_device_name(0))
        check("VRAM", True, f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        check("bf16 support", torch.cuda.is_bf16_supported())

    # 4. Models (offline)
    print("\n[4] Models (offline cache)")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    for model_name in ["bert-base-uncased", "roberta-base"]:
        try:
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True)
            all_ok &= check(f"{model_name}", True, "cached")
        except Exception as e:
            all_ok &= check(f"{model_name}", False, str(e)[:80])

    # 5. Datasets (offline)
    print("\n[5] Datasets (offline cache)")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    for task in ["sst2", "mrpc", "qnli", "rte"]:
        try:
            ds = load_dataset("glue", task)
            train_len = len(ds["train"])
            all_ok &= check(f"glue/{task}", True, f"train={train_len}")
        except Exception as e:
            all_ok &= check(f"glue/{task}", False, str(e)[:80])

    # 6. PEFT configs
    print("\n[6] PEFT method configs")
    from peft import LoraConfig, IA3Config
    try:
        LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, target_modules=["query", "value"])
        all_ok &= check("LoraConfig", True)
    except Exception as e:
        all_ok &= check("LoraConfig", False, str(e)[:80])
    try:
        IA3Config(task_type="SEQ_CLS", target_modules=["query", "value", "intermediate.dense"], feedforward_modules=["intermediate.dense"])
        all_ok &= check("IA3Config", True)
    except Exception as e:
        all_ok &= check("IA3Config", False, str(e)[:80])

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — ready for Smoke Test")
    else:
        print("SOME CHECKS FAILED — fix before proceeding")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
