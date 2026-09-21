"""
Minimal reference training recipe for fine-tuning a local model on Showdown preference datasets.
Supports Hugging Face TRL (DPOTrainer) with LoRA / QLoRA parameter-efficient fine-tuning.

Usage:
    # 1. Export your Showdown preference dataset with train/val split:
    showdown export --all --format dpo --split 0.8 --out-dir ./dataset

    # 2. Run DPO training:
    python examples/finetuning/finetune_dpo.py \
        --model-name "Qwen/Qwen2.5-7B-Instruct" \
        --train-file ./dataset/train.jsonl \
        --val-file ./dataset/val.jsonl \
        --output-dir ./dpo_aligned_model
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig
    from trl import DPOTrainer
except ImportError:
    # Allow inspecting CLI help or importing without heavy ML libraries pre-installed
    torch = None


def load_showdown_jsonl(path: str) -> Dataset:
    """Load Showdown DPO jsonl format into a Hugging Face Dataset."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Standardize column names for DPOTrainer
    formatted = []
    for r in records:
        formatted.append({
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
    return Dataset.from_list(formatted)


def train_dpo(
    model_name: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512,
):
    if torch is None:
        raise ImportError(
            "Fine-tuning dependencies are not installed in the current environment.\n"
            "Install them with: pip install torch transformers datasets trl peft bitsandbytes accelerate"
        )

    print(f"[Showdown DPO] Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 4-bit / bfloat16 base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"[Showdown DPO] Loading train dataset: {train_file}")
    train_dataset = load_showdown_jsonl(train_file)
    print(f"  • {len(train_dataset)} training pairs loaded")

    val_dataset = None
    if val_file and os.path.exists(val_file):
        print(f"[Showdown DPO] Loading validation dataset: {val_file}")
        val_dataset = load_showdown_jsonl(val_file)
        print(f"  • {len(val_dataset)} validation pairs loaded")

    # Configure LoRA parameter-efficient adapters
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    print(f"[Showdown DPO] Starting DPOTrainer (beta={beta})...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicit reference model sharing base weights with adapter disabled
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    dpo_trainer.train()
    print(f"[Showdown DPO] Training complete! Saving adapters to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Showdown DPO] Artifacts saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune local models on Showdown preference datasets using DPO.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model identifier")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train.jsonl exported from Showdown")
    parser.add_argument("--val-file", type=str, default=None, help="Path to val.jsonl exported from Showdown")
    parser.add_argument("--output-dir", type=str, default="./dpo_aligned_model", help="Directory to save trained LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO implicit reward temperature beta")

    args = parser.parse_args()
    train_dpo(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
    )
