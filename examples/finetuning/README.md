# Fine-Tuning with Showdown Preference Datasets

Showdown captures verified pairwise human and LLM judge preferences across subjective generation tasks (branding, coding style, system architecture, narrative generation). This directory provides instructions and runnable scripts for turning these preferences into personalized aligned models using Direct Preference Optimization (DPO), Kahneman-Tversky Optimization (KTO), or Reward Modeling.

---

## 1. Exporting Showdown Preferences

Showdown allows you to aggregate preferences across individual or multi-tournament datasets with automated lineage-based train/validation splitting:

```bash
# Export all tournaments with an 80/20 train/val split partitioned by prompt lineage:
showdown export --all --format dpo --split 0.8 --out-dir ./dataset

# Output created:
#   ./dataset/train.jsonl          (Prompt, chosen, rejected)
#   ./dataset/val.jsonl            (Independent validation set)
#   ./dataset/dataset_info.json    (Lineage and split statistics)
```

### Advanced Filtering Options:
```bash
# Filter by task type (e.g. only coding tasks):
showdown export --all --format dpo --type code --split 0.8 --out-dir ./code_dataset

# Filter by unanimous consensus across multi-evaluator matches:
showdown export --all --format dpo --consensus strict --output strict_dpo.jsonl

# Export with Bradley-Terry rating margins for margin-loss DPO:
showdown export --all --format pairwise_margins --output margins.json
```

---

## 2. Dataset Formats

### Direct Preference Optimization (DPO)
Standard format directly consumed by Hugging Face `trl.DPOTrainer` and Unsloth:
```json
{
  "prompt": "Compare and rank deployment architecture approaches for multi-agent autonomous runtime.",
  "chosen": "Docker Compose on Single Big Iron Bare-Metal: Minimal latency, zero k8s overlay overhead...",
  "rejected": "Serverless MicroVMs (Firecracker / Cloudflare Workers): Instant cold starts...",
  "chosen_id": "cand_02",
  "rejected_id": "cand_03",
  "voter": "human",
  "task_type": "text",
  "critique": "For a homelab, Docker Compose on one bare-metal box wins on operational simplicity..."
}
```

### Kahneman-Tversky Optimization (KTO)
Binary feedback format consumed by `trl.KTOTrainer`:
```json
{
  "prompt": "Write a punchy headline for an event-driven AI platform.",
  "completion": "Novasync: Zero-latency intelligence on your bare metal.",
  "label": true,
  "candidate_id": "cand_01",
  "status": "favorite",
  "task_type": "text"
}
```

---

## 3. Running DPO Training

Install training dependencies:
```bash
pip install torch transformers datasets trl peft bitsandbytes accelerate
```

Launch the training recipe:
```bash
python examples/finetuning/finetune_dpo.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --train-file ./dataset/train.jsonl \
    --val-file ./dataset/val.jsonl \
    --epochs 3 \
    --batch-size 2 \
    --lr 5e-6 \
    --beta 0.1 \
    --output-dir ./dpo_aligned_model
```

---

## 4. Serving the Aligned Model

Once training finishes, load the trained LoRA adapter directly in your local vLLM or Ollama cluster:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules showdown-aligned=./dpo_aligned_model
```

The resulting model naturally prioritizes the stylistic, architectural, and reasoning choices captured during your Showdown sessions.
