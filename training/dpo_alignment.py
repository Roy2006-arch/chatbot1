"""
training/dpo_alignment.py
--------------------------
Direct Preference Optimization (DPO) alignment using self-improvement
pipeline corrections as preference pairs.

Reads DPO-format data from:
  1. Self-improvement pipeline exports (feedback/connector.py dpo_pairs.jsonl)
  2. Retraining pipeline DPO exports (data/feedback/retrain_dpo_candidates.jsonl)
  3. Auto-improve DPO exports (data/feedback/auto_improve_dpo_*.jsonl)

Replaces the original placeholder Anthropic/hh-rlhf dataset.

Usage:
    python training/dpo_alignment.py                                           # default DPO
    python training/dpo_alignment.py --dataset ./path/to/dpo_pairs.jsonl
    python training/dpo_alignment.py --qlora-checkpoint ./results/stage4_expert/final_adapter
"""

import argparse
import json
import glob
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset as HFDataset
from trl import DPOTrainer

log = logging.getLogger("chatbot.dpo_alignment")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./dpo_results"

DPO_SOURCE_PATHS = [
    os.path.join("self_improvement_data", "curriculum", "dpo_pairs.jsonl"),
    os.path.join("data", "feedback", "retrain_dpo_candidates.jsonl"),
]

AUTO_IMPROVE_PATTERN = os.path.join("data", "feedback", "auto_improve_dpo_*.jsonl")


def discover_dpo_datasets() -> List[str]:
    """Discover available DPO dataset files from all sources."""
    paths = []
    for p in DPO_SOURCE_PATHS:
        if os.path.exists(p):
            paths.append(p)
    for p in sorted(glob.glob(AUTO_IMPROVE_PATTERN)):
        if p not in paths:
            paths.append(p)
    return paths


def load_dpo_dataset(dataset_paths: List[str]) -> HFDataset:
    """Load DPO pairs from JSONL files into a Hugging Face Dataset."""
    records = []
    for path in dataset_paths:
        if not os.path.exists(path):
            log.warning("DPO dataset not found: %s", path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    prompt = record.get("prompt", "")

                    # DPO pair can be in different formats
                    if "chosen" in record and "rejected" in record:
                        chosen = record["chosen"]
                        rejected = record["rejected"]
                    elif "corrected_response" in record and "original_response" in record:
                        chosen = record["corrected_response"]
                        rejected = record["original_response"]
                    else:
                        continue

                    if prompt and chosen and rejected:
                        records.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                        })
                except (json.JSONDecodeError, ValueError):
                    continue

    if not records:
        # Fallback: build DPO pairs from dataset_bridge
        log.info("No DPO datasets found. Building from dataset_bridge...")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from training.dataset_bridge import build_dataset
            ds = build_dataset(max_examples=2000)
            for ex in ds.train:
                if ex.source == "self_improvement" and ex.metadata.get("export_source") == "correction":
                    records.append({
                        "prompt": ex.prompt,
                        "chosen": ex.response,
                        "rejected": ex.metadata.get("original_response", ""),
                    })
        except Exception as e:
            log.error("Fallback dataset_bridge failed: %s", e)

    log.info("Loaded %d DPO pairs from %d sources", len(records), len(dataset_paths))
    return HFDataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="DPO Alignment Training")
    parser.add_argument("--dataset", default="", help="Path to DPO JSONL (overrides auto-discover)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model or QLoRA checkpoint")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    args = parser.parse_args()

    log.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if args.dataset:
        dataset_paths = [args.dataset]
    else:
        dataset_paths = discover_dpo_datasets()

    dataset = load_dpo_dataset(dataset_paths)

    if len(dataset) == 0:
        log.error("No DPO training data available. Create DPO pairs via:")
        log.error("  python scripts/run_self_improvement.py --dpo-export")
        log.error("  or place DPO JSONL at: %s", DPO_SOURCE_PATHS[0])
        sys.exit(1)

    log.info("Starting DPO training with %d pairs...", len(dataset))
    log.info("  beta=%.2f  lr=%s  max_steps=%d  batch_size=%d",
             args.beta, args.lr, args.max_steps, args.batch_size)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()

    aligned_path = os.path.join(args.output, "aligned_model")
    dpo_trainer.model.save_pretrained(aligned_path)
    tokenizer.save_pretrained(aligned_path)
    log.info("DPO alignment complete. Model saved to %s", aligned_path)


if __name__ == "__main__":
    main()
