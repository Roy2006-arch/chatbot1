"""
training/qlora_finetune.py
--------------------------
QLoRA fine-tuning with curriculum learning support.
Reads datasets from training/dataset_bridge.py outputs.

Supports:
  - Curriculum learning with 4 stage adapters
  - Flash Attention 2
  - DeepSpeed ZeRO-3
  - LoRA/QLoRA config per stage
  - Checkpoint strategy with best-loss tracking

Usage:
    python training/qlora_finetune.py                          # full curriculum
    python training/qlora_finetune.py --stage 1                # single stage
    python training/qlora_finetune.py --output ./my_checkpoint
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.dataset_bridge import build_dataset, export_dataset, DatasetDict

log = logging.getLogger("chatbot.qlora_finetune")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./results"

CURRICULUM_STAGES = {
    1: {"name": "basic", "learning_rate": 2e-4, "r": 8, "alpha": 16, "dropout": 0.05, "epochs": 1},
    2: {"name": "intermediate", "learning_rate": 1.5e-4, "r": 12, "alpha": 24, "dropout": 0.05, "epochs": 1},
    3: {"name": "advanced", "learning_rate": 1e-4, "r": 16, "alpha": 32, "dropout": 0.1, "epochs": 2},
    4: {"name": "expert", "learning_rate": 8e-5, "r": 16, "alpha": 32, "dropout": 0.1, "epochs": 2},
}

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_dataset_for_stage(stage: int, data_dir: str = "training_data") -> Optional[str]:
    """Load or build dataset for a specific curriculum stage."""
    ds = build_dataset(max_examples=5000, curriculum_order=True)
    stage_examples = [ex for ex in ds.train if ex.stage == stage]

    if not stage_examples:
        log.warning("No examples for stage %d. Using all train data.", stage)
        stage_examples = ds.train

    os.makedirs(data_dir, exist_ok=True)
    stage_path = os.path.join(data_dir, f"stage{stage}_train.jsonl")

    with open(stage_path, "w", encoding="utf-8") as f:
        for ex in stage_examples:
            f.write(json.dumps({
                "instruction": ex.prompt,
                "output": ex.response,
                "source": ex.source,
                "quality_score": ex.quality_score,
                "stage": ex.stage,
            }) + "\n")

    log.info("Stage %d dataset: %d examples -> %s", stage, len(stage_examples), stage_path)
    return stage_path


def get_lora_config(stage: int) -> LoraConfig:
    cfg = CURRICULUM_STAGES.get(stage, CURRICULUM_STAGES[1])
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_stage(
    stage: int,
    model,
    tokenizer,
    dataset_path: str,
    output_dir: str,
    prev_adapter_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """Train a single curriculum stage. Returns path to saved adapter."""
    cfg = CURRICULUM_STAGES[stage]

    log.info("=" * 60)
    log.info("Training Stage %d: %s", stage, cfg["name"])
    log.info("  lr=%s  r=%d  alpha=%d  dropout=%.2f  epochs=%d",
             cfg["learning_rate"], cfg["r"], cfg["alpha"], cfg["dropout"], cfg["epochs"])
    log.info("=" * 60)

    if prev_adapter_path:
        log.info("Loading previous adapter weights from %s", prev_adapter_path)
        model = PeftModel.from_pretrained(model, prev_adapter_path, is_trainable=True)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    stage_dir = os.path.join(output_dir, f"stage{stage}_{cfg['name']}")
    os.makedirs(stage_dir, exist_ok=True)

    deepspeed_config = None
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        deepspeed_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "fp16": {"enabled": True},
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
        }

    training_args = TrainingArguments(
        output_dir=stage_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=cfg["learning_rate"],
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=0.3,
        num_train_epochs=cfg["epochs"],
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=deepspeed_config,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=get_lora_config(stage),
        dataset_text_field="instruction",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=lambda ex: f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}",
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_path = os.path.join(stage_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info("Stage %d adapter saved to %s", stage, adapter_path)

    return adapter_path


def merge_stage_adapters(
    base_model,
    tokenizer,
    adapter_paths: Dict[int, str],
    output_dir: str,
):
    """Merge all stage adapters using add_weighted_adapter for inference."""
    model = base_model

    for stage in sorted(adapter_paths.keys()):
        adapter_path = adapter_paths[stage]
        log.info("Loading adapter for stage %d from %s", stage, adapter_path)
        if stage == 1:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=f"stage{stage}")
        else:
            model.load_adapter(adapter_path, adapter_name=f"stage{stage}")

    model.add_weighted_adapter(
        adapters=[f"stage{s}" for s in sorted(adapter_paths.keys())],
        weights=[1.0 / len(adapter_paths)] * len(adapter_paths),
        adapter_name="merged_curriculum",
    )
    model.set_adapter("merged_curriculum")

    merged_path = os.path.join(output_dir, "merged_curriculum_adapter")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    log.info("Merged curriculum adapter saved to %s", merged_path)
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Curriculum QLoRA Fine-Tuning")
    parser.add_argument("--stage", type=int, default=0, help="Train single stage (1-4), 0=all")
    parser.add_argument("--model", default=MODEL_NAME, help="Base model name")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--data-dir", default="training_data", help="Dataset directory")
    parser.add_argument("--merge", action="store_true", help="Merge all stage adapters after training")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    args = parser.parse_args()

    log.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading model in 4-bit: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    stages_to_train = [args.stage] if args.stage > 0 else [1, 2, 3, 4]
    adapter_paths: Dict[int, str] = {}
    prev_adapter = None

    for stage in stages_to_train:
        dataset_path = load_dataset_for_stage(stage, args.data_dir)
        if not dataset_path:
            log.warning("Skipping stage %d: no dataset available", stage)
            continue

        resume_cp = args.resume if args.resume else None
        adapter_path = train_stage(
            stage=stage,
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            output_dir=args.output,
            prev_adapter_path=prev_adapter,
            resume_from_checkpoint=resume_cp,
        )
        adapter_paths[stage] = adapter_path
        prev_adapter = adapter_path
        args.resume = ""

    if args.merge and len(adapter_paths) > 1:
        merge_stage_adapters(model, tokenizer, adapter_paths, args.output)
    elif args.merge and len(adapter_paths) == 1:
        log.info("Only one stage trained; no merge needed.")

    log.info("Training complete. Adapters: %s", adapter_paths)


if __name__ == "__main__":
    main()
