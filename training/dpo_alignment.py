import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer

# We will align the model we fine-tuned previously in `qlora_finetune.py`!
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # In reality, use "./results/final_adapter"
DATASET_NAME = "Anthropic/hh-rlhf" # Anthropic's open-source Helpful & Harmless preference dataset

def main():
    print("Loading Model & Tokenizer for Alignment...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Load the active model we want to train to be "polite/safe"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # 2. Load a Frozen Reference Model 
    # DPO compares the active model to this frozen one to ensure it doesn't degrade general intelligence
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    # Add LoRA adapters so we don't run out of memory during DPO
    peft_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    print("Loading Preference Dataset...")
    # DPO requires a dataset with three columns: 
    # 1. The Prompt
    # 2. A 'Chosen' (good) response
    # 3. A 'Rejected' (bad/rude) response
    dataset = load_dataset(DATASET_NAME, split="train[:1000]")

    def format_dpo(example):
        # Extract prompt and responses from the Anthropic dataset strings
        chosen_split = example['chosen'].rpartition('\n\nAssistant: ')
        rejected_split = example['rejected'].rpartition('\n\nAssistant: ')
        return {
            "prompt": chosen_split[0] + "\n\nAssistant: ",
            "chosen": chosen_split[2],
            "rejected": rejected_split[2],
        }

    formatted_dataset = dataset.map(format_dpo)

    print("Starting Direct Preference Optimization (DPO)...")
    training_args = TrainingArguments(
        output_dir="./dpo_results",
        per_device_train_batch_size=1,
        learning_rate=5e-6, # Important: DPO requires much lower learning rates than standard fine-tuning
        max_steps=200,
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.1, # Controls how strongly we penalize deviating from the reference model
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    
    print("Alignment Complete! Saving your Claude-like safe model...")
    dpo_trainer.model.save_pretrained("./dpo_results/aligned_model")

if __name__ == "__main__":
    main()
