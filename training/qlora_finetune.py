import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Note: In a real scenario, you might want to use 'unsloth' for 2x faster training. 
# This is a standard Hugging Face PEFT implementation.

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME = "your_custom_dataset.jsonl"
OUTPUT_DIR = "./results"

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Model in 4-bit...")
    # Requires bitsandbytes library
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    print("Loading Dataset...")
    # Expects JSONL format: {"text": "<s>[INST] What is X? [/INST] X is Y. </s>"}
    # You will need to write a formatting function based on your specific JSON structure
    dataset = load_dataset("json", data_files=DATASET_NAME, split="train")

    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=500, # Adjust based on dataset size
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training...")
    trainer.train()

    print("Saving Model Adapters...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    print("Training Complete!")

if __name__ == "__main__":
    main()
