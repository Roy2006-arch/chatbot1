import json
import re
import random
from datasets import load_dataset
import os

# ==========================================
# 1. CLEANING TEXT
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive newlines and spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove obvious PII (Example: simple email regex)
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]', text)
    
    return text.strip()

# ==========================================
# 2. AUGMENTATION
# ==========================================
SYSTEM_PROMPTS = [
    "You are a helpful and polite AI assistant.",
    "You are an expert customer support bot.",
    "You are an AI trained to provide concise, accurate answers."
]

def augment_conversation(messages):
    """
    Randomly injects a varied system prompt to make the model robust.
    """
    has_system = any(msg.get("role") == "system" for msg in messages)
    
    if not has_system:
        sys_prompt = random.choice(SYSTEM_PROMPTS)
        messages.insert(0, {"role": "system", "content": sys_prompt})
        
    return messages

# ==========================================
# 3. FORMATTING TO JSONL
# ==========================================
def process_huggingface_dataset(output_file, num_samples=1000):
    """
    Downloads a public dataset, cleans it, formats it, and saves to JSONL.
    We use 'databricks/databricks-dolly-15k' as a great public example of Instruction data.
    """
    print("Downloading dataset...")
    # Load public instruction dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    processed_count = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in dataset:
            if processed_count >= num_samples: break
            
            # Dolly dataset has 'instruction', 'context', and 'response'
            user_input = clean_text(row['instruction'])
            if row.get('context'):
                user_input += f"\n\nContext: {clean_text(row['context'])}"
            
            assistant_response = clean_text(row['response'])
            
            # Format to ChatML structure
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ]
            
            # Augment
            messages = augment_conversation(messages)
            
            # Write to JSONL
            json_line = json.dumps({"messages": messages})
            f.write(json_line + '\n')
            processed_count += 1
            
    print(f"Successfully processed {processed_count} conversations into {output_file}")

if __name__ == "__main__":
    output_path = "processed_data/train.jsonl"
    print("Starting Data Pipeline...")
    process_huggingface_dataset(output_path, num_samples=500)
    print("Pipeline Complete! Check the 'processed_data' folder.")
