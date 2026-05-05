import os
import re
import uuid
import time
import json
import requests

# Pointing to the generated evaluation prompts artifact
PROMPTS_FILE = r"C:\Users\Kaustav Roy\.gemini\antigravity\brain\1093fd29-7aec-4b21-bdd6-7b88a0e6df5d\evaluation_prompts.md"
BASE_URL = "http://localhost:8000"
RESULTS_FILE = "evaluation_results.json"

def extract_prompts(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Matches the exact prompt format from the markdown
    pattern = r'\*\s\*\*Prompt\*\*:\s"(.*?)"'
    prompts = re.findall(pattern, content)
    return prompts

def query_chatbot(prompt):
    session_id = str(uuid.uuid4())
    payload = {"message": prompt, "session_id": session_id}
    
    full_response = ""
    start_time = time.time()
    first_token_time = None
    
    try:
        response = requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}", 0, 0
            
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    if not first_token_time:
                        first_token_time = time.time() - start_time
                    
                    data_str = decoded_line.replace("data: ", "").strip()
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data_json = json.loads(data_str)
                        full_response += data_json.get("content", "")
                    except json.JSONDecodeError:
                        pass
                        
        total_time = time.time() - start_time
        return full_response.strip(), first_token_time or 0, total_time
        
    except requests.exceptions.ConnectionError:
        return "Error: Connection refused. Is the FastAPI server running?", 0, 0
    except Exception as e:
        return f"Error: {e}", 0, 0

def run_evaluations():
    print(f"Reading prompts from: {PROMPTS_FILE}")
    prompts = extract_prompts(PROMPTS_FILE)
    
    if not prompts:
        print("No prompts extracted. Please check the markdown file path and format.")
        return

    print(f"Successfully extracted {len(prompts)} prompts.")
    print(f"Starting evaluation run against {BASE_URL}...\n")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Testing: {prompt}")
        
        response, ttft, total_time = query_chatbot(prompt)
        
        result_entry = {
            "id": i,
            "prompt": prompt,
            "response": response,
            "ttft_seconds": round(ttft, 2),
            "total_time_seconds": round(total_time, 2)
        }
        results.append(result_entry)
        
        # Handle print formatting for long/empty responses
        print_resp = response[:100] + "..." if len(response) > 100 else response
        print_resp = print_resp.replace('\n', ' ')
        
        print(f"   -> TTFT: {result_entry['ttft_seconds']}s | Total: {result_entry['total_time_seconds']}s")
        print(f"   -> Response: {print_resp}\n")
        
        # Stop early if the server isn't running
        if "Connection refused" in response:
            print("Aborting evaluation: Could not connect to the backend.")
            break
            
        # Save intermediate results so we don't lose data if it crashes
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        # Optional small delay to prevent overloading the local server
        time.sleep(1)

    print(f"Evaluation complete. Full results saved to {os.path.abspath(RESULTS_FILE)}")

if __name__ == "__main__":
    run_evaluations()
