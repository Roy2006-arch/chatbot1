import os
import json
import uuid
import time
import requests

BASE_URL = "http://localhost:8000"
RESULTS_FILE = "memory_evaluation_results.json"

SCENARIOS = {
    "Scenario 1: Short-Term Memory & Coreference Resolution": [
        "Hi, I'm planning a trip to Kyoto next month for 5 days. I love historical temples and vegetarian food.",
        "What's the weather usually like there during that time?",
        "Can you recommend a hotel near the best spots you mentioned? Keep my dietary preference in mind."
    ],
    "Scenario 2: Context Switching & Resumption": [
        "Can you help me write a Python script to parse a CSV file?",
        "Actually, pause on that. I need to know how to reset my router immediately. It's a Netgear Nighthawk.",
        "Got it, the lights are blinking. Now, back to the code, how do I handle missing values in it?",
        "Will doing this affect my internet connection?"
    ],
    "Scenario 3: Long-Term Context & Summarization": [
        "Let's brainstorm a marketing campaign for a new smartwatch called 'PulseX'.",
        "The target audience is fitness enthusiasts between 20 and 35.",
        "Our main differentiator is a 14-day battery life and built-in hydration tracking.",
        "The budget is $50,000, mainly focused on Instagram and TikTok.",
        "We want to launch on Black Friday.",
        "Can you write a one-page summary of our entire campaign strategy based on everything we've discussed so far?"
    ],
    "Scenario 4: Evolving Constraints & Rule Overwriting": [
        "Draft a professional email to my boss, Sarah, asking for next Friday off. Tone should be very formal.",
        "Actually, make it much more casual. We are good friends.",
        "Add that I will be checking Slack occasionally, but only in the mornings.",
        "Change the day off from Friday to Thursday.",
        "Please give me the final version of the email now."
    ],
    "Scenario 5: Negative Constraints (Instruction Following over Time)": [
        "We are going to write a short sci-fi story. Rule: You must not use the words 'alien' or 'spacecraft' anywhere in this conversation.",
        "Start by describing the arrival of the visitors.",
        "Now describe the vehicle they arrived in.",
        "Summarize the story so far."
    ]
}

def query_chatbot(prompt, session_id):
    payload = {"message": prompt, "session_id": session_id}
    full_response = ""
    
    try:
        response = requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    data_str = decoded_line.replace("data: ", "").strip()
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data_json = json.loads(data_str)
                        full_response += data_json.get("content", "")
                    except json.JSONDecodeError:
                        pass
        return full_response.strip()
        
    except requests.exceptions.ConnectionError:
        return "Error: Connection refused. Is the FastAPI server running?"
    except Exception as e:
        return f"Error: {e}"

def run_memory_evaluations():
    print(f"Starting Memory Evaluation run against {BASE_URL}...\n")
    results = {}
    
    for scenario_name, prompts in SCENARIOS.items():
        print(f"\n{'='*50}\nTesting {scenario_name}\n{'='*50}")
        session_id = str(uuid.uuid4())
        results[scenario_name] = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nUser: {prompt}")
            response = query_chatbot(prompt, session_id)
            
            # Print truncated response for readability
            print_resp = response[:150] + "..." if len(response) > 150 else response
            print_resp = print_resp.replace('\n', ' ')
            print(f"Bot:  {print_resp}")
            
            results[scenario_name].append({
                "turn": i,
                "user": prompt,
                "bot": response
            })
            
            if "Connection refused" in response:
                print("\n[!] Aborting evaluation: Could not connect to the backend.")
                return
                
            time.sleep(1)
            
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation complete. Full results saved to {os.path.abspath(RESULTS_FILE)}")

if __name__ == "__main__":
    run_memory_evaluations()
