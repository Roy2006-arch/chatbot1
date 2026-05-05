import requests
import json
import uuid
import time
import sys

BASE_URL = "http://localhost:8000"

def print_result(test_name, passed, msg=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} | {test_name} {msg}")

def test_health_check():
    try:
        response = requests.get(f"{BASE_URL}/")
        passed = response.status_code == 200 and "status" in response.json()
        print_result("Health Check Endpoint", passed)
    except Exception as e:
        print_result("Health Check Endpoint", False, f"- Error: {e}")

def test_chat_stream():
    session_id = str(uuid.uuid4())
    payload = {"message": "Say 'hello world'", "session_id": session_id}
    
    try:
        start_time = time.time()
        # Note: stream=True is required to read SSE properly
        response = requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True)
        
        if response.status_code != 200:
            print_result("Chat Stream Endpoint", False, f"- Status code {response.status_code}")
            return
            
        first_token_time = None
        full_response = ""
        
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
                        
        passed = len(full_response) > 0
        ttft_msg = f"- TTFT: {first_token_time:.2f}s" if first_token_time else ""
        print_result("Chat Stream Endpoint", passed, ttft_msg)
        
    except Exception as e:
        print_result("Chat Stream Endpoint", False, f"- Error: {e}")

def test_empty_message():
    session_id = str(uuid.uuid4())
    payload = {"message": "", "session_id": session_id}
    
    try:
        # We expect a 400 Bad Request or similar, depending on how your backend handles it
        response = requests.post(f"{BASE_URL}/chat/stream", json=payload)
        
        # If it returns 200 and tries to generate, that might be considered a failure in strict testing
        passed = response.status_code in [400, 422]
        msg = f"- Status: {response.status_code}"
        if response.status_code == 200:
            msg += " (Warning: Backend accepted an empty message!)"
        print_result("Empty Message Validation", passed, msg)
    except Exception as e:
        print_result("Empty Message Validation", False, f"- Error: {e}")

if __name__ == "__main__":
    print("Starting Automated API Tests...\n")
    test_health_check()
    test_empty_message()
    test_chat_stream()
    print("\nTests completed.")
