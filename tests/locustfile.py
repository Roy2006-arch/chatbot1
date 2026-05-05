import time
import json
import uuid
from locust import HttpUser, task, between, events

class ChatbotUser(HttpUser):
    # Simulate a user thinking for 5 to 15 seconds before sending a new message
    wait_time = between(5, 15)

    @task(1)
    def health_check(self):
        """Phase 1: Basic API Latency Check"""
        self.client.get("/", name="Health Check")

    @task(3)
    def chat_stream(self):
        """Phase 2 & 3: Model Inference & Streaming Metrics"""
        session_id = str(uuid.uuid4())
        payload = {
            "message": "Explain the concept of quantum computing in one short paragraph.",
            "session_id": session_id
        }

        start_time = time.time()
        first_token_time = None
        
        # Start the request, timing the overall connection
        with self.client.post("/chat/stream", json=payload, stream=True, catch_response=True, name="Chat Stream") as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}")
                return

            try:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            # Record Time-To-First-Token
                            if not first_token_time:
                                first_token_time = time.time() - start_time
                                # Custom event logging for TTFT
                                events.request.fire(
                                    request_type="STREAM",
                                    name="Time To First Token",
                                    response_time=first_token_time * 1000,
                                    response_length=0,
                                    exception=None,
                                )
                                
                            data_str = decoded_line.replace("data: ", "").strip()
                            if data_str == "[DONE]":
                                break
                response.success()
            except Exception as e:
                response.failure(f"Stream parsing failed: {e}")
