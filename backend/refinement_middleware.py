"""
refinement_middleware.py
========================
A 7-step pipeline to intercept and polish chatbot responses.

Pipeline steps:
1. Analyze user intent
2. Determine ideal response length
3. Generate draft response (intercepted)
4. Remove repetitive content
5. Remove unrelated sentences
6. Improve clarity and tone
7. Return final polished response
"""

import re
from typing import List

class ResponseRefinementMiddleware:
    def __init__(self):
        # Configuration for intents and ideal max lengths (in sentences)
        self.intent_rules = {
            "greeting": {"max_sentences": 1},
            "factual": {"max_sentences": 3},
            "coding": {"max_sentences": 10},  # Code blocks have more leeway
            "opinion": {"max_sentences": 4},
            "general": {"max_sentences": 3}
        }
        
        self.robotic_phrases = [
            r'As an AI(?: language model)?[,.]?\s*',
            r'I am an AI[,.]?\s*',
            r'I can certainly help with that[!.]?\s*',
            r'Here is the information you requested:?\s*',
            r'Let me know if you need anything else[!.]?\s*',
            r'I hope this helps[!.]?\s*',
            r'Feel free to ask if you have more questions[!.]?\s*',
            r'Certainly[!.]?\s*',
            r'Of course[!.]?\s*',
            r'Absolutely[!.]?\s*',
        ]

    def _analyze_intent(self, user_prompt: str) -> str:
        """1. Analyze user intent based on keywords."""
        prompt = user_prompt.lower()
        if any(word in prompt for word in ['hi', 'hello', 'hey', 'greetings']):
            return 'greeting'
        elif any(word in prompt for word in ['code', 'python', 'script', 'function', 'bug', 'fix']):
            return 'coding'
        elif any(word in prompt for word in ['what', 'who', 'when', 'where', 'fact', 'define']):
            return 'factual'
        elif any(word in prompt for word in ['think', 'opinion', 'recommend', 'should']):
            return 'opinion'
        return 'general'

    def _determine_ideal_length(self, intent: str) -> int:
        """2. Determine ideal response length in sentences."""
        return self.intent_rules.get(intent, self.intent_rules["general"])["max_sentences"]

    def _remove_repetitive_content(self, text: str) -> str:
        """4. Remove repetitive content."""
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        seen = set()
        unique_sentences = []
        for s in sentences:
            # Normalize sentence for similarity check
            normalized = re.sub(r'[^a-z0-9]', '', s.lower())
            if not normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(s)
        return ' '.join(unique_sentences)

    def _remove_unrelated_sentences(self, sentences: List[str], intent: str) -> List[str]:
        """5. Remove unrelated sentences (prevent multiple unnecessary answers & overexplaining)."""
        max_sentences = self._determine_ideal_length(intent)
        
        # We don't truncate coding intent aggressively as it may contain blocks
        if intent == 'coding':
            return sentences
            
        return sentences[:max_sentences]

    def _improve_clarity_and_tone(self, text: str) -> str:
        """6. Improve clarity and tone (prevent robotic wording)."""
        refined_text = text
        for phrase in self.robotic_phrases:
            refined_text = re.sub(phrase, '', refined_text, flags=re.IGNORECASE)
            
        # Clean up extra spaces, floating punctuation, and empty lines
        refined_text = re.sub(r'\s+', ' ', refined_text).strip()
        refined_text = re.sub(r'^\s*[,.]\s*', '', refined_text) # remove leading commas/dots
        
        # Ensure capitalization
        if refined_text and not refined_text[0].isupper():
            refined_text = refined_text[0].upper() + refined_text[1:]
            
        return refined_text

    def refine_response(self, user_prompt: str, draft_response: str) -> str:
        """
        Executes the 7-step pipeline.
        Step 3 (Generate draft response) is represented by `draft_response`.
        """
        # 1. Analyze user intent
        intent = self._analyze_intent(user_prompt)
        
        # 2. Determine ideal response length happens internally during step 5
        
        # 4. Remove repetitive content
        text_no_reps = self._remove_repetitive_content(draft_response)
        
        # Split into sentences for the next step, preserving newlines mostly if needed
        # For simplicity, we split on sentence terminators followed by space
        sentences = re.split(r'(?<=[.!?]) +', text_no_reps.strip())
        
        # 5. Remove unrelated sentences / Enforce length constraints
        focused_sentences = self._remove_unrelated_sentences(sentences, intent)
        focused_text = ' '.join(focused_sentences)
        
        # 6. Improve clarity and tone
        final_response = self._improve_clarity_and_tone(focused_text)
        
        # 7. Return final polished response
        return final_response

# Example Usage
if __name__ == "__main__":
    middleware = ResponseRefinementMiddleware()
    
    examples = [
        {
            "prompt": "What is the capital of France?",
            "draft": "As an AI language model, I can certainly help with that! The capital of France is Paris. Paris is a beautiful city known for the Eiffel Tower and the Louvre. Paris is the capital city of France. I hope this helps! Let me know if you need anything else."
        },
        {
            "prompt": "Hello there!",
            "draft": "Hello! I am an AI. How may I assist you today? I can answer questions, write code, or just chat."
        },
        {
            "prompt": "Write a Python function to add two numbers.",
            "draft": "Certainly! Here is the information you requested. To add two numbers in Python, you can use the + operator. You can use the + operator to add them. Here is the code: def add(a, b): return a + b. This function adds a and b together and returns the result. Feel free to ask if you have more questions."
        }
    ]
    
    print("-" * 50)
    for i, ex in enumerate(examples, 1):
        print(f"Example {i}")
        print(f"User Prompt   : {ex['prompt']}")
        print(f"Draft Response: {ex['draft']}")
        print(f"Refined       : {middleware.refine_response(ex['prompt'], ex['draft'])}")
        print("-" * 50)
