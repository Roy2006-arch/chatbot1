import re
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CodingIntent:
    is_coding: bool
    sub_intent: str  # 'direct_solution', 'debugging', 'explanation', 'optimization', 'conversion', 'algorithm', 'none'
    is_coding_challenge: bool

class CodingIntentClassifier:
    """
    Detects specific coding intents to tailor the response strategy.
    Sub-intents: direct_solution, debugging, explanation, optimization, conversion, algorithm
    """
    def __init__(self):
        self.patterns = {
            "direct_solution": [
                r"\b(complete the function|give (me )?(the )?solution|solve this|write code for|implement this)\b",
                r"\b(hackerrank|leetcode|codeforces|codewars)\b",
                r"(sample input|sample output|constraints:)",
                r"\b(write a program|return the result)\b"
            ],
            "debugging": [
                r"\b(fix this( code)?|bug|error|exception|not working|traceback|why is this failing|diagnose)\b",
                r"\b(syntax error|runtime error|logic error)\b"
            ],
            "explanation": [
                r"\b(explain this code|how does this work|walk me through|what does this do)\b",
                r"\b(step by step explanation)\b"
            ],
            "optimization": [
                r"\b(optimize|make it faster|time complexity|space complexity|improve performance|reduce memory|more efficient)\b",
                r"\b(o\(n\)|o\(1\)|o\(log n\)|o\(n\^2\))\b"
            ],
            "conversion": [
                r"\b(convert this to|translate this to|rewrite in|port to)\b",
                r"\b(from python to|from java to|from cpp to|from javascript to)\b"
            ],
            "algorithm": [
                r"\b(how does (the )?algorithm work|explain (the )?algorithm|concept behind|theory behind)\b",
                r"\b(bfs|dfs|dynamic programming|greedy|backtracking|divide and conquer)\b"
            ]
        }
        
    def classify(self, text: str) -> CodingIntent:
        text_lower = text.lower()
        
        # Check for indicators of an incomplete snippet or constraints
        has_snippet = bool(re.search(r"(def \w+\(.*pass\b|\btodo\b|\.\.\.)", text_lower))
        has_challenge = bool(re.search(r"(leetcode|hackerrank|constraints:|sample input|sample output)", text_lower))
        
        scores = {k: 0.0 for k in self.patterns.keys()}
        
        for intent, regex_list in self.patterns.items():
            for pattern in regex_list:
                if re.search(pattern, text_lower):
                    scores[intent] += 1.0
                    
        # Add weights for challenge indicators
        if has_challenge or has_snippet:
            scores["direct_solution"] += 1.5
            
        best_intent_tuple = max(scores.items(), key=lambda x: x[1])
        best_intent = best_intent_tuple[0]
        max_score = best_intent_tuple[1]
        
        is_coding = max_score > 0
        if not is_coding:
            return CodingIntent(is_coding=False, sub_intent="none", is_coding_challenge=False)
            
        return CodingIntent(
            is_coding=True,
            sub_intent=best_intent,
            is_coding_challenge=has_challenge or (best_intent == "direct_solution" and has_snippet)
        )
