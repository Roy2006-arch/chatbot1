import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("chatbot.constraint_validator")

_PATTERNS = {
    "exact_count": re.compile(r"list exactly (\d+)", re.I),
    "max_items": re.compile(r"(?:no |not |at )?more? than (\d+)", re.I),
    "max_words": re.compile(r"(?:no |not |at )?more? than (\d+) words?", re.I),
    "max_sentences": re.compile(r"(?:no |not |at )?more? than (\d+) sentenc", re.I),
    "single_word": re.compile(r"reply with only (?:the |a |an |single )?(?:word )?[\"']?(\w+)[\"']?", re.I),
    "exactly_sentences": re.compile(r"exactly (\d+) sentenc", re.I),
    "concise": re.compile(r"\bconcise\b|\bshort\b|\bbrief\b|\bquick\b", re.I),
}

def detect_constraints(prompt: str) -> Dict[str, object]:
    constraints = {}
    for key, pattern in _PATTERNS.items():
        m = pattern.search(prompt)
        if m:
            if key == "single_word":
                constraints[key] = m.group(1).lower()
            elif key in ("max_words", "max_sentences", "exact_count", "max_items", "exactly_sentences"):
                constraints[key] = int(m.group(1))
            elif key == "concise":
                constraints[key] = True
    return constraints

def check_constraints(response: str, constraints: Dict[str, object]) -> List[str]:
    violations = []
    words = response.split()
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    items = re.findall(r'(?:^|\n)\s*(?:\d+[.)]\s*|[-*]\s*)(.+?)(?=\n|$)', response)

    if "single_word" in constraints:
        expected = constraints["single_word"]
        if response.strip().lower().rstrip(".!?").strip() != expected:
            violations.append(f"Response should be only the word '{expected}' but got multiple words")
    if "max_words" in constraints:
        limit = constraints["max_words"]
        if len(words) > limit:
            violations.append(f"Response has {len(words)} words, expected at most {limit}")
    if "max_sentences" in constraints:
        limit = constraints["max_sentences"]
        if len(sentences) > limit:
            violations.append(f"Response has {len(sentences)} sentences, expected at most {limit}")
    if "exactly_sentences" in constraints:
        count = constraints["exactly_sentences"]
        if len(sentences) != count:
            violations.append(f"Response has {len(sentences)} sentences, expected exactly {count}")
    if "exact_count" in constraints:
        limit = constraints["exact_count"]
        if items and len(items) != limit:
            violations.append(f"Response has {len(items)} items, expected exactly {limit}")
    if "concise" in constraints and len(words) > 50:
        violations.append(f"Requested concise/short/brief but response is {len(words)} words")
    return violations

def enforce_constraints(response: str, constraints: Dict[str, object]) -> Tuple[str, bool]:
    modified = False
    if "single_word" in constraints:
        expected = constraints["single_word"]
        response = expected
        modified = True
    if "max_words" in constraints:
        limit = constraints["max_words"]
        words = response.split()
        if len(words) > limit:
            response = " ".join(words[:limit])
            modified = True
    if "exactly_sentences" in constraints:
        count = constraints["exactly_sentences"]
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        if len(sentences) > count:
            response = " ".join(sentences[:count])
            modified = True
    if "exact_count" in constraints:
        limit = constraints["exact_count"]
        items = re.findall(r'(?:^|\n)\s*(?:\d+[.)]\s*|[-*]\s*)(.+?)(?=\n|$)', response)
        if items:
            bare_items = [re.sub(r'^.*?[.:] ', '', item).strip() for item in items]
            if len(bare_items) > limit:
                kept = items[:limit]
                response_lines = response.split("\n")
                kept_indices = set()
                for item in kept:
                    for i, line in enumerate(response_lines):
                        if item in line:
                            kept_indices.add(i)
                            break
                modified = True

    return response, modified
