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
    "bullet_points": re.compile(r"(?:in )?(?:a )?(?:bulleted? )?(?:list|format) with (\d+) (?:points?|items?|entries?)", re.I),
    "max_characters": re.compile(r"(?:no |not |at )?more? than (\d+) characters?", re.I),
    "min_words": re.compile(r"at (?:least|minimum) (\d+) words?", re.I),
    "max_lines": re.compile(r"(?:no |not |at )?more? than (\d+) lines?", re.I),
    "exactly_words": re.compile(r"(?:exactly|precisely) (\d+) words?", re.I),
    "no_code": re.compile(r"\b(?:without|no|don'?t|do not) (?:use |include )?(?:any )?code\b", re.I),
    "code_only": re.compile(r"\b(?:only|just|purely|exclusively) (?:the |show )?code\b", re.I),
    "numbered_list": re.compile(r"(?:in )?(?:a )?(?:numbered? )?(?:list|format) with (\d+)", re.I),
    "one_line": re.compile(r"\b(?:in )?one line\b", re.I),
    "max_paragraphs": re.compile(r"(?:no |not |at )?more? than (\d+) paragraphs?", re.I),
    "formal_tone": re.compile(r"\b(?:professional|formal|business|corporate)\b", re.I),
    "casual_tone": re.compile(r"\b(?:casual|informal|friendly|conversational|colloquial)\b", re.I),
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
    paragraphs = [p for p in response.split('\n\n') if p.strip()]
    lines = [l for l in response.split('\n') if l.strip()]

    if "single_word" in constraints:
        expected = constraints["single_word"]
        if response.strip().lower().rstrip(".!?").strip() != expected:
            violations.append(f"Response should be only the word '{expected}' but got multiple words")
    if "max_words" in constraints:
        limit = constraints["max_words"]
        if len(words) > limit:
            violations.append(f"Response has {len(words)} words, expected at most {limit}")
    if "min_words" in constraints:
        limit = constraints["min_words"]
        if len(words) < limit:
            violations.append(f"Response has {len(words)} words, expected at least {limit}")
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
    if "max_characters" in constraints:
        limit = constraints["max_characters"]
        if len(response) > limit:
            violations.append(f"Response has {len(response)} characters, expected at most {limit}")
    if "max_lines" in constraints:
        limit = constraints["max_lines"]
        if len(lines) > limit:
            violations.append(f"Response has {len(lines)} lines, expected at most {limit}")
    if "exactly_words" in constraints:
        count = constraints["exactly_words"]
        if len(words) != count:
            violations.append(f"Response has {len(words)} words, expected exactly {count}")
    if "no_code" in constraints:
        if "```" in response or re.search(r'\b(def|class|function|import|from|const|let|var|SELECT)\b', response):
            violations.append("Response contains code but 'no code' was requested")
    if "code_only" in constraints:
        if "```" not in response and not re.search(r'\b(def|class|function|import|from|const|let|var)\b', response):
            violations.append("Response should contain only code")
    if "numbered_list" in constraints:
        limit = constraints["numbered_list"]
        numbered = re.findall(r'(?:^|\n)\s*\d+[.)]\s*', response)
        if numbered and len(numbered) != limit:
            violations.append(f"Response has {len(numbered)} numbered items, expected {limit}")
    if "one_line" in constraints:
        if len([l for l in response.split('\n') if l.strip()]) > 1:
            violations.append("Response should be on one line")
    if "max_paragraphs" in constraints:
        limit = constraints["max_paragraphs"]
        if len(paragraphs) > limit:
            violations.append(f"Response has {len(paragraphs)} paragraphs, expected at most {limit}")
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
    if "exactly_words" in constraints:
        count = constraints["exactly_words"]
        words = response.split()
        if len(words) > count:
            response = " ".join(words[:count])
            modified = True
        elif len(words) < count:
            response = response + " " + " ".join(["..."] * (count - len(words)))
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
    if "max_characters" in constraints:
        limit = constraints["max_characters"]
        if len(response) > limit:
            response = response[:limit]
            modified = True
    if "max_lines" in constraints:
        limit = constraints["max_lines"]
        lines = response.split('\n')
        if len([l for l in lines if l.strip()]) > limit:
            non_empty = [l for l in lines if l.strip()]
            response = '\n'.join(non_empty[:limit])
            modified = True
    if "one_line" in constraints:
        lines = [l for l in response.split('\n') if l.strip()]
        if len(lines) > 1:
            response = ' '.join(lines)
            modified = True
    if "max_paragraphs" in constraints:
        limit = constraints["max_paragraphs"]
        paragraphs = [p for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) > limit:
            response = '\n\n'.join(paragraphs[:limit])
            modified = True
    if "numbered_list" in constraints:
        limit = constraints["numbered_list"]
        numbered = re.findall(r'(?:^|\n)\s*\d+[.)]\s*', response)
        if numbered and len(numbered) > limit:
            lines = response.split('\n')
            kept_lines = []
            count = 0
            for line in lines:
                if re.match(r'\s*\d+[.)]\s', line):
                    count += 1
                    if count <= limit:
                        kept_lines.append(line)
                else:
                    kept_lines.append(line)
            response = '\n'.join(kept_lines)
            modified = True

    return response, modified
