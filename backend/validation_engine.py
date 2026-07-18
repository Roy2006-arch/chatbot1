import re
import ast
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from backend.url_verifier import URLVerifier

logger = logging.getLogger("validation_engine")


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str
    suggested_repair: Optional[str] = None
    section_index: Optional[int] = None


@dataclass
class ValidationReport:
    is_valid: bool
    issues: List[ValidationIssue]
    repaired_text: str
    needs_regeneration: bool = False


class ResponseValidationEngine:
    def __init__(self):
        self.complexity_patterns = {
            "O(1)": r"\bO\(1\)\b",
            "O(log n)": r"\bO\(log\s*n\)\b",
            "O(n)": r"\bO\(n\)\b",
            "O(n log n)": r"\bO\(n\s*log\s*n\)\b",
            "O(n^2)": r"\bO\(n\^2\)\b",
        }
        self.url_verifier = URLVerifier()

    def validate(self, text: str, context: str = "", expected_mode: str = "normal") -> ValidationReport:
        issues = []
        repaired_text = text

        markdown_issues, repaired_text = self._validate_markdown(repaired_text)
        issues.extend(markdown_issues)

        code_issues, repaired_text = self._validate_code(repaired_text)
        issues.extend(code_issues)

        reasoning_issues = self._validate_reasoning(repaired_text)
        issues.extend(reasoning_issues)

        complexity_issues = self._validate_complexity(repaired_text)
        issues.extend(complexity_issues)

        content_issues = self._validate_content(repaired_text, context)
        issues.extend(content_issues)

        url_issues = self._validate_urls(repaired_text)
        issues.extend(url_issues)

        needs_regeneration = any(issue.severity == 'critical' for issue in issues) or \
            any(issue.code == 'unfinished_reasoning' for issue in issues)

        is_valid = len([i for i in issues if i.severity in ['critical', 'warning']]) == 0

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            repaired_text=repaired_text,
            needs_regeneration=needs_regeneration,
        )

    def _validate_markdown(self, text: str) -> Tuple[List[ValidationIssue], str]:
        issues = []
        repaired = text

        if text.count("```") % 2 != 0:
            issues.append(ValidationIssue(
                code="unclosed_markdown",
                message="Unclosed markdown code block detected.",
                severity="warning",
                suggested_repair="Append closing backticks.",
            ))
            repaired += "\n```"

        bold_matches = re.findall(r"\*\*", text)
        if len(bold_matches) % 2 != 0:
            if not (text.rstrip().endswith("**") or text.rstrip().endswith("*")):
                issues.append(ValidationIssue(
                    code="unclosed_formatting",
                    message="Unclosed bold formatting (**).",
                    severity="info",
                ))

        return issues, repaired

    def _validate_code(self, text: str) -> Tuple[List[ValidationIssue], str]:
        issues = []
        repaired = text

        # Match both closed and unclosed code blocks
        closed_blocks = re.findall(r"```(\w*)\n(.*?)\n```", text, re.DOTALL)
        unclosed_match = re.search(r"```(\w*)\n(.*?)(?:\n```|$)", text, re.DOTALL)
        all_blocks = list(closed_blocks)

        if unclosed_match:
            lang = unclosed_match.group(1)
            code = unclosed_match.group(2)
            # Only add if this is actually unclosed (not already captured)
            if not any(c[0] == lang and c[1] == code for c in closed_blocks):
                all_blocks.append((lang, code))

        for lang, code in all_blocks:
            lang = lang.lower()

            if lang in ["python", "py", ""]:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    if "unexpected EOF" in str(e) or "was never closed" in str(e):
                        issues.append(ValidationIssue(
                            code="incomplete_code",
                            message=f"Python code seems truncated: {str(e)}",
                            severity="critical",
                        ))
                    else:
                        issues.append(ValidationIssue(
                            code="syntax_error",
                            message=f"Python syntax error: {str(e)}",
                            severity="critical",
                        ))

            lang_brackets = {
                "javascript": ("{", "}"), "js": ("{", "}"),
                "typescript": ("{", "}"), "ts": ("{", "}"),
                "java": ("{", "}"),
                "cpp": ("{", "}"), "c++": ("{", "}"),
                "go": ("{", "}"),
                "rust": ("{", "}"),
                "c#": ("{", "}"), "cs": ("{", "}"),
            }

            if lang in lang_brackets:
                open_b, close_b = lang_brackets[lang]
                if code.count(open_b) > code.count(close_b):
                    issues.append(ValidationIssue(
                        code="unclosed_block",
                        message=f"Unclosed '{open_b}' detected in {lang} block.",
                        severity="warning",
                        suggested_repair=f"Append closing '{close_b}'.",
                    ))

            trimmed_code = code.strip()
            truncation_indicators = (".", "+", "-", "*", "/", "=", "and", "or", ":", ",", "&&", "||", "<<", ">>")
            if trimmed_code.endswith(truncation_indicators):
                issues.append(ValidationIssue(
                    code="truncated_logic",
                    message="Code block ends abruptly, suggesting it was cut off during generation.",
                    severity="critical",
                ))

            for quote in ["'", '"', '"""', "'''"]:
                if trimmed_code.count(quote) % 2 != 0:
                    if len(re.findall(f"{quote}", trimmed_code[-20:])) > 0:
                        issues.append(ValidationIssue(
                            code="unclosed_string",
                            message=f"Unclosed string literal ({quote}) detected near end of code.",
                            severity="warning",
                        ))

            if lang in ["python", "py"]:
                if re.search(r"(def|class)\s+\w+.*:\s*$", trimmed_code):
                    issues.append(ValidationIssue(
                        code="empty_block_header",
                        message="Python block ends with a header but no implementation.",
                        severity="critical",
                    ))

        return issues, repaired

    def _validate_reasoning(self, text: str) -> List[ValidationIssue]:
        issues = []

        has_start = "<thought>" in text.lower()
        has_end = "</thought>" in text.lower()

        if has_start and not has_end:
            issues.append(ValidationIssue(
                code="unfinished_reasoning",
                message="Internal reasoning block was started but never finished.",
                severity="critical",
            ))

        if has_start and has_end:
            thought_content = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL | re.IGNORECASE)
            if thought_content and len(thought_content.group(1).strip()) < 10:
                issues.append(ValidationIssue(
                    code="hollow_reasoning",
                    message="Reasoning block is suspiciously short.",
                    severity="warning",
                ))

        return issues

    def _validate_complexity(self, text: str) -> List[ValidationIssue]:
        issues = []
        code_blocks = re.findall(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)

        if not code_blocks:
            return issues

        stated = None
        for label, pattern in self.complexity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                stated = label
                break

        if stated:
            code = code_blocks[0]
            lines = code.split("\n")
            max_depth = 0
            current_depth = 0
            for line in lines:
                indent = len(line) - len(line.lstrip())
                if re.search(r"\b(for|while)\b", line):
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif line.strip() == "" or (line.strip() and indent == 0 and not line.startswith((" ", "\t"))):
                    current_depth = 0

            is_recursive = False
            func_match = re.search(r"def (\w+)\(", code)
            if func_match:
                name = func_match.group(1)
                if re.search(rf"\b{name}\(", code[func_match.end():]):
                    is_recursive = True

            if re.search(r"while\s+True:", code) and "break" not in code:
                issues.append(ValidationIssue(
                    code="infinite_loop",
                    message="Potential infinite loop detected (while True without break).",
                    severity="critical",
                ))

            if is_recursive and "O(1)" in stated:
                issues.append(ValidationIssue(
                    code="complexity_mismatch",
                    message=f"Stated {stated} but code is recursive (likely logarithmic or linear).",
                    severity="warning",
                ))
            elif max_depth >= 2 and stated == "O(n)":
                issues.append(ValidationIssue(
                    code="complexity_mismatch",
                    message=f"Stated {stated} but code has nested loops (likely O(n^2)).",
                    severity="warning",
                ))

        return issues

    def _validate_content(self, text: str, context: str) -> List[ValidationIssue]:
        issues = []

        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 5:
            seen = set()
            repeated_count = 0
            for s in sentences:
                if len(s) < 20:
                    continue
                normalized = re.sub(r'\W+', '', s.lower())
                if normalized in seen:
                    repeated_count += 1
                seen.add(normalized)

            if repeated_count > 1:
                issues.append(ValidationIssue(
                    code="textual_repetition",
                    message="Significant textual repetition detected.",
                    severity="warning",
                ))

        if re.search(r'\b(user:|assistant:|system:|human:|<\|user\|>|<\|assistant\|>|<\|system\|>)\b', text, re.IGNORECASE):
            issues.append(ValidationIssue(
                code="hallucination_leak",
                message="Internal conversation tokens detected in response.",
                severity="critical",
            ))

        if re.search(r"\bas an ai language model\b|\bi don't have feelings\b|\bi cannot feel\b|\bi am just a\b|\bi am an ai\b", text, re.IGNORECASE):
            issues.append(ValidationIssue(
                code="robotic_artifact",
                message="Robotic AI disclaimer detected.",
                severity="info",
            ))

        if re.search(r'\b(I\'m not sure|I cannot|I can\'t|I don\'t know)\b', text, re.IGNORECASE):
            if not re.search(r'\b(I\'m not sure about this specific|I cannot verify|I can\'t confirm|I don\'t have access)\b', text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    code="uncertainty_artifact",
                    message="Excessive uncertainty markers detected. Consider being more direct.",
                    severity="info",
                ))

        words = text.split()
        if len(words) > 3:
            word_freq = {}
            for w in words:
                w_lower = w.lower().strip('.,!?;:()"\'')
                if len(w_lower) > 3:
                    word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
            overused = [w for w, c in word_freq.items() if c > len(words) * 0.08 and c > 3]
            if overused:
                issues.append(ValidationIssue(
                    code="word_repetition",
                    message=f"Words repeated excessively: {', '.join(overused[:3])}",
                    severity="info",
                ))

        if re.search(r'\*\*[^*]+\*\*\s*\*\*[^*]+\*\*', text):
            issues.append(ValidationIssue(
                code="excessive_formatting",
                message="Excessive bold formatting detected.",
                severity="info",
            ))

        return issues

    def _validate_urls(self, text: str) -> List[ValidationIssue]:
        issues = []
        urls = self.url_verifier.extract_urls(text)
        if not urls:
            return issues

        for url in urls:
            if not self.url_verifier.validate_format(url):
                issues.append(ValidationIssue(
                    code="invalid_url_format",
                    message=f"Invalid URL format detected: {url}",
                    severity="warning",
                    suggested_repair=f"Wrap {url} in backticks or remove it.",
                ))

        return issues

    def get_affected_section(self, text: str, issue: ValidationIssue) -> str:
        if "code" in issue.code:
            blocks = re.findall(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)
            if blocks:
                return blocks[0][-100:]
        sentences = re.split(r'(?<=[.!?]) +', text)
        return " ".join(sentences[-2:])

    def repair(self, text: str, report: ValidationReport) -> str:
        result = report.repaired_text

        for issue in report.issues:
            if issue.code == "hallucination_leak":
                result = re.sub(
                    r'\b(user:|assistant:|system:|human:|<\|user\|>|<\|assistant\|>|<\|system\|>)\b.*',
                    '', result, flags=re.IGNORECASE,
                )

            if issue.code == "textual_repetition":
                sentences = re.split(r'(?<=[.!?]) +', result)
                seen = set()
                final_sentences = []
                for s in sentences:
                    norm = re.sub(r'\W+', '', s.lower())
                    if norm not in seen:
                        seen.add(norm)
                        final_sentences.append(s)
                result = ' '.join(final_sentences)

        return result.strip()
