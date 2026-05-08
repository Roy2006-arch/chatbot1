import re
import ast
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger("validation_engine")

@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str  # 'critical', 'warning', 'info'
    suggested_repair: Optional[str] = None
    section_index: Optional[int] = None

@dataclass
class ValidationReport:
    is_valid: bool
    issues: List[ValidationIssue]
    repaired_text: str
    needs_regeneration: bool = False

class ResponseValidationEngine:
    """
    Advanced response validation engine for AI chatbot.
    Detects incomplete code, syntax errors, unclosed markdown, hallucinations,
    repeated text, unfinished reasoning, and invalid complexity analysis.
    """

    def __init__(self):
        self.complexity_patterns = {
            "O(1)": r"\bO\(1\)\b",
            "O(log n)": r"\bO\(log\s*n\)\b",
            "O(n)": r"\bO\(n\)\b",
            "O(n log n)": r"\bO\(n\s*log\s*n\)\b",
            "O(n^2)": r"\bO\(n\^2\)\b",
            "O(2^n)": r"\bO\(2\^n\)\b"
        }

    def validate(self, text: str, context: str = "", expected_mode: str = "normal") -> ValidationReport:
        issues = []
        repaired_text = text
        
        # 1. Markdown Integrity Validator
        markdown_issues, repaired_text = self._validate_markdown(repaired_text)
        issues.extend(markdown_issues)

        # 2. Syntax-Aware Parsing & Code Completeness
        code_issues, repaired_text = self._validate_code(repaired_text)
        issues.extend(code_issues)

        # 3. Reasoning Consistency Checker
        reasoning_issues = self._validate_reasoning(repaired_text)
        issues.extend(reasoning_issues)

        # 4. Complexity Validator
        complexity_issues = self._validate_complexity(repaired_text)
        issues.extend(complexity_issues)

        # 5. Redundancy & Hallucination Check
        content_issues = self._validate_content(repaired_text, context)
        issues.extend(content_issues)

        # Determine if regeneration is needed
        # Critical syntax errors or massive hallucinations usually need regeneration
        needs_regeneration = any(issue.severity == 'critical' for issue in issues) or \
                             any(issue.code == 'unfinished_reasoning' for issue in issues)

        is_valid = len([i for i in issues if i.severity in ['critical', 'warning']]) == 0

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            repaired_text=repaired_text,
            needs_regeneration=needs_regeneration
        )

    def _validate_markdown(self, text: str) -> Tuple[List[ValidationIssue], str]:
        issues = []
        repaired = text
        
        # Unclosed code blocks
        if text.count("```") % 2 != 0:
            issues.append(ValidationIssue(
                code="unclosed_markdown",
                message="Unclosed markdown code block detected.",
                severity="warning",
                suggested_repair="Append closing backticks."
            ))
            repaired += "\n```"

        # Unclosed bold/italic
        bold_matches = re.findall(r"\*\*", text)
        if len(bold_matches) % 2 != 0:
            # Only repair if it's at the very end, otherwise it might be a mess
            if text.rstrip().endswith("**") or text.rstrip().endswith("*"):
                pass # Already likely intentional or just a single star
            else:
                issues.append(ValidationIssue(
                    code="unclosed_formatting",
                    message="Unclosed bold formatting (**).",
                    severity="info"
                ))

        return issues, repaired

    def _validate_code(self, text: str) -> Tuple[List[ValidationIssue], str]:
        issues = []
        repaired = text
        
        # Extract code blocks with language tags
        blocks = re.findall(r"```(\w*)\n(.*?)\n```", text, re.DOTALL)
        
        for lang, code in blocks:
            lang = lang.lower()
            
            # 1. Syntax Check
            if lang in ["python", "py", ""]:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    # Check if it's just a missing closing paren/brace which we can guess
                    if "unexpected EOF" in str(e) or "was never closed" in str(e):
                        issues.append(ValidationIssue(
                            code="incomplete_code",
                            message=f"Python code seems truncated: {str(e)}",
                            severity="critical"
                        ))
                    else:
                        issues.append(ValidationIssue(
                            code="syntax_error",
                            message=f"Python syntax error: {str(e)}",
                            severity="critical"
                        ))
            
            # 2. General Completeness Heuristics (JS, C++, Java, etc.)
            lang_brackets = {
                "javascript": ("{", "}"), "js": ("{", "}"),
                "typescript": ("{", "}"), "ts": ("{", "}"),
                "java": ("{", "}"),
                "cpp": ("{", "}"), "c++": ("{", "}"),
                "go": ("{", "}"),
                "rust": ("{", "}"),
                "c#": ("{", "}"), "cs": ("{", "}")
            }
            
            if lang in lang_brackets:
                open_b, close_b = lang_brackets[lang]
                if code.count(open_b) > code.count(close_b):
                    issues.append(ValidationIssue(
                        code="unclosed_block",
                        message=f"Unclosed '{open_b}' detected in {lang} block.",
                        severity="warning",
                        suggested_repair=f"Append closing '{close_b}'."
                    ))

            # 3. Truncation detection (trailing operators or colons)
            trimmed_code = code.strip()
            # Language-specific trailing characters that indicate incompleteness
            truncation_indicators = (".", "+", "-", "*", "/", "=", "and", "or", ":", ",", "&&", "||", "<<", ">>")
            if trimmed_code.endswith(truncation_indicators):
                issues.append(ValidationIssue(
                    code="truncated_logic",
                    message="Code block ends abruptly, suggesting it was cut off during generation.",
                    severity="critical"
                ))
            
            # Check for unclosed strings
            for quote in ["'", '"', '"""', "'''"]:
                if trimmed_code.count(quote) % 2 != 0:
                    # Heuristic: only flag if it's near the end
                    if len(re.findall(f"{quote}", trimmed_code[-20:])) > 0:
                        issues.append(ValidationIssue(
                            code="unclosed_string",
                            message=f"Unclosed string literal ({quote}) detected near end of code.",
                            severity="warning"
                        ))
            
            # Check for unclosed function/class headers
            if lang in ["python", "py"]:
                if re.search(r"(def|class)\s+\w+.*:\s*$", trimmed_code):
                     issues.append(ValidationIssue(
                        code="empty_block_header",
                        message="Python block ends with a header but no implementation.",
                        severity="critical"
                    ))

        return issues, repaired

    def _validate_reasoning(self, text: str) -> List[ValidationIssue]:
        issues = []
        
        # Check <thought> tags
        has_start = "<thought>" in text.lower()
        has_end = "</thought>" in text.lower()
        
        if has_start and not has_end:
            issues.append(ValidationIssue(
                code="unfinished_reasoning",
                message="Internal reasoning block was started but never finished.",
                severity="critical"
            ))
            
        # Check for empty thought blocks
        if has_start and has_end:
            thought_content = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL | re.IGNORECASE)
            if thought_content and len(thought_content.group(1).strip()) < 10:
                issues.append(ValidationIssue(
                    code="hollow_reasoning",
                    message="Reasoning block is suspiciously short.",
                    severity="warning"
                ))

        return issues

    def _validate_complexity(self, text: str) -> List[ValidationIssue]:
        issues = []
        code_blocks = re.findall(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)
        
        if not code_blocks:
            return issues

        # Extract stated complexity
        stated = None
        for label, pattern in self.complexity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                stated = label
                break
        
        if stated:
            code = code_blocks[0]
            # 1. Nested Loop Check
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
            
            # 2. Recursion Check
            is_recursive = False
            func_match = re.search(r"def (\w+)\(", code)
            if func_match:
                name = func_match.group(1)
                if re.search(rf"\b{name}\(", code[func_match.end():]):
                    is_recursive = True
            
            # 3. Infinite Loop Check
            if re.search(r"while\s+True:", code) and "break" not in code:
                issues.append(ValidationIssue(
                    code="infinite_loop",
                    message="Potential infinite loop detected (while True without break).",
                    severity="critical"
                ))

            if is_recursive and "O(1)" in stated:
                issues.append(ValidationIssue(
                    code="complexity_mismatch",
                    message=f"Stated {stated} but code is recursive (likely logarithmic or linear).",
                    severity="warning"
                ))
            elif max_depth >= 2 and stated == "O(n)":
                issues.append(ValidationIssue(
                    code="complexity_mismatch",
                    message=f"Stated {stated} but code has nested loops (likely O(n^2)).",
                    severity="warning"
                ))

        return issues

    def _validate_content(self, text: str, context: str) -> List[ValidationIssue]:
        issues = []
        
        # 1. Repetition detection
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 5:
            seen = set()
            repeated_count = 0
            for s in sentences:
                if len(s) < 20: continue
                normalized = re.sub(r'\W+', '', s.lower())
                if normalized in seen:
                    repeated_count += 1
                seen.add(normalized)
            
            if repeated_count > 1:
                issues.append(ValidationIssue(
                    code="textual_repetition",
                    message="Significant textual repetition detected.",
                    severity="warning"
                ))

        # 2. Hallucinated tokens (role leakage)
        if re.search(r'\b(user:|assistant:|system:|<\|user\|>)\b', text, re.IGNORECASE):
            issues.append(ValidationIssue(
                code="hallucination_leak",
                message="Internal conversation tokens detected in response.",
                severity="critical"
            ))
            
        # 3. Robotic Artifacts (As an AI...)
        if re.search(r"\bas an ai language model\b|\bi don't have feelings\b", text, re.IGNORECASE):
            issues.append(ValidationIssue(
                code="robotic_artifact",
                message="Robotic AI disclaimer detected.",
                severity="info"
            ))

        return issues

    def get_affected_section(self, text: str, issue: ValidationIssue) -> str:
        """Extracts the specific snippet affected by the issue."""
        if "code" in issue.code:
            blocks = re.findall(r"```(?:\w*)\n(.*?)\n```", text, re.DOTALL)
            if blocks: return blocks[0][-100:] # Last 100 chars of code
        
        # Fallback to last few sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        return " ".join(sentences[-2:])

    def repair(self, text: str, report: ValidationReport) -> str:
        """Applies safe repairs to the text."""
        result = report.repaired_text
        
        for issue in report.issues:
            if issue.code == "hallucination_leak":
                result = re.sub(r'\b(user:|assistant:|system:|<\|user\|>|<\|assistant\|>)\b.*', '', result, flags=re.IGNORECASE)
            
            if issue.code == "textual_repetition":
                # Simple deduplication
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
