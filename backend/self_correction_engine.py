import re
import ast
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("self_correction")

@dataclass
class CorrectionReport:
    is_valid: bool
    issues: List[str]
    repair_actions: List[str]
    needs_regeneration: bool = False

class SelfCorrectionEngine:
    """
    Advanced diagnostic and repair engine for chatbot responses.
    Detects and repairs code, logic, markdown, and complexity errors.
    """

    def __init__(self):
        self.complexity_patterns = {
            "O(1)": r"\bO\(1\)\b",
            "O(log n)": r"\bO\(log\s*n\)\b",
            "O(n)": r"\bO\(n\)\b",
            "O(n log n)": r"\bO\(n\s*log\s*n\)\b",
            "O(n^2)": r"\bO\(n\^2\)\b"
        }

    def analyze(self, text: str, code_intent: bool = False) -> CorrectionReport:
        issues = []
        repairs = []
        needs_regeneration = False

        # 1. Broken Markdown / Incomplete Code
        if text.count("```") % 2 != 0:
            issues.append("Broken markdown: Unclosed code block.")
            repairs.append("Closing markdown block.")
            text += "\n```"

        # 2. Code Validity (AST)
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)\n```", text, re.DOTALL)
        for block in code_blocks:
            try:
                ast.parse(block)
            except SyntaxError as e:
                issues.append(f"Syntax error in code: {str(e)}")
                needs_regeneration = True

        # 3. Complexity Consistency (Heuristic-based)
        if "complexity" in text.lower() and code_blocks:
            detected_complexity = self._detect_actual_complexity(code_blocks[0])
            stated_complexity = self._extract_stated_complexity(text)
            
            if stated_complexity and detected_complexity != stated_complexity:
                # If the code has nested loops but says O(n), flag it
                if "n^2" in detected_complexity and "O(n)" in stated_complexity:
                    issues.append(f"Logical inconsistency: Stated {stated_complexity} but code appears to be {detected_complexity}.")
                    repairs.append("Updating complexity analysis section.")

        # 4. Hallucinations (Role Leakage)
        if re.search(r'\b(user:|assistant:|system:)\b', text, re.IGNORECASE):
            issues.append("Hallucination detected: Internal role tokens leaked.")
            repairs.append("Stripping conversation artifacts.")

        # 5. Repeated Text
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 4:
            unique = set(s.strip().lower() for s in sentences if len(s) > 10)
            if len(unique) < len([s for s in sentences if len(s) > 10]) * 0.7:
                issues.append("High textual redundancy detected.")
                repairs.append("Deduplicating sentences.")

        return CorrectionReport(
            is_valid=len(issues) == 0,
            issues=issues,
            repair_actions=repairs,
            needs_regeneration=needs_regeneration
        )

    def _detect_actual_complexity(self, code: str) -> str:
        """Simple heuristic to detect Big O from loops."""
        loop_count = code.count("for ") + code.count("while ")
        if loop_count >= 2 and ("for" in code and "range" in code):
            # Check for nesting (naive check)
            lines = code.split("\n")
            max_indent = 0
            for line in lines:
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
            if max_indent >= 8: return "O(n^2)"
        if loop_count == 1: return "O(n)"
        return "O(1)"

    def _extract_stated_complexity(self, text: str) -> Optional[str]:
        for label, pattern in self.complexity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return None

    def repair(self, text: str, report: CorrectionReport) -> str:
        """Applies rule-based repairs for non-structural issues."""
        result = text
        
        for action in report.repair_actions:
            if "Closing markdown" in action:
                if result.count("```") % 2 != 0: result += "\n```"
            
            if "Stripping conversation artifacts" in action:
                result = re.sub(r'\b(user:|assistant:|system:)\b.*', '', result, flags=re.IGNORECASE)
            
            if "Deduplicating" in action:
                sentences = re.split(r'(?<=[.!?]) +', result)
                seen = set()
                final = []
                for s in sentences:
                    norm = re.sub(r'\W', '', s.lower())
                    if norm not in seen:
                        seen.add(norm)
                        final.append(s)
                result = ' '.join(final)

        return result.strip()
