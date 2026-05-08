import ast
import json
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ingestion import DatasetExample


class CodeValidator:
    def __init__(self, languages: Optional[List[str]] = None):
        self.languages = languages or ["python", "javascript", "typescript", "java", "cpp", "rust", "go", "sql"]
        self.stats = {"valid": 0, "invalid": 0, "skipped": 0}

    def validate(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._validate_single, ex): ex for ex in examples}
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _validate_single(self, example: DatasetExample) -> Optional[DatasetExample]:
        code_blocks = self._extract_code_blocks(example.output)
        if not code_blocks:
            self.stats["skipped"] += 1
            return example

        valid_blocks = []
        for block in code_blocks:
            if self._validate_block(block):
                valid_blocks.append(block)

        if len(valid_blocks) == len(code_blocks):
            self.stats["valid"] += 1
            return example
        elif len(valid_blocks) > 0:
            example.metadata["code_validation"] = {
                "total_blocks": len(code_blocks),
                "valid_blocks": len(valid_blocks),
                "has_invalid": True,
            }
            self.stats["valid"] += 1
            return example
        else:
            self.stats["invalid"] += 1
            return None

    def _extract_code_blocks(self, text: str) -> List[Dict]:
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{"language": lang or "", "code": code.strip()} for lang, code in matches]

    def _validate_block(self, block: Dict) -> bool:
        lang = block["language"].lower()
        code = block["code"]

        if not code:
            return False

        if lang in ("", "text", "markdown", "json", "yaml", "xml", "html", "css", "sql", "bash", "sh", "powershell", "dockerfile", "makefile"):
            return True

        if len(code) > 5000:
            code = code[:5000]

        if lang == "python":
            return self._validate_python(code)
        elif lang in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
            return self._validate_javascript(code)
        elif lang in ("java", "cpp", "c++", "c", "rust", "go"):
            return self._validate_bracket_syntax(code)
        return True

    def _validate_python(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            try:
                compile(code, "<test>", "exec")
                return True
            except SyntaxError:
                return False

    def _validate_javascript(self, code: str) -> bool:
        return self._validate_bracket_syntax(code)

    def _validate_bracket_syntax(self, code: str) -> bool:
        stack = []
        brackets = {"{": "}", "[": "]", "(": ")"}
        in_string = False
        string_char = None
        i = 0

        while i < len(code):
            ch = code[i]
            if in_string:
                if ch == "\\":
                    i += 1
                elif ch == string_char:
                    in_string = False
                    string_char = None
            else:
                if ch in ("'", '"', "`"):
                    in_string = True
                    string_char = ch
                elif ch in brackets:
                    stack.append(brackets[ch])
                elif ch in brackets.values():
                    if not stack or stack.pop() != ch:
                        return False
            i += 1
        return len(stack) == 0

    def has_valid_code(self, text: str, language: str = "python") -> bool:
        blocks = self._extract_code_blocks(text)
        for block in blocks:
            if block["language"].lower() == language:
                return self._validate_block(block)
        return False

    def get_stats(self) -> Dict:
        return self.stats


class MarkdownValidator:
    def __init__(self):
        self.stats = {"valid": 0, "invalid": 0, "warnings": 0}

    def validate(self, examples: List[DatasetExample], num_workers: int = 8) -> List[DatasetExample]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._validate_single, ex) for ex in examples]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _validate_single(self, example: DatasetExample) -> Optional[DatasetExample]:
        issues = self._check_markdown(example.output)
        if len(issues) > 5:
            self.stats["invalid"] += 1
            return None
        example.metadata["markdown_issues"] = issues
        self.stats["valid"] += 1
        return example

    def _check_markdown(self, text: str) -> List[str]:
        issues = []

        code_blocks = re.findall(r"```(\w*)\n.*?```", text, re.DOTALL)
        backtick_count = text.count("```")
        if backtick_count % 2 != 0:
            issues.append("Unmatched code fences")
        if backtick_count > 0 and backtick_count % 2 == 0:
            pairs = re.findall(r"```\w*\n.*?```", text, re.DOTALL)
            if len(pairs) * 2 != backtick_count:
                issues.append("Malformed code blocks")

        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^#{7,}\s", stripped):
                issues.append(f"Line {i+1}: heading depth > 6")
            if re.match(r"^(\|.*){3,}$", stripped):
                cols = stripped.split("|")
                if len(cols) > 20:
                    issues.append(f"Line {i+1}: table with too many columns ({len(cols)})")

        return issues

    def get_stats(self) -> Dict:
        return self.stats


class SchemaValidator:
    REQUIRED_FIELDS = ["instruction", "output"]
    OPTIONAL_FIELDS = ["input", "category", "difficulty", "metadata"]

    @classmethod
    def validate_schema(cls, example: DatasetExample) -> Tuple[bool, List[str]]:
        errors = []
        if not isinstance(example.instruction, str):
            errors.append("instruction must be a string")
        if not isinstance(example.output, str):
            errors.append("output must be a string")
        if not isinstance(example.input, str):
            errors.append("input must be a string")
        if example.difficulty is not None and not isinstance(example.difficulty, (int, float)):
            errors.append("difficulty must be numeric")
        return len(errors) == 0, errors
