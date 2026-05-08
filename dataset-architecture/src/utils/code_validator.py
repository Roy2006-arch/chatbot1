import ast
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CodeSyntaxValidator:
    @staticmethod
    def validate_python(code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def validate_python_execution(code: str, timeout: int = 5) -> Tuple[bool, str]:
        try:
            compile(code, "<string>", "exec")
            restricted_globals = {"__builtins__": {"print": print, "range": range, "len": len, "int": int, "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple, "float": float, "bool": bool, "True": True, "False": False, "None": None, "abs": abs, "max": max, "min": min, "sum": sum, "sorted": sorted, "reversed": reversed, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "any": any, "all": all}}
            exec(code, restricted_globals)
            return True, ""
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_java(code: str) -> Tuple[bool, Optional[str]]:
        if not re.search(r'class\s+\w+', code):
            return False, "No class definition found"
        if not re.search(r'public\s+static\s+void\s+main', code) and not re.search(r'void\s+\w+\s*\(', code):
            return False, "No method definition found"
        return CodeSyntaxValidator._check_brackets(code)

    @staticmethod
    def validate_javascript(code: str) -> Tuple[bool, Optional[str]]:
        return CodeSyntaxValidator._check_brackets(code)

    @staticmethod
    def _check_brackets(code: str) -> Tuple[bool, Optional[str]]:
        stack = []
        pairs = {"{": "}", "[": "]", "(": ")"}
        in_string = False
        string_char = None
        i = 0

        while i < len(code):
            ch = code[i]
            if ch == "\\" and in_string:
                i += 2
                continue
            if ch in ("'", '"', "`") and not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char and in_string:
                in_string = False
                string_char = None
            elif not in_string:
                if ch in pairs:
                    stack.append(pairs[ch])
                elif ch in pairs.values():
                    if not stack:
                        return False, f"Unmatched closing bracket '{ch}' at position {i}"
                    expected = stack.pop()
                    if ch != expected:
                        return False, f"Mismatched bracket: expected '{expected}', got '{ch}' at position {i}"
            i += 1

        if stack:
            return False, f"Unclosed brackets: {len(stack)} remaining"
        return True, None

    @staticmethod
    def detect_language(code: str) -> str:
        patterns = [
            (r'import\s+java\.|public\s+(class|static|void|interface|enum)', "java"),
            (r'#include\s*<|int\s+main\s*\(|std::|cout|cin', "cpp"),
            (r'fn\s+\w+|let\s+mut|pub\s+fn|impl\s+|->\s*\w+', "rust"),
            (r'package\s+\w+|import\s+\w+|func\s+\w+|defer\s+', "go"),
            (r'function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+|console\.|=>', "javascript"),
            (r'def\s+\w+\s*\(|if\s+__name__\s*==|print\s*\(|class\s+\w+\s*:', "python"),
            (r'SELECT\s+|FROM\s+|WHERE\s+|JOIN\s+|INSERT\s+|UPDATE\s+|DELETE\s+|CREATE\s+TABLE', "sql"),
        ]
        score = {lang: 0 for _, lang in patterns}
        for pattern, lang in patterns:
            if re.search(pattern, code):
                score[lang] += 1
        best = max(score, key=score.get)
        return best if score[best] > 0 else "unknown"
