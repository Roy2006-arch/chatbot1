import random
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from ..schema import (
    BuggyExample, SourceCode, ErrorInfo, TestCase,
    Language, BugCategory, ErrorType, Severity, BUG_CATEGORY_DIFFICULTY,
)


class BugInjector:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stats = {"injected": 0, "by_category": {}}

    def inject(
        self,
        correct_code: str,
        language: Language,
        category: Optional[BugCategory] = None,
        count: int = 1,
    ) -> List[BuggyExample]:
        correct_code = correct_code.strip()
        examples = []
        categories = [category] if category else self._applicable_categories(language)

        for _ in range(count):
            cat = random.choice(categories)
            injector = self._get_injector(cat, language)
            if injector:
                example = injector(correct_code, language, cat)
                if example:
                    self._enrich_example(example)
                    examples.append(example)
                    self.stats["injected"] += 1
                    self.stats["by_category"][cat.value] = self.stats["by_category"].get(cat.value, 0) + 1

        return examples

    def inject_batch(
        self,
        code_pairs: List[Tuple[str, Language]],
        examples_per_pair: int = 2,
        num_workers: int = 8,
    ) -> List[BuggyExample]:
        all_examples = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for code, lang in code_pairs:
                future = executor.submit(self.inject, code, lang, count=examples_per_pair)
                futures[future] = (code, lang)

            for future in as_completed(futures):
                try:
                    all_examples.extend(future.result())
                except Exception:
                    pass
        return all_examples

    def _get_injector(self, category: BugCategory, language: Language):
        injectors = {
            BugCategory.OFF_BY_ONE: self._inject_off_by_one,
            BugCategory.NULL_POINTER: self._inject_null_pointer,
            BugCategory.TYPE_MISMATCH: self._inject_type_mismatch,
            BugCategory.LOGIC_ERROR: self._inject_logic_error,
            BugCategory.INFINITE_LOOP: self._inject_infinite_loop,
            BugCategory.INDEX_OUT_OF_BOUNDS: self._inject_index_bound,
            BugCategory.DIVISION_BY_ZERO: self._inject_division_zero,
            BugCategory.UNINITIALIZED_VAR: self._inject_uninitialized,
            BugCategory.EDGE_CASE: self._inject_edge_case,
            BugCategory.SYNTAX: self._inject_syntax_error,
            BugCategory.IMPORT_ERROR: self._inject_import_error,
            BugCategory.NAMING_CONFLICT: self._inject_naming_conflict,
            BugCategory.PERFORMANCE: self._inject_performance_bug,
        }
        return injectors.get(category)

    def _applicable_categories(self, language: Language) -> List[BugCategory]:
        return [
            BugCategory.OFF_BY_ONE, BugCategory.LOGIC_ERROR,
            BugCategory.INDEX_OUT_OF_BOUNDS, BugCategory.DIVISION_BY_ZERO,
            BugCategory.UNINITIALIZED_VAR, BugCategory.EDGE_CASE,
            BugCategory.TYPE_MISMATCH,
        ]

    def _inject_off_by_one(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "range(" in stripped and ")" in stripped:
                mod_lines[i] = line.replace("range(", "range(0, ")
                injection_point = i
                break
            if "for " in stripped and (" < " in stripped or " <= " in stripped):
                if " < " in stripped:
                    mod_lines[i] = line.replace(" < ", " <= ")
                else:
                    mod_lines[i] = line.replace(" <= ", " < ")
                injection_point = i
                break
            if "while " in stripped and (" < " in stripped or " <= " in stripped):
                if " < " in stripped:
                    mod_lines[i] = line.replace(" < ", " <= ")
                else:
                    mod_lines[i] = line.replace(" <= ", " < ")
                injection_point = i
                break
            if "range(" in stripped:
                mod_lines[i] = re.sub(r"range\((\w+)\)", r"range(\1 + 1)", line)
                injection_point = i
                break
            if "range(" in stripped:
                mod_lines[i] = re.sub(r"range\((\w+)\)", r"range(\1 - 1)", line)
                injection_point = i
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.LOGICAL, "Off-by-one error in loop boundary condition. The loop iterates one too many/few times.", line_number=injection_point + 1),
            explanation=f"The loop boundary at line {injection_point + 1} has an off-by-one error. This causes the loop to access one element beyond (or short of) the valid range.",
            fix_strategy=f"Fix the comparison operator or range boundary on line {injection_point + 1}. Use '<' instead of '<=' (or vice versa) depending on the intended behavior.",
            severity=Severity.MEDIUM,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["loop", "boundary", "off_by_one"],
        )

    def _inject_null_pointer(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if lang == Language.PYTHON:
                if "def " in stripped and "(" in stripped:
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    func_match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", stripped)
                    if func_match and func_match.group(2).strip():
                        params = [p.strip() for p in func_match.group(2).split(",")]
                        if params:
                            first_param = params[0]
                            if "self" not in first_param:
                                mod_lines.insert(i + 1, f"{indent}if {first_param} is None: return None  # BUG: missing None check in actual code")
                                injection_point = i + 1
                                break
            elif lang == Language.JAVA:
                if "class " in stripped or "public " in stripped:
                    indent = "\t"
                    mod_lines.insert(i + 1, f"{indent}// BUG: potential null dereference")
                    injection_point = i + 1
                    break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.RUNTIME, "NullPointerException / AttributeError on NoneType object", line_number=injection_point + 1),
            explanation=f"A null/None check is missing at line {injection_point + 1}. When the value is null, accessing its attributes/methods raises a runtime error.",
            fix_strategy=f"Add a null check before accessing the object's members on line {injection_point + 1}.",
            severity=Severity.HIGH,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 3),
            tags=["null", "nil", "pointer"],
        )

    def _inject_type_mismatch(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        type_replacements = {
            Language.PYTHON: [("str(42)", "42"), ("int(x)", "str(x)"), ('"count: " + str(n)', '"count: " + n')],
            Language.JAVASCRIPT: [('Number(x)', 'String(x)'), ('parseInt(x)', 'x'), ('String(n)', 'n')],
        }

        replacements = type_replacements.get(lang, [])
        for i, line in enumerate(lines):
            for old, new in replacements:
                if old in line:
                    mod_lines[i] = line.replace(old, new)
                    injection_point = i
                    break
            if injection_point is not None:
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.COMPILE_TIME if lang != Language.PYTHON else ErrorType.RUNTIME,
                               "Type mismatch: expected string but got integer (or vice versa)",
                               line_number=injection_point + 1),
            explanation=f"Type mismatch at line {injection_point + 1}. The expression produces a different type than expected by the operation.",
            fix_strategy=f"Add explicit type conversion at line {injection_point + 1}.",
            severity=Severity.MEDIUM,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["type", "coercion", "conversion"],
        )

    def _inject_logic_error(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "if " in stripped and " > " in stripped:
                mod_lines[i] = line.replace(" > ", " < ")
                injection_point = i
                break
            if "if " in stripped and " < " in stripped:
                mod_lines[i] = line.replace(" < ", " > ")
                injection_point = i
                break
            if "if " in stripped and "==" in stripped and "!" not in stripped:
                mod_lines[i] = line.replace("==", "!=")
                injection_point = i
                break
            if "if " in stripped and "!=" in stripped:
                mod_lines[i] = line.replace("!=", "==")
                injection_point = i
                break
            if "return " in stripped and "and" in stripped:
                mod_lines[i] = line.replace("and", "or")
                injection_point = i
                break
            if "if " in stripped and " not in " in stripped:
                mod_lines[i] = line.replace(" not in ", " in ")
                injection_point = i
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.LOGICAL, "Logic error: inverted comparison or condition. The code compiles but produces wrong results.",
                               line_number=injection_point + 1),
            explanation=f"A logical operator or comparison is inverted at line {injection_point + 1}. The code will execute the wrong branch.",
            fix_strategy=f"Restore the correct comparison/operator on line {injection_point + 1}.",
            severity=Severity.HIGH,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 3),
            tags=["logic", "comparison", "branch"],
        )

    def _inject_infinite_loop(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "while " in stripped:
                if "True" in stripped or "true" in stripped:
                    continue
                if ":" in stripped:
                    indent = " " * (len(line) - len(line.lstrip()))
                    cond = stripped.split(":")[0].replace("while ", "").strip()
                    mod_lines[i] = f"{indent}while {cond}:"
                    injection_point = i
                    break

        if injection_point is None:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if "for " in stripped and "in " in stripped:
                    indent = " " * (len(line) - len(line.lstrip()))
                    mod_lines.insert(i + 1, f"{indent}    # BUG: missing loop variable update")
                    injection_point = i + 1
                    break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.RUNTIME, "Infinite loop: loop termination condition is never reached",
                               line_number=injection_point + 1),
            explanation=f"The loop at line {injection_point + 1} lacks a proper termination condition or the loop variable is not updated, causing it to run indefinitely.",
            fix_strategy=f"Ensure the loop variable is updated each iteration or add a proper exit condition on line {injection_point + 1}.",
            severity=Severity.HIGH,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 3),
            tags=["loop", "termination", "infinite"],
        )

    def _inject_index_bound(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if lang == Language.PYTHON:
                matches = re.findall(r'(\w+)\[(\w+)\]', stripped)
                for var, idx in matches:
                    if idx not in ("i", "j", "k", "index", "idx") or idx == "0":
                        continue
                    if "if " not in stripped:
                        mod_lines[i] = line.replace(f"[{idx}]", f"[{idx} + 1]")
                        injection_point = i
                        break
            elif lang in (Language.JAVA, Language.CPP, Language.JAVASCRIPT):
                matches = re.findall(r'(\w+)\[(\w+)\]', stripped)
                for var, idx in matches:
                    if idx in ("i", "j", "k"):
                        mod_lines[i] = line.replace(f"[{idx}]", f"[{idx} + 1]")
                        injection_point = i
                        break
            if injection_point is not None:
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.RUNTIME, "Index out of bounds: array/list index exceeds valid range",
                               line_number=injection_point + 1),
            explanation=f"Array index at line {injection_point + 1} may exceed the valid bounds. The '+1' shift can cause access beyond the last element.",
            fix_strategy=f"Correct the array index on line {injection_point + 1}. Ensure the index stays within [0, len-1].",
            severity=Severity.HIGH,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["index", "array", "bounds"],
        )

    def _inject_division_zero(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            for op in [" / ", " // ", " % "]:
                if op in stripped:
                    parts = stripped.split(op)
                    if len(parts) >= 2:
                        divisor = parts[1].split()[0] if parts[1].split() else ""
                        if divisor and divisor not in ("0", "1", "2") and "/" not in divisor:
                            indent = " " * (len(line) - len(line.lstrip()))
                            mod_lines.insert(i + 1, f"{indent}    # BUG: missing zero divisor check")
                            injection_point = i + 1
                            break
            if injection_point is not None:
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.RUNTIME, "Division by zero: denominator can be zero at runtime",
                               line_number=injection_point + 1),
            explanation=f"The division operation before line {injection_point + 1} lacks a zero-divisor check. If the divisor is zero, this raises a ZeroDivisionError/ArithmeticException.",
            fix_strategy=f"Add a check for zero divisor before the division operation.",
            severity=Severity.HIGH,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["division", "zero", "arithmetic"],
        )

    def _inject_uninitialized(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if lang == Language.PYTHON:
                match = re.search(r"(\w+)\s*=\s*(\w+)", stripped)
                if match and match.group(2) not in ("True", "False", "None", "0", "1", "[]", "{}"):
                    var_name = match.group(2)
                    if var_name not in stripped[:stripped.index("=")]:
                        indent = " " * (len(line) - len(line.lstrip()))
                        mod_lines.insert(i, f"{indent}# BUG: {var_name} may not be initialized")
                        injection_point = i + 1
                        break
            elif lang in (Language.JAVA, Language.CPP):
                match = re.search(r"(int|String|double|float|bool)\s+(\w+)\s*=\s*(\w+)", stripped)
                if match and match.group(3) not in ("0", "1", "null", "true", "false"):
                    mod_lines[i] = line.replace(f"= {match.group(3)}", "// uninitialized")
                    injection_point = i + 1
                    break
            if injection_point is not None:
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.RUNTIME, "Uninitialized variable error: variable used before assignment",
                               line_number=injection_point + 1),
            explanation=f"A variable at line {injection_point + 1} may be used without being initialized. This can cause unpredictable behavior or runtime errors.",
            fix_strategy=f"Initialize the variable with a default value before use around line {injection_point + 1}.",
            severity=Severity.MEDIUM,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["initialization", "variable"],
        )

    def _inject_edge_case(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "def " in stripped and "(" in stripped:
                indent = " " * (len(line) - len(line.lstrip()) + 4)
                mod_lines.insert(i + 1, f"{indent}# BUG: missing empty/edge case check")
                injection_point = i + 1
                break
            if "function " in stripped and "(" in stripped:
                indent = " " * (len(line) - len(line.lstrip())) + "  "
                mod_lines.insert(i + 1, f"{indent}// BUG: missing edge case check")
                injection_point = i + 1
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.LOGICAL, "Edge case not handled: empty/null input causes incorrect behavior",
                               line_number=injection_point + 1),
            explanation=f"The function at line {injection_point + 1} does not handle edge cases such as empty input, null values, or boundary conditions.",
            fix_strategy=f"Add input validation and edge case handling at the beginning of the function around line {injection_point + 1}.",
            severity=Severity.MEDIUM,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 2),
            tags=["edge_case", "validation", "boundary"],
        )

    def _inject_syntax_error(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        syntax_bugs = {
            Language.PYTHON: [
                (r":", ""), (r"def ", "def  "), (r"    ", "\t"),
            ],
            Language.JAVASCRIPT: [
                (r";", ""), (r"{", ""), (r"}", ""),
            ],
            Language.JAVA: [
                (r";", ""), (r"{", ""), (r"public", "publik"),
            ],
            Language.CPP: [
                (r";", ""), (r"{", ""), (r"#include", "#iclude"),
            ],
        }

        replacements = syntax_bugs.get(lang, [])
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "/*")):
                continue
            for old, new in replacements:
                if old in stripped and old.strip():
                    mod_lines[i] = line.replace(old, new, 1)
                    injection_point = i
                    break
            if injection_point is not None:
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        error_msg = {
            Language.PYTHON: "SyntaxError: invalid syntax",
            Language.JAVASCRIPT: "SyntaxError: Unexpected token",
            Language.JAVA: "Compilation error: ';' expected",
            Language.CPP: "error: expected ';' before...",
        }
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.COMPILE_TIME, error_msg.get(lang, "Syntax error"),
                               line_number=injection_point + 1),
            explanation=f"Syntax error at line {injection_point + 1}. A required token is missing or misformatted, preventing compilation/parsing.",
            fix_strategy=f"Correct the syntax on line {injection_point + 1} by adding the missing token or fixing the formatting.",
            severity=Severity.LOW,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 1),
            tags=["syntax", "parse"],
        )

    def _inject_import_error(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if lang == Language.PYTHON and ("import " in stripped or "from " in stripped):
                mod_lines[i] = line.replace("import ", "import non_existent_module_xyz_")
                injection_point = i
                break
            elif lang in (Language.JAVA, Language.CPP) and "#include" in stripped or "import " in stripped:
                mod_lines[i] = line.replace("import ", "import nonexistent.")
                injection_point = i
                break
            elif lang == Language.JAVASCRIPT and ("require(" in stripped or "import " in stripped):
                mod_lines[i] = line.replace("require(", "require('nonexistent_module')  // ")
                injection_point = i
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.COMPILE_TIME, f"ImportError/ModuleNotFoundError: cannot import non_existent_module",
                               line_number=injection_point + 1),
            explanation=f"Import error at line {injection_point + 1}. The module being imported does not exist or is misspelled.",
            fix_strategy=f"Correct the import statement on line {injection_point + 1} to use the correct module name.",
            severity=Severity.LOW,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 1),
            tags=["import", "module", "dependency"],
        )

    def _inject_naming_conflict(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        variables = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            match = re.search(r"(\w+)\s*=", stripped)
            if match:
                var = match.group(1)
                if var not in ("True", "False", "None", "self") and "def " not in stripped and "class " not in stripped:
                    variables.append((var, i))

        if len(variables) >= 2:
            var1, idx1 = variables[0]
            var2, idx2 = variables[1]
            if var1 != var2:
                mod_lines[idx2] = mod_lines[idx2].replace(f"{var2} =", f"{var1} =", 1)
                injection_point = idx2 + 1

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.LOGICAL, f"Variable naming conflict: '{var1}' is reassigned unexpected value",
                               line_number=injection_point + 1),
            explanation=f"Variable naming conflict at line {injection_point + 1}. A variable name is reused with a different meaning, shadowing the original value.",
            fix_strategy=f"Rename the variable on line {injection_point + 1} to avoid shadowing the previously defined '{var1}'.",
            severity=Severity.LOW,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 1),
            tags=["naming", "shadow", "scope"],
        )

    def _inject_performance_bug(self, code: str, lang: Language, cat: BugCategory) -> Optional[BuggyExample]:
        lines = code.split("\n")
        mod_lines = lines.copy()
        injection_point = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "for " in stripped:
                indent = " " * (len(line) - len(line.lstrip()))
                mod_lines.insert(i + 1, f"{indent}    # BUG: O(n²) instead of O(n)")
                injection_point = i + 1
                break

        if injection_point is None:
            return None

        buggy = "\n".join(mod_lines)
        return self._make_example(code, buggy, lang, cat,
            error_info=ErrorInfo(ErrorType.WARNING, "Performance issue: suboptimal algorithm with avoidable O(n²) complexity",
                               line_number=injection_point + 1),
            explanation=f"Performance bug near line {injection_point + 1}. The algorithm uses a nested loop where a hash map or single pass would suffice, leading to O(n²) instead of O(n).",
            fix_strategy=f"Replace the nested loop with a more efficient approach (hash map, two-pointer, etc.) near line {injection_point + 1}.",
            severity=Severity.MEDIUM,
            difficulty=BUG_CATEGORY_DIFFICULTY.get(cat, 3),
            tags=["performance", "complexity", "optimization"],
        )

    def _make_example(self, correct: str, buggy: str, lang: Language, cat: BugCategory,
                      error_info: ErrorInfo, explanation: str, fix_strategy: str,
                      severity: Severity, difficulty: int, tags: List[str]) -> BuggyExample:
        return BuggyExample(
            buggy_code=SourceCode(language=lang, code=buggy),
            corrected_code=SourceCode(language=lang, code=correct),
            language=lang,
            category=cat,
            error_info=error_info,
            severity=severity,
            title=f"{cat.value.replace('_', ' ').title()} in {lang.value}",
            description=f"A {cat.value.replace('_', ' ')} bug in {lang.value} code.",
            explanation=explanation,
            fix_strategy=fix_strategy,
            difficulty=difficulty,
            tags=tags,
            test_cases=[TestCase(input_data="", expected_output="", description="Basic test")],
        )

    def _enrich_example(self, example: BuggyExample):
        if not example.id:
            raw = f"{example.language.value}:{example.category.value}:{example.buggy_code.code[:50]}"
            example.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def get_stats(self) -> Dict:
        return self.stats
