import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Generator, Any
from pathlib import Path

from ..schema import Problem, TestCase, Solution, Language, Platform, DifficultyLevel


class BaseProblemParser(ABC):
    @abstractmethod
    def parse(self, data: Dict) -> Problem:
        pass

    @abstractmethod
    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        pass

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{"language": lang.lower() if lang else "text", "code": code.strip()} for lang, code in matches]

    def _parse_complexity(self, text: str) -> tuple:
        time_match = re.search(r"O\([^)]+\)", text or "")
        space_match = re.findall(r"O\([^)]+\)", text or "")
        time = time_match.group(0) if time_match else "O(n)"
        space = space_match[-1] if len(space_match) > 1 else "O(1)"
        if space == time and len(space_match) > 0:
                space = space_match[0] if space_match else "O(1)"
        if space == time:
            space = "O(1)"
        return time, space

    def _detect_language(self, code: str) -> Language:
        patterns = {
            Language.PYTHON: [r"def\s+\w+\s*\(", r"import\s+\w+", r"class\s+\w+\s*:", r"print\s*\("],
            Language.JAVA: [r"public\s+class", r"private\s+\w+", r"import\s+java\.", r"System\.out"],
            Language.CPP: [r"#include", r"int\s+main\s*\(", r"std::", r"using\s+namespace"],
            Language.JAVASCRIPT: [r"function\s+\w+\s*\(", r"const\s+\w+\s*=", r"let\s+\w+", r"console\.log"],
            Language.RUST: [r"fn\s+\w+", r"let\s+mut", r"pub\s+fn", r"impl\s+"],
            Language.GO: [r"func\s+\w+", r"package\s+\w+", r"import\s+\("],
        }
        scores = {lang: 0 for lang in Language}
        for lang, pats in patterns.items():
            for p in pats:
                if re.search(p, code):
                    scores[lang] += 1
                if lang in (Language.PYTHON, Language.JAVA, Language.CPP):
                    scores[lang] *= 1.5
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else Language.PYTHON


class LeetCodeParser(BaseProblemParser):
    def parse(self, data: Dict) -> Problem:
        p = Problem(
            platform=Platform.LEETCODE,
            platform_id=str(data.get("id", data.get("questionId", ""))),
            title=data.get("title", data.get("titleSlug", "")),
            difficulty=self._parse_difficulty(data.get("difficulty", "Medium")),
            problem_statement=data.get("content", data.get("description", "")),
            constraints=self._extract_constraints(data.get("content", "")),
            input_format=data.get("input_format", ""),
            output_format=data.get("output_format", ""),
            solution_approach=data.get("solution_approach", ""),
            complexity_analysis=data.get("complexity_analysis", ""),
            hints=data.get("hints", data.get("similarQuestions", [])),
            topics=data.get("topicTags", data.get("tags", [])),
            companies=data.get("companies", []),
            acceptance_rate=float(data.get("acceptance_rate", data.get("acRate", 0))),
            frequency=float(data.get("frequency", 0)),
            rating=int(data.get("rating", 0)),
        )

        p.problem_statement = self._clean_html(p.problem_statement)

        examples = data.get("examples", data.get("sampleTestCases", []))
        if isinstance(examples, str):
            examples = json.loads(examples) if examples.startswith("[") else [{"input": "", "output": examples}]
        for ex in examples[:5]:
            if isinstance(ex, dict):
                tc = TestCase(
                    input=ex.get("input", ex.get("Input", "")),
                    expected_output=ex.get("output", ex.get("Output", "")),
                    explanation=ex.get("explanation", ex.get("Explanation", "")),
                )
                p.sample_test_cases.append(tc)

        solutions_data = data.get("solutions", data.get("code", {}))
        if isinstance(solutions_data, dict):
            for lang_name, code in solutions_data.items():
                try:
                    lang = Language(lang_name.lower())
                except ValueError:
                    lang = self._detect_language(code)
                time_c, space_c = self._parse_complexity(data.get("complexity_analysis", ""))
                sol = Solution(
                    language=lang,
                    code=code,
                    time_complexity=time_c,
                    space_complexity=space_c,
                    approach=data.get("approach_tip", ""),
                )
                p.solutions[lang.value] = sol

        return p

    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        p = Path(path)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield self.parse(item)
            else:
                yield self.parse(data)
        elif p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    if line.strip():
                        yield self.parse(json.loads(line))

    def _parse_difficulty(self, d: str) -> DifficultyLevel:
        mapping = {"Easy": DifficultyLevel.EASY, "Medium": DifficultyLevel.MEDIUM, "Hard": DifficultyLevel.HARD}
        return mapping.get(d, DifficultyLevel.MEDIUM)

    def _extract_constraints(self, html: str) -> List[str]:
        constraints = re.findall(r"<li>(.*?)</li>", html)
        return [re.sub(r"<[^>]+>", "", c).strip() for c in constraints if c.strip()]

    def _clean_html(self, html: str) -> str:
        text = re.sub(r"<pre>.*?</pre>", "", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&amp;", "&", text)
        return text.strip()


class CodeforcesParser(BaseProblemParser):
    def parse(self, data: Dict) -> Problem:
        p = Problem(
            platform=Platform.CODEFORCES,
            platform_id=data.get("contestId", "") + data.get("index", ""),
            title=data.get("name", ""),
            difficulty=self._parse_difficulty(data.get("rating", 0)),
            problem_statement=data.get("statement", ""),
            constraints=self._extract_constraints(data.get("statement", "")),
            input_format=data.get("inputSpecification", ""),
            output_format=data.get("outputSpecification", ""),
            rating=int(data.get("rating", 0)),
            tags=data.get("tags", []),
        )

        examples = data.get("examples", data.get("sampleTests", []))
        for ex in examples[:3]:
            if isinstance(ex, dict):
                tc = TestCase(
                    input=ex.get("input", ""),
                    expected_output=ex.get("output", ""),
                )
                p.sample_test_cases.append(tc)

        solutions_data = data.get("solutions", {})
        for lang_name, code in solutions_data.items():
            lang = self._detect_language(code)
            sol = Solution(
                language=lang,
                code=code,
                time_complexity=data.get("timeComplexity", "O(n log n)"),
                space_complexity=data.get("spaceComplexity", "O(n)"),
                approach=data.get("approach", ""),
            )
            p.solutions[lang.value] = sol

        return p

    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        p = Path(path)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            items = data.get("problems", data.get("result", data if isinstance(data, list) else [data]))
            for item in items:
                yield self.parse(item)

    def _parse_difficulty(self, rating: int) -> DifficultyLevel:
        if rating <= 1200: return DifficultyLevel.EASY
        elif rating <= 1800: return DifficultyLevel.MEDIUM
        elif rating <= 2400: return DifficultyLevel.HARD
        return DifficultyLevel.EXPERT

    def _extract_constraints(self, text: str) -> List[str]:
        constraints = re.findall(r"(?i)(\d+\s*[<≤]\s*\w+\s*[<≤]\s*\d+(?:[^.]*))", text)
        return [c.strip() for c in constraints]


class HackerRankParser(BaseProblemParser):
    def parse(self, data: Dict) -> Problem:
        p = Problem(
            platform=Platform.HACKERRANK,
            platform_id=str(data.get("slug", data.get("id", ""))),
            title=data.get("name", data.get("title", "")),
            difficulty=self._parse_difficulty(data.get("difficulty", "Easy")),
            problem_statement=data.get("problem_statement", data.get("description", "")),
            constraints=data.get("constraints", []),
            input_format=data.get("input_format", ""),
            output_format=data.get("output_format", ""),
            topics=data.get("topics", data.get("subdomain", [])),
            companies=data.get("companies", []),
        )

        examples = data.get("sample_test_cases", data.get("sampleInput", []))
        for ex in examples:
            if isinstance(ex, dict):
                tc = TestCase(
                    input=ex.get("input", ex.get("sample_input", "")),
                    expected_output=ex.get("output", ex.get("sample_output", "")),
                    explanation=ex.get("explanation", ""),
                )
                p.sample_test_cases.append(tc)

        return p

    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        p = Path(path)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            items = data.get("models", data.get("data", data if isinstance(data, list) else [data]))
            for item in items:
                yield self.parse(item)

    def _parse_difficulty(self, d: str) -> DifficultyLevel:
        mapping = {"Easy": DifficultyLevel.EASY, "Medium": DifficultyLevel.MEDIUM, "Hard": DifficultyLevel.HARD, "Advanced": DifficultyLevel.EXPERT}
        return mapping.get(d, DifficultyLevel.MEDIUM)


class CodeChefParser(BaseProblemParser):
    def parse(self, data: Dict) -> Problem:
        return Problem(
            platform=Platform.CODECHEF,
            platform_id=data.get("code", data.get("problemCode", "")),
            title=data.get("name", data.get("problemName", "")),
            difficulty=self._parse_difficulty(data.get("difficulty", data.get("rating", 0))),
            problem_statement=data.get("statement", data.get("body", "")),
            constraints=data.get("constraints", []),
            input_format=data.get("input_format", ""),
            output_format=data.get("output_format", ""),
            rating=int(data.get("rating", 0)),
            topics=data.get("tags", data.get("topics", [])),
        )

    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        p = Path(path)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                yield self.parse(item)

    def _parse_difficulty(self, d) -> DifficultyLevel:
        if isinstance(d, int):
            if d <= 1200: return DifficultyLevel.EASY
            elif d <= 1800: return DifficultyLevel.MEDIUM
            elif d <= 2200: return DifficultyLevel.HARD
            return DifficultyLevel.EXPERT
        mapping = {"Easy": DifficultyLevel.EASY, "Medium": DifficultyLevel.MEDIUM, "Hard": DifficultyLevel.HARD}
        return mapping.get(d, DifficultyLevel.MEDIUM)


class GeeksForGeeksParser(BaseProblemParser):
    def parse(self, data: Dict) -> Problem:
        p = Problem(
            platform=Platform.GEEKSFORGEEKS,
            platform_id=str(data.get("id", "")),
            title=data.get("title", data.get("name", "")),
            difficulty=self._parse_difficulty(data.get("difficulty", "Medium")),
            problem_statement=data.get("description", data.get("problem_statement", "")),
            constraints=data.get("constraints", []),
            input_format=data.get("input_format", ""),
            output_format=data.get("output_format", ""),
            topics=data.get("tags", data.get("topics", [])),
            companies=data.get("companies", []),
        )

        examples = data.get("examples", data.get("sample_test_cases", []))
        for ex in examples:
            if isinstance(ex, dict):
                tc = TestCase(
                    input=ex.get("input", ""),
                    expected_output=ex.get("output", ""),
                    explanation=ex.get("explanation", ""),
                )
                p.sample_test_cases.append(tc)

        solutions_data = data.get("solutions", data.get("code", {}))
        for lang_name, code in solutions_data.items():
            lang = self._detect_language(code)
            sol = Solution(
                language=lang,
                code=code,
                time_complexity=data.get("time_complexity", "O(n)"),
                space_complexity=data.get("space_complexity", "O(1)"),
                approach=data.get("approach_tip", ""),
            )
            p.solutions[lang.value] = sol

        return p

    def parse_batch(self, path: str) -> Generator[Problem, None, None]:
        p = Path(path)
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
            items = data.get("data", data if isinstance(data, list) else [data])
            for item in items:
                yield self.parse(item)

    def _parse_difficulty(self, d: str) -> DifficultyLevel:
        mapping = {"Basic": DifficultyLevel.EASY, "Easy": DifficultyLevel.EASY, "Medium": DifficultyLevel.MEDIUM, "Hard": DifficultyLevel.HARD}
        return mapping.get(d, DifficultyLevel.MEDIUM)


class PlatformParserFactory:
    _parsers = {
        Platform.LEETCODE: LeetCodeParser,
        Platform.CODEFORCES: CodeforcesParser,
        Platform.HACKERRANK: HackerRankParser,
        Platform.CODECHEF: CodeChefParser,
        Platform.GEEKSFORGEEKS: GeeksForGeeksParser,
    }

    @classmethod
    def get_parser(cls, platform: Platform) -> BaseProblemParser:
        parser_cls = cls._parsers.get(platform)
        if not parser_cls:
            raise ValueError(f"No parser for platform: {platform}")
        return parser_cls()

    @classmethod
    def auto_detect_platform(cls, data: Dict) -> Optional[Platform]:
        hints = {
            Platform.LEETCODE: ["questionId", "titleSlug", "acRate", "topicTags"],
            Platform.CODEFORCES: ["contestId", "index", "rating", "timeLimit"],
            Platform.HACKERRANK: ["slug", "subdomain", "difficulty"],
            Platform.CODECHEF: ["problemCode", "problemName", "code"],
            Platform.GEEKSFORGEEKS: ["problem_statement", "gfg"],
        }
        for platform, keys in hints.items():
            if any(k in data for k in keys):
                return platform
        return None
