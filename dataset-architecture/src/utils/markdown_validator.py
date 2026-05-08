import re
from typing import Dict, List, Optional, Tuple


class MarkdownValidator:
    MAX_HEADING_DEPTH = 6
    MAX_TABLE_COLS = 20
    MAX_LIST_NESTING = 6

    @staticmethod
    def validate(text: str) -> Tuple[bool, List[str]]:
        issues = []
        lines = text.split("\n")

        MarkdownValidator._check_headings(lines, issues)
        MarkdownValidator._check_code_blocks(text, issues)
        MarkdownValidator._check_tables(lines, issues)
        MarkdownValidator._check_lists(lines, issues)
        MarkdownValidator._check_html(text, issues)
        MarkdownValidator._check_links(text, issues)

        return len(issues) == 0, issues

    @staticmethod
    def _check_headings(lines: List[str], issues: List[str]):
        heading_pattern = re.compile(r"^(#{1,7})\s+")
        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                depth = len(match.group(1))
                if depth > MarkdownValidator.MAX_HEADING_DEPTH:
                    issues.append(f"Line {i+1}: Heading depth {depth} exceeds max {MarkdownValidator.MAX_HEADING_DEPTH}")

    @staticmethod
    def _check_code_blocks(text: str, issues: List[str]):
        fences = re.findall(r"```", text)
        if len(fences) % 2 != 0:
            issues.append("Unmatched code fences (odd number of ```)")

        blocks = re.findall(r"```(\w*)\n.*?```", text, re.DOTALL)
        fence_pairs = len(blocks)
        expected_fences = fence_pairs * 2
        if expected_fences != len(fences):
            issues.append(f"Found {len(fences)} fences but {expected_fences} expected from {fence_pairs} blocks")

    @staticmethod
    def _check_tables(lines: List[str], issues: List[str]):
        in_table = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cols = [c for c in stripped.split("|") if c.strip()]
                if len(cols) > MarkdownValidator.MAX_TABLE_COLS:
                    issues.append(f"Line {i+1}: Table has {len(cols)} columns (max {MarkdownValidator.MAX_TABLE_COLS})")

                if re.match(r"^\|[\s:-]+\|$", stripped):
                    if not in_table:
                        issues.append(f"Line {i+1}: Separator row without header row")
                else:
                    in_table = True
            else:
                in_table = False

    @staticmethod
    def _check_lists(lines: List[str], issues: List[str]):
        nesting_level = 0
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped.strip():
                nesting_level = 0
                continue

            indent = len(stripped) - len(stripped.lstrip())
            if re.match(r"^[\s]*[-*+]\s", stripped) or re.match(r"^[\s]*\d+\.\s", stripped):
                current_nesting = indent // 2
                if current_nesting > MarkdownValidator.MAX_LIST_NESTING:
                    issues.append(f"Line {i+1}: List nesting depth {current_nesting} exceeds max {MarkdownValidator.MAX_LIST_NESTING}")
                nesting_level = current_nesting

    @staticmethod
    def _check_html(text: str, issues: List[str]):
        unclosed_tags = re.findall(r"<(?!br|hr|img|input|meta|link|area|base|col|embed|source|track|wbr\s*/?|!\[CDATA\[|!--)(\w+)[^>]*>(?!(.*?</\1>))", text, re.IGNORECASE)
        if unclosed_tags:
            issues.append(f"Potentially unclosed HTML tags: {', '.join(set(unclosed_tags[:5]))}")

    @staticmethod
    def _check_links(text: str, issues: List[str]):
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
        link_pattern = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]*)\)")

        for match in image_pattern.finditer(text):
            alt_text, url = match.groups()
            if not alt_text.strip():
                issues.append(f"Image with empty alt text: {url[:50]}")

        for match in link_pattern.finditer(text):
            link_text, url = match.groups()
            if not link_text.strip():
                issues.append(f"Link with empty display text: {url[:50]}")
            if not url.strip():
                issues.append(f"Empty URL in link: [{link_text}]()")

    @staticmethod
    def fix_common_issues(text: str) -> str:
        text = re.sub(r"(```\w*)\n[\s]*\n(.*?)\n[\s]*\n(```)", r"\1\n\2\n\3", text, flags=re.DOTALL)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" +\n", "\n", text)
        return text
