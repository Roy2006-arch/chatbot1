import re
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity


class AdvancedMarkdownValidator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_heading_depth = self.config.get("max_heading_depth", 6)
        self.max_table_columns = self.config.get("max_table_columns", 20)
        self.max_list_nesting = self.config.get("max_list_nesting", 6)
        self.stats = {"checked": 0, "valid": 0, "invalid": 0}

    def check(self, text: str) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}
        lines = text.split("\n")

        fence_issues = self._check_code_fences(text)
        issues.extend(fence_issues)
        dim_scores["code_fences"] = 1.0 - len(fence_issues) * 0.25

        heading_issues = self._check_headings(lines)
        issues.extend(heading_issues)
        dim_scores["headings"] = 1.0 - len(heading_issues) * 0.25

        if self.config.get("check_heading_order", True):
            order_issues = self._check_heading_order(lines)
            issues.extend(order_issues)
            dim_scores["heading_order"] = 1.0 - len(order_issues) * 0.2

        table_issues = self._check_tables(lines)
        issues.extend(table_issues)
        dim_scores["tables"] = 1.0 - len(table_issues) * 0.2

        list_issues = self._check_lists(lines)
        issues.extend(list_issues)
        dim_scores["lists"] = 1.0 - len(list_issues) * 0.25

        if self.config.get("check_html_closure", True):
            html_issues = self._check_html(text)
            issues.extend(html_issues)
            dim_scores["html"] = 1.0 - len(html_issues) * 0.25

        link_issues = self._check_links(text)
        issues.extend(link_issues)
        dim_scores["links"] = 1.0 - len(link_issues) * 0.2

        image_issues = self._check_images(text)
        issues.extend(image_issues)
        dim_scores["images"] = 1.0 - len(image_issues) * 0.25

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1)
        critical = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        passed = len(critical) == 0

        if passed:
            self.stats["valid"] += 1
        else:
            self.stats["invalid"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"line_count": len(lines), "issue_count": len(issues)},
        )

    def check_batch(self, texts: List[str], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, t) for t in texts]
            return [f.result() for f in as_completed(futures)]

    def _check_code_fences(self, text: str) -> List[FilterIssue]:
        issues = []
        fence_count = text.count("```")
        if fence_count % 2 != 0:
            issues.append(FilterIssue(
                code="MD_UNMATCHED_FENCES",
                message=f"Unmatched code fences: {fence_count} (odd number)",
                severity=Severity.HIGH,
                dimension="markdown",
            ))

        blocks = re.findall(r"```(\w*)\n.*?```", text, re.DOTALL)
        for i, block in enumerate(blocks):
            if len(block.strip()) > 0 and not re.search(r"\n", block.strip()[:100]):
                issues.append(FilterIssue(
                    code="MD_INLINE_CODE_FENCE",
                    message=f"Code block {i+1} appears to be inline (missing newline)",
                    severity=Severity.LOW,
                    dimension="markdown",
                ))

        return issues

    def _check_headings(self, lines: List[str]) -> List[FilterIssue]:
        issues = []
        heading_pattern = re.compile(r"^(#{1,7})\s+")
        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                depth = len(match.group(1))
                if depth > self.max_heading_depth:
                    issues.append(FilterIssue(
                        code="MD_HEADING_DEPTH",
                        message=f"Line {i+1}: Heading depth {depth} exceeds max {self.max_heading_depth}",
                        severity=Severity.MEDIUM,
                        dimension="markdown",
                    ))
                after_hash = line[len(match.group(1)):]
                if after_hash.strip() == "":
                    issues.append(FilterIssue(
                        code="MD_EMPTY_HEADING",
                        message=f"Line {i+1}: Empty heading text",
                        severity=Severity.MEDIUM,
                        dimension="markdown",
                    ))
        return issues

    def _check_heading_order(self, lines: List[str]) -> List[FilterIssue]:
        issues = []
        heading_pattern = re.compile(r"^(#{1,6})\s+")
        last_depth = 0
        h1_found = False

        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                depth = len(match.group(1))
                if depth == 1:
                    h1_found = True
                if depth > last_depth + 1 and last_depth > 0:
                    issues.append(FilterIssue(
                        code="MD_HEADING_SKIP",
                        message=f"Line {i+1}: Heading jumps from level {last_depth} to {depth} (skipped level {last_depth + 1})",
                        severity=Severity.LOW,
                        dimension="markdown",
                    ))
                last_depth = depth

        return issues

    def _check_tables(self, lines: List[str]) -> List[FilterIssue]:
        issues = []
        in_table = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cols = [c for c in stripped.split("|") if c.strip()]
                if len(cols) > self.max_table_columns:
                    issues.append(FilterIssue(
                        code="MD_TABLE_COLUMNS",
                        message=f"Line {i+1}: Table has {len(cols)} columns (max {self.max_table_columns})",
                        severity=Severity.MEDIUM,
                        dimension="markdown",
                    ))

                if re.match(r"^\|[\s:-]+\|$", stripped):
                    if not in_table:
                        issues.append(FilterIssue(
                            code="MD_TABLE_NO_HEADER",
                            message=f"Line {i+1}: Separator row without header",
                            severity=Severity.MEDIUM,
                            dimension="markdown",
                        ))
                    in_table = False
                else:
                    in_table = True

                if stripped.count("|") < 2:
                    issues.append(FilterIssue(
                        code="MD_TABLE_SYNTAX",
                        message=f"Line {i+1}: Malformed table (too few columns)",
                        severity=Severity.LOW,
                        dimension="markdown",
                    ))
            else:
                in_table = False

        return issues

    def _check_lists(self, lines: List[str]) -> List[FilterIssue]:
        issues = []
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip())
            if re.match(r"^[\s]*[-*+]\s", stripped) or re.match(r"^[\s]*\d+\.\s", stripped):
                current_nesting = indent // 2
                if current_nesting > self.max_list_nesting:
                    issues.append(FilterIssue(
                        code="MD_LIST_NESTING",
                        message=f"Line {i+1}: List nesting depth {current_nesting} exceeds max {self.max_list_nesting}",
                        severity=Severity.LOW,
                        dimension="markdown",
                    ))
        return issues

    def _check_html(self, text: str) -> List[FilterIssue]:
        issues = []
        void_elements = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
        open_tags = re.findall(r"<(\w+)[^>]*>", text)
        close_tags = re.findall(r"</(\w+)>", text)

        tag_counts: Dict[str, int] = {}
        for tag in open_tags:
            if tag.lower() not in void_elements:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for tag in close_tags:
            if tag.lower() not in void_elements:
                tag_counts[tag] = tag_counts.get(tag, 0) - 1

        unclosed = [tag for tag, count in tag_counts.items() if count > 0]
        if unclosed:
            issues.append(FilterIssue(
                code="MD_UNCLOSED_HTML",
                message=f"Potentially unclosed HTML tags: {', '.join(unclosed[:5])}",
                severity=Severity.MEDIUM,
                dimension="markdown",
                details={"unclosed_tags": unclosed[:10]},
            ))

        return issues

    def _check_links(self, text: str) -> List[FilterIssue]:
        issues = []
        link_pattern = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]*)\)")

        for match in link_pattern.finditer(text):
            link_text, url = match.groups()
            if not link_text.strip():
                issues.append(FilterIssue(
                    code="MD_EMPTY_LINK_TEXT",
                    message=f"Link with empty display text: {url[:50]}",
                    severity=Severity.LOW,
                    dimension="markdown",
                ))
            if not url.strip():
                issues.append(FilterIssue(
                    code="MD_EMPTY_URL",
                    message=f"Empty URL in link: [{link_text}]()",
                    severity=Severity.MEDIUM,
                    dimension="markdown",
                ))
            if url.startswith("http"):
                if not re.match(r"^https?://[^\s]+$", url):
                    issues.append(FilterIssue(
                        code="MD_INVALID_URL",
                        message=f"Possibly invalid URL: {url[:50]}",
                        severity=Severity.LOW,
                        dimension="markdown",
                    ))

        return issues

    def _check_images(self, text: str) -> List[FilterIssue]:
        issues = []
        image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")

        for match in image_pattern.finditer(text):
            alt_text, url = match.groups()
            if not alt_text.strip():
                issues.append(FilterIssue(
                    code="MD_EMPTY_ALT",
                    message=f"Image with empty alt text: {url[:50]}",
                    severity=Severity.LOW,
                    dimension="markdown",
                ))

        return issues

    def fix_common_issues(self, text: str) -> str:
        text = re.sub(r"(```\w*)\n[\s]*\n(.*?)\n[\s]*\n(```)", r"\1\n\2\n\3", text, flags=re.DOTALL)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"#{7,}\s", "#" * self.max_heading_depth + " ", text)
        return text

    def get_stats(self) -> Dict:
        return self.stats
