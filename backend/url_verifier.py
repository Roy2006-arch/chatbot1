from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("url_verifier")

URL_PATTERN = re.compile(
    r"https?://[\w.-]+(?::\d+)?(?:/[\w./~%!$&'()*+,;=:@?-]*)?",
    re.IGNORECASE,
)

OFFICIAL_DOMAINS: dict[str, set[str]] = {
    "python.org": {"docs.python.org", "pypi.org", "peps.python.org", "bugs.python.org"},
    "readthedocs.io": set(),
    "github.com": {"github.com", "raw.githubusercontent.com"},
    "gitlab.com": set(),
    "stackoverflow.com": {"stackoverflow.com"},
    "stackexchange.com": {"stackoverflow.com", "serverfault.com", "superuser.com"},
    "w3.org": {"w3.org", "www.w3.org"},
    "mozilla.org": {"developer.mozilla.org", "mdn.dev"},
    "microsoft.com": {"docs.microsoft.com", "learn.microsoft.com", "msdn.microsoft.com"},
    "nodejs.org": {"nodejs.org"},
    "npmjs.com": {"docs.npmjs.com", "www.npmjs.com"},
    "docker.com": {"docs.docker.com"},
    "kubernetes.io": {"kubernetes.io"},
    "linux.org": {"kernel.org", "linux.die.net", "man7.org"},
    "oracle.com": {"docs.oracle.com"},
    "ibm.com": {"cloud.ibm.com"},
    "aws.amazon.com": {"docs.aws.amazon.com"},
    "google.com": {"developers.google.com", "cloud.google.com"},
    "jetbrains.com": {"jetbrains.com", "plugins.jetbrains.com"},
    "postgresql.org": {"postgresql.org"},
    "sqlite.org": {"sqlite.org"},
    "mongodb.com": {"docs.mongodb.com"},
    "redis.io": {"redis.io"},
    "nginx.org": {"nginx.org"},
    "apache.org": {"apache.org"},
    "cncf.io": {"cncf.io"},
    "ietf.org": {"ietf.org", "tools.ietf.org", "datatracker.ietf.org"},
    "rfc-editor.org": {"rfc-editor.org"},
    "json.org": {"json.org"},
    "yaml.org": {"yaml.org"},
    "markdownguide.org": {"markdownguide.org"},
}

FLATTENED_OFFICIAL: set[str] = set()
for group, domains in OFFICIAL_DOMAINS.items():
    FLATTENED_OFFICIAL.update(domains)
    FLATTENED_OFFICIAL.add(group)

KNOWN_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "shorturl.at",
    "ow.ly", "is.gd", "buff.ly", "tiny.cc", "bl.ink",
}


@dataclass
class URLReport:
    url: str
    is_valid_format: bool = False
    in_whitelist: bool = False
    is_reachable: bool = False
    is_shortener: bool = False
    confidence: float = 0.0
    normalized: str = ""
    error: str = ""


@dataclass
class ResponseURLReport:
    urls: list[URLReport] = field(default_factory=list)
    has_unverified_urls: bool = False
    has_fake_urls: bool = False
    all_verified: bool = True
    worst_confidence: float = 1.0


class URLVerifier:
    def __init__(
        self,
        request_timeout: float = 3.0,
        max_concurrent_checks: int = 5,
        confidence_threshold: float = 0.6,
        max_urls_per_response: int = 15,
    ):
        self.request_timeout = request_timeout
        self.max_concurrent = max_concurrent_checks
        self.confidence_threshold = confidence_threshold
        self.max_urls = max_urls_per_response
        self._session: Optional[object] = None

    async def get_or_create_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                headers={"User-Agent": "Mozilla/5.0 (compatible; URLVerifier/1.0)"},
            )
        return self._session

    async def warmup(self):
        """Pre-create the aiohttp session at startup."""
        await self.get_or_create_session()
        logger.info("[URLVerifier] Session created.")

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def extract_urls(self, text: str) -> list[str]:
        return list(dict.fromkeys(URL_PATTERN.findall(text)))[:self.max_urls]

    def validate_format(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.netloc:
                return False
            if not re.match(r"^[\w\-\.]+(?::\d+)?$", parsed.netloc):
                parsed_net = urllib.parse.urlparse(f"https://{parsed.netloc}")
                if not parsed_net.netloc or not re.match(r"^[\w\-\.]+$", parsed_net.netloc):
                    return False
            return True
        except Exception:
            return False

    def check_whitelist(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or parsed.netloc.split(":")[0]
            hostname = hostname.lower()
            return hostname in FLATTENED_OFFICIAL or any(
                hostname.endswith("." + d) for d in FLATTENED_OFFICIAL
            )
        except Exception:
            return False

    def is_shortener(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or parsed.netloc.split(":")[0]
            return hostname.lower() in KNOWN_SHORTENERS
        except Exception:
            return False

    def compute_confidence(self, report: URLReport) -> float:
        score = 0.0
        if report.is_valid_format:
            score += 0.3
        if report.in_whitelist:
            score += 0.4
        if report.is_reachable:
            score += 0.3
        if report.is_shortener:
            score -= 0.2
        return max(0.0, min(1.0, score))

    async def verify_url(self, url: str) -> URLReport:
        report = URLReport(url=url)
        report.is_valid_format = self.validate_format(url)
        if not report.is_valid_format:
            report.confidence = 0.0
            report.error = "invalid_format"
            return report

        report.normalized = url.rstrip("/.,;:!?)")
        report.in_whitelist = self.check_whitelist(url)
        report.is_shortener = self.is_shortener(url)

        report.is_reachable = await self._check_reachable(report.normalized)
        report.confidence = self.compute_confidence(report)

        return report

    async def _check_reachable(self, url: str) -> bool:
        try:
            session = await self.get_or_create_session()
            async with session.head(url, allow_redirects=True, timeout=self.request_timeout) as resp:
                return resp.status < 500
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("HEAD %s failed: %s", url, e)
            return False

    async def verify_response(self, text: str) -> ResponseURLReport:
        raw_urls = self.extract_urls(text)
        if not raw_urls:
            return ResponseURLReport()

        sem = asyncio.Semaphore(self.max_concurrent)

        async def bounded(url: str) -> URLReport:
            async with sem:
                return await self.verify_url(url)

        reports = await asyncio.gather(*(bounded(u) for u in raw_urls))

        report = ResponseURLReport(urls=reports)
        report.has_unverified_urls = any(
            r.confidence < self.confidence_threshold and r.is_valid_format for r in reports
        )
        report.has_fake_urls = any(
            r.is_valid_format and not r.in_whitelist and not r.is_reachable
            for r in reports
        )
        worst = 1.0
        for r in reports:
            if r.is_valid_format and r.confidence < worst:
                worst = r.confidence
        report.worst_confidence = worst
        report.all_verified = worst >= self.confidence_threshold

        return report

    def has_any_urls(self, text: str) -> bool:
        return bool(URL_PATTERN.search(text))

    def generate_uncertainty_notice(self, report: URLReport) -> str:
        if report.in_whitelist and not report.is_reachable:
            return (
                f"The link {report.url} is from a known documentation domain "
                f"but could not be verified as reachable right now. "
                f"Please verify the URL independently."
            )
        if report.is_valid_format and not report.in_whitelist:
            return (
                f"The link {report.url} could not be verified against known documentation sources. "
                f"Please verify this URL before relying on it."
            )
        return (
            f"The link {report.url} may not be accurate. "
            f"Please verify it independently."
        )

    def sanitize_response(
        self, text: str, report: ResponseURLReport
    ) -> tuple[str, list[str]]:
        if report.all_verified:
            return text, []

        warnings: list[str] = []
        lines = text.split("\n")
        result_lines: list[str] = []
        seen_bad_urls: set[str] = set()

        for line in lines:
            urls_in_line = URL_PATTERN.findall(line)
            modified = line
            for url in urls_in_line:
                for r in report.urls:
                    if r.url != url:
                        continue
                    if r.confidence >= self.confidence_threshold:
                        continue
                    if url in seen_bad_urls:
                        continue
                    if not r.is_valid_format:
                        modified = modified.replace(url, f"`{url}`")
                        seen_bad_urls.add(url)
                        warnings.append(
                            f"Removed invalid URL: {url} - does not match http/https format"
                        )
                    elif not r.in_whitelist and not r.is_reachable:
                        warnings.append(
                            f"Unverifiable URL: {url} - not in official domain whitelist and unreachable"
                        )
                        seen_bad_urls.add(url)

            result_lines.append(modified)

        return "\n".join(result_lines), warnings

    @staticmethod
    def would_generate_url(
        model_output: str, instruction_category: str = ""
    ) -> bool:
        url_indicators = [
            r"https?://",
            r"www\.\w+",
            r"check\s+(?:out\s+)?(?:this\s+)?(?:link|page|documentation)",
            r"for\s+more\s+(?:information|details)\s+(?:see|visit|check|refer)",
            r"(?:see|refer|visit)\s+(?:the\s+)?(?:official\s+)?docs?(?:umentation)?",
        ]
        if any(re.search(p, model_output, re.IGNORECASE) for p in url_indicators):
            return True
        if instruction_category in ("document_query", "explanation"):
            return True
        return False

    @staticmethod
    def strip_urls(text: str) -> str:
        return URL_PATTERN.sub("[link removed]", text)

    @staticmethod
    def normalize_url(url: str) -> str:
        url = url.strip().rstrip(".,;:!?)}]>")
        parsed = urllib.parse.urlparse(url)
        normalized = parsed.scheme + "://" + parsed.netloc.lower() + parsed.path
        if parsed.query:
            normalized += "?" + parsed.query
        return normalized
