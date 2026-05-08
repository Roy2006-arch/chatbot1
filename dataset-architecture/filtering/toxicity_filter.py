import re
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import FilterResult, FilterIssue, Severity

PROFANITY_PATTERNS = [
    r"\bf+u+c+k+\b", r"\bs+h+i+t+\b", r"\ba+s+s+\b", r"\bb+i+t+c+h+\b",
    r"\bd+a+m+n+\b", r"\bc+r+a+p+\b", r"\bd+i+c+k+\b", r"\bc+o+c+k+\b",
    r"\bp+i+s+s+\b", r"\bs+l+u+t+\b", r"\bw+h+o+r+e+\b", r"\bb+a+s+t+a+r+d+\b",
]

HATE_SPEECH_PATTERNS = [
    r"(?i)\br\s*[a4]c\s*[1i]s[mt]\b",
    r"(?i)\bn\s*[1i]g{1,2}[a4e3]+\w*\b",
    r"(?i)\bf[a4]gg[o0]t\w*\b",
    r"(?i)\bsp[i1]c\w*\b",
    r"(?i)\bk\s*[1i]k\s*[e3]\w*\b",
    r"(?i)\bwhite\s+(supremac\w*|power|genocide)\b",
    r"(?i)\bblack (inferior|subhuman|genocide)\b",
    r"(?i)\b(heil|hail) hitler\b",
    r"(?i)\bnazi\w*\b.*\b(proud|superior|great)\b",
]

HARASSMENT_PATTERNS = [
    r"(?i)\byou (should|must|need to) (kill|hurt|harm) (yourself|yourselves)\b",
    r"(?i)\b(go|just) (kill|hurt|harm) yourself\b",
    r"(?i)\b(end|take|finish) your (life|own life)\b",
    r"(?i)\byou['']?re (worthless|useless|pathetic|garbage|trash)\b",
    r"(?i)\bnobody (wants|loves|cares about) you\b",
    r"(?i)\byou should (die|disappear)\b",
    r"(?i)\b(bully|bullying|harassment|harassing)\b",
]

SELF_HARM_PATTERNS = [
    r"(?i)\b(how to|ways to|methods? to) (commit |)suicide\b",
    r"(?i)\bbest (ways?|methods?) to (kill|hurt) yourself\b",
    r"(?i)\b(self.?harm|self.?hurt)\b",
    r"(?i)\bcutting (myself|yourself|oneself)\b",
    r"(?i)\bwant(?:s|ed)? to die\b",
    r"(?i)\bsuicide (notes?|methods?|attempt|prevention|hotline)\b",
    r"(?i)\b(overdose|hang myself|jump off)\b",
]

VIOLENCE_PATTERNS = [
    r"(?i)\b(kill|murder|assassinate|execute)\s+(someone|people|them|him|her|everyone)\b",
    r"(?i)\b(how to|ways? to) (torture|murder|assassinate)\b",
    r"(?i)\bmass shooting\b", r"(?i)\bterrorist (attack|act)\b",
    r"(?i)\bbomb (making|instructions|recipe)\b",
    r"(?i)\b(chemical|biological|nuclear) (weapon|attack|warfare)\b",
    r"(?i)\b(behead|decapitate|eviscerate|disembowel)\b",
    r"(?i)\b(rape|sexual assault|molest)\b",
    r"(?i)\b(genocide|ethnic cleansing|war crime)\b",
]

NSFW_PATTERNS = [
    r"(?i)\bexplicit (content|material|images?|videos?)\b",
    r"(?i)\bporn\w*\b", r"(?i)\bnudity\b", r"(?i)\b(sexual|erotic)\s+(content|act|explicit)\b",
    r"(?i)\bxxx\b", r"(?i)\badult (content|material|entertainment)\b",
]

TOXICITY_WEIGHT_MAP = {
    "profanity": PROFANITY_PATTERNS,
    "hate_speech": HATE_SPEECH_PATTERNS,
    "harassment": HARASSMENT_PATTERNS,
    "self_harm": SELF_HARM_PATTERNS,
    "violence": VIOLENCE_PATTERNS,
    "nsfw": NSFW_PATTERNS,
}


class ToxicityFilter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.thresholds = {
            "profanity": self.config.get("profanity_threshold", 0.5),
            "hate_speech": self.config.get("hate_speech_threshold", 0.5),
            "harassment": self.config.get("harassment_threshold", 0.5),
            "self_harm": self.config.get("self_harm_threshold", 0.3),
            "violence": self.config.get("violence_threshold", 0.5),
            "nsfw": self.config.get("nsfw_threshold", 0.5),
        }
        self.stats = {"checked": 0, "flagged_toxic": 0}

    def check(self, text: str) -> FilterResult:
        self.stats["checked"] += 1
        issues: List[FilterIssue] = []
        dim_scores = {}

        if not text.strip():
            return FilterResult(passed=True, score=1.0)

        for category, patterns in TOXICITY_WEIGHT_MAP.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text)
                matches.extend(found)

            if matches:
                severity_map = {
                    "profanity": Severity.MEDIUM,
                    "hate_speech": Severity.CRITICAL,
                    "harassment": Severity.HIGH,
                    "self_harm": Severity.CRITICAL,
                    "violence": Severity.CRITICAL,
                    "nsfw": Severity.HIGH,
                }
                threshold = self.thresholds.get(category, 0.5)
                score = min(1.0, len(matches) * 0.6)
                dim_scores[category] = 1.0 - score

                if score >= threshold:
                    issues.append(FilterIssue(
                        code=f"TOXICITY_{category.upper()}",
                        message=f"Toxic content detected: {category} ({len(matches)} matches)",
                        severity=severity_map.get(category, Severity.HIGH),
                        dimension="toxicity",
                        details={"category": category, "matches": matches[:5], "score": score},
                    ))
            else:
                dim_scores[category] = 1.0

        composite = sum(dim_scores.values()) / max(len(dim_scores), 1)
        critical = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        passed = len(critical) == 0
        if not passed:
            self.stats["flagged_toxic"] += 1

        return FilterResult(
            passed=passed,
            score=composite,
            issues=issues,
            dimension_scores=dim_scores,
            metadata={"toxicity_breakdown": {k: round(v, 4) for k, v in dim_scores.items()}},
        )

    def check_batch(self, texts: List[str], num_workers: int = 8) -> List[FilterResult]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.check, t) for t in texts]
            return [f.result() for f in as_completed(futures)]

    def get_stats(self) -> Dict:
        return self.stats
