from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Optional

logger = logging.getLogger("realtime_utils")

try:
    from zoneinfo import ZoneInfo

    _HAS_ZONEINFO = True
except ImportError:
    _HAS_ZONEINFO = False
    ZoneInfo = None

TIMEZONE_ALIASES: dict[str, str] = {
    "ist": "Asia/Kolkata",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "cst": "America/Chicago",
    "est": "America/New_York",
    "edt": "America/New_York",
    "mst": "America/Denver",
    "utc": "UTC",
    "gmt": "UTC",
    "bst": "Europe/London",
    "cet": "Europe/Berlin",
    "eet": "Europe/Helsinki",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "nzst": "Pacific/Auckland",
    "nzdt": "Pacific/Auckland",
    "hkt": "Asia/Hong_Kong",
    "sgt": "Asia/Singapore",
    "china": "Asia/Shanghai",
    "japan": "Asia/Tokyo",
    "uk": "Europe/London",
    "india": "Asia/Kolkata",
}

TIMEZONE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(what time is it|current time|tell me the time|what's the time|time now)\b", "current_time"),
    (r"\b(today'?s date|what'?s the date|current date|date today|what is the date)\b", "current_date"),
    (r"\b(what day is it|what day is today|current day|today'?s day)\b", "current_day"),
    (r"\b(current (year|month)|what (year|month) is it|this (year|month))\b", "current_year_month"),
    (r"\b(timestamp|unix timestamp|epoch time|current timestamp)\b", "timestamp"),
    (r"\btime in\b", "time_in_location"),
    (r"\bconvert\b.*\bto\b", "timezone_conversion"),
    (r"\bconvert.*(timezone|time|utc|gmt)\b", "timezone_conversion"),
    (r"\b(timezone|time zone|tz)\b", "timezone_info"),
]

LOCATION_TZ_MAP: dict[str, str] = {
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "london": "Europe/London",
    "uk": "Europe/London",
    "england": "Europe/London",
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "chicago": "America/Chicago",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "denver": "America/Denver",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "seoul": "Asia/Seoul",
    "korea": "Asia/Seoul",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "sydney": "Australia/Sydney",
    "australia": "Australia/Sydney",
    "melbourne": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
    "paris": "Europe/Paris",
    "france": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "moscow": "Europe/Moscow",
    "russia": "Europe/Moscow",
    "dubai": "Asia/Dubai",
    "uae": "Asia/Dubai",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo",
    "amsterdam": "Europe/Amsterdam",
    "rome": "Europe/Rome",
    "madrid": "Europe/Madrid",
    "stockholm": "Europe/Stockholm",
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

FORMAT_12H = 12
FORMAT_24H = 24


def _resolve_tz(tz_name: str | None = None) -> tzinfo:
    if not tz_name:
        return timezone.utc
    normalized = tz_name.strip().lower()
    mapped = TIMEZONE_ALIASES.get(normalized, tz_name)
    if _HAS_ZONEINFO:
        try:
            tz = ZoneInfo(mapped)
            return tz
        except (KeyError, TypeError, Exception):
            pass
    try:
        cleaned = mapped.replace("utc", "").replace("gmt", "").replace("+", "").strip()
        if cleaned:
            offset = int(cleaned)
        else:
            offset = 0
        return timezone(timedelta(hours=offset))
    except (ValueError, AttributeError):
        logger.warning("Could not resolve timezone '%s' (mapped='%s'), falling back to UTC", tz_name, mapped)
        return timezone.utc


def get_current_time(tz: tzinfo | None = None, fmt: int = FORMAT_12H) -> str:
    now = datetime.now(tz or timezone.utc)
    if fmt == FORMAT_24H:
        return now.strftime("%H:%M:%S")
    return now.strftime("%I:%M:%S %p").lstrip("0")


def get_current_date(tz: tzinfo | None = None) -> str:
    return datetime.now(tz or timezone.utc).strftime("%A, %B %d, %Y")


def get_current_day(tz: tzinfo | None = None) -> str:
    return datetime.now(tz or timezone.utc).strftime("%A")


def get_current_year(tz: tzinfo | None = None) -> int:
    return datetime.now(tz or timezone.utc).year


def get_current_month(tz: tzinfo | None = None) -> tuple[int, str]:
    now = datetime.now(tz or timezone.utc)
    return now.month, MONTHS[now.month - 1]


def get_current_day_number(tz: tzinfo | None = None) -> int:
    return datetime.now(tz or timezone.utc).day


def get_unix_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def get_iso_timestamp(tz: tzinfo | None = None) -> str:
    return datetime.now(tz or timezone.utc).isoformat(timespec="seconds")


def convert_timezone(
    source_tz: str,
    target_tz: str,
    time_str: str | None = None,
) -> dict[str, str]:
    src = _resolve_tz(source_tz)
    dst = _resolve_tz(target_tz)
    if time_str:
        try:
            parsed = datetime.strptime(time_str, "%H:%M")
            now = datetime.now(src)
            source_dt = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
        except ValueError:
            source_dt = datetime.now(src)
    else:
        source_dt = datetime.now(src)
    target_dt = source_dt.astimezone(dst)
    return {
        "source_timezone": source_tz,
        "source_time": source_dt.strftime("%I:%M %p").lstrip("0"),
        "source_date": source_dt.strftime("%A, %B %d, %Y"),
        "target_timezone": target_tz,
        "target_time": target_dt.strftime("%I:%M %p").lstrip("0"),
        "target_date": target_dt.strftime("%A, %B %d, %Y"),
        "utc_offset": target_dt.strftime("%z"),
    }


def get_timezone_offset(tz_name: str) -> str:
    tz = _resolve_tz(tz_name)
    now = datetime.now(tz)
    offset = now.strftime("%z")
    return f"UTC{offset[:3]}:{offset[3:]}"


def list_supported_timezones() -> list[str]:
    return sorted(set(TIMEZONE_ALIASES.values()) | set(LOCATION_TZ_MAP.values()))


class RealtimeIntentDetector:
    _PATTERNS: list[tuple[str, str]] = [
        *TIMEZONE_PATTERNS,
    ]

    def detect(self, message: str) -> tuple[str, dict]:
        msg_lower = message.lower().strip()
        for pattern, intent in self._PATTERNS:
            if re.search(pattern, msg_lower):
                params = self._extract_params(msg_lower, intent)
                return intent, params
        return "", {}

    def _extract_params(self, msg: str, intent: str) -> dict:
        params: dict = {}
        if intent in ("current_time", "time_in_location"):
            params["format"] = FORMAT_12H
            if "24" in msg or "24h" in msg or "military" in msg:
                params["format"] = FORMAT_24H
        for location, tz_name in LOCATION_TZ_MAP.items():
            if location in msg:
                params["timezone"] = tz_name
                break
        for alias, tz_name in TIMEZONE_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, msg):
                params["timezone"] = tz_name
                break
        if intent == "timezone_conversion":
            self._extract_conversion_params(msg, params)
        return params

    def _extract_conversion_params(self, msg: str, params: dict) -> None:
        tokens = re.findall(r"[A-Za-z]+", msg)
        tz_candidates = [t for t in tokens if t.lower() in TIMEZONE_ALIASES or t.lower() in LOCATION_TZ_MAP]
        if len(tz_candidates) >= 2:
            params["source_tz"] = TIMEZONE_ALIASES.get(tz_candidates[0].lower(), LOCATION_TZ_MAP.get(tz_candidates[0].lower(), tz_candidates[0]))
            params["target_tz"] = TIMEZONE_ALIASES.get(tz_candidates[1].lower(), LOCATION_TZ_MAP.get(tz_candidates[1].lower(), tz_candidates[1]))
        elif len(tz_candidates) == 1:
            params["target_tz"] = TIMEZONE_ALIASES.get(tz_candidates[0].lower(), LOCATION_TZ_MAP.get(tz_candidates[0].lower(), tz_candidates[0]))
        time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", msg)
        if time_match:
            params["time"] = time_match.group(0)


class RealtimeHandler:
    def __init__(self):
        self.detector = RealtimeIntentDetector()
        self._user_timezones: dict[str, str] = {}

    def set_user_timezone(self, session_id: str, tz_name: str) -> None:
        self._user_timezones[session_id] = tz_name

    def get_user_timezone(self, session_id: str) -> Optional[str]:
        return self._user_timezones.get(session_id)

    def _get_resolved_label(self, tz: tzinfo | None, tz_name: str | None) -> str:
        if tz is None:
            return "UTC"
        tz_str = str(tz)
        is_utc = tz_str.upper().startswith("UTC")
        if tz_name and not is_utc:
            return tz_name
        if is_utc:
            return f"UTC{tz_str[3:]}" if len(tz_str) > 3 else "UTC"
        return tz_str

    def handle(self, message: str, session_id: str = "") -> Optional[str]:
        intent, params = self.detector.detect(message)
        if not intent:
            return None

        tz_name = params.get("timezone") or self._user_timezones.get(session_id)
        tz = _resolve_tz(tz_name) if tz_name else None
        fmt = params.get("format", FORMAT_12H)
        resolved_label = self._get_resolved_label(tz, tz_name)

        if intent == "current_time":
            return self._format_time_response(tz, fmt, resolved_label, tz_name)

        if intent == "current_date":
            return self._format_date_response(tz, tz_name, resolved_label)

        if intent == "current_day":
            day = get_current_day(tz)
            if resolved_label and resolved_label != "UTC":
                return f"It's {day} in {resolved_label}."
            return f"Today is {day}."

        if intent == "current_year_month":
            year = get_current_year(tz)
            _, month_name = get_current_month(tz)
            if resolved_label and resolved_label != "UTC":
                return f"It's {month_name} {year} in {resolved_label}."
            return f"It's {month_name} {year}."

        if intent == "timestamp":
            unix = get_unix_timestamp()
            iso = get_iso_timestamp(tz)
            return f"Current timestamp is **{unix}** (Unix epoch). In ISO 8601: **{iso}**."

        if intent == "time_in_location":
            return self._format_time_response(tz, fmt, resolved_label, tz_name)

        if intent == "timezone_conversion":
            return self._format_conversion_response(params)

        if intent == "timezone_info":
            if tz_name:
                offset = get_timezone_offset(tz_name)
                return f"The timezone {tz_name} is currently {offset}."
            return "I need a timezone name to look up. For example: EST, IST, UTC, Asia/Tokyo, etc."

        return None

    def _format_time_response(self, tz: tzinfo | None, fmt: int, label: str, tz_name: str | None) -> str:
        t = get_current_time(tz, fmt)
        d = get_current_date(tz)
        if label:
            return f"It's **{t}** on **{d}** ({label})."
        return f"It's **{t}** on **{d}**."

    def _format_date_response(self, tz: tzinfo | None, tz_name: str | None, resolved_label: str = "") -> str:
        d = get_current_date(tz)
        label = resolved_label or tz_name or ""
        if label:
            return f"Today's date is **{d}** ({label})."
        return f"Today's date is **{d}**."

    def _format_conversion_response(self, params: dict) -> str:
        source_tz = params.get("source_tz", "UTC")
        target_tz = params.get("target_tz", "UTC")
        time_str = params.get("time")
        result = convert_timezone(source_tz, target_tz, time_str)
        return (
            f"**Time Conversion:**\n"
            f"- {result['source_timezone']}: {result['source_time']} on {result['source_date']}\n"
            f"- {result['target_timezone']}: {result['target_time']} on {result['target_date']}\n"
            f"- Offset: {result['utc_offset']}"
        )

    def get_realtime_context_block(self, session_id: str = "") -> str:
        tz_name = self._user_timezones.get(session_id)
        tz = _resolve_tz(tz_name) if tz_name else timezone.utc
        now = datetime.now(tz)
        resolved_label = str(tz) if hasattr(tz, '__str__') else str(tz)
        return (
            f"[REALTIME CONTEXT]\n"
            f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
            f"Timezone: {resolved_label}\n"
            f"Unix timestamp: {int(now.timestamp())}\n"
            f"[END REALTIME CONTEXT]\n"
            f"- Use the REALTIME CONTEXT above for ANY time/date questions.\n"
            f"- Do NOT guess or hallucinate time/date information.\n"
            f"- If the user asks about time/date, use the data provided above."
        )


realtime_handler = RealtimeHandler()
