"""Pure retrieval scope shared by vector, lexical and defensive post-filtering."""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


class FilterCoverageError(ValueError):
    """The bounded scope scan cannot establish complete filter coverage."""


def _next_month(value):
    return value.replace(year=value.year + 1, month=1) if value.month == 12 else value.replace(month=value.month + 1)


def date_window(date_hint: str, now: datetime):
    """Return a timezone-aware [start, end) interval; reject unknown date hints."""
    hint = date_hint.strip().lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week = today - timedelta(days=today.weekday())
    month = today.replace(day=1)
    previous_month = (month - timedelta(days=1)).replace(day=1)
    aliases = {
        "今天": (today, today + timedelta(days=1)), "today": (today, today + timedelta(days=1)),
        "昨天": (today - timedelta(days=1), today), "yesterday": (today - timedelta(days=1), today),
        "本周": (week, week + timedelta(days=7)), "这周": (week, week + timedelta(days=7)),
        "this week": (week, week + timedelta(days=7)),
        "上周": (week - timedelta(days=7), week), "last week": (week - timedelta(days=7), week),
        "本月": (month, _next_month(month)), "这个月": (month, _next_month(month)),
        "this month": (month, _next_month(month)),
        "上月": (previous_month, month), "上个月": (previous_month, month), "last month": (previous_month, month),
        "今年": (today.replace(month=1, day=1), today.replace(year=today.year + 1, month=1, day=1)),
        "this year": (today.replace(month=1, day=1), today.replace(year=today.year + 1, month=1, day=1)),
        "最近": (now - timedelta(days=14), now), "recent": (now - timedelta(days=14), now),
        "recently": (now - timedelta(days=14), now),
    }
    if hint in aliases:
        return aliases[hint]
    dates = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:至|到|to|~|—)\s*(\d{4}-\d{2}-\d{2})", hint)
    if dates:
        start = datetime.fromisoformat(dates[1]).replace(tzinfo=now.tzinfo)
        end = datetime.fromisoformat(dates[2]).replace(tzinfo=now.tzinfo) + timedelta(days=1)
        if start >= end:
            raise ValueError("date_hint 起始日期不能晚于结束日期")
        return start, end
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", hint):
        start = datetime.fromisoformat(hint).replace(tzinfo=now.tzinfo)
        return start, start + timedelta(days=1)
    if re.fullmatch(r"\d{4}-\d{2}", hint):
        start = datetime.fromisoformat(hint + "-01").replace(tzinfo=now.tzinfo)
        return start, _next_month(start)
    raise ValueError("不支持的 date_hint，请使用日历关键词或 ISO 日期/日期区间")


@dataclass(frozen=True)
class FilterSpec:
    sender: str = ""
    labels: frozenset[str] = frozenset()
    start: datetime | None = None
    end: datetime | None = None

    @classmethod
    def from_mapping(cls, filters: dict, *, now: datetime):
        if not isinstance(filters, dict):
            raise ValueError("filters 必须是对象")
        if set(filters) - {"query", "sender", "date_hint", "labels"}:
            raise ValueError("filters 包含不支持的条件")
        for key in ("query", "sender", "date_hint"):
            if key in filters and not isinstance(filters[key], str):
                raise ValueError(f"filters.{key} 必须是字符串")
        labels = filters.get("labels", [])
        if not isinstance(labels, list) or any(not isinstance(item, str) or not item.strip() for item in labels):
            raise ValueError("filters.labels 必须是非空字符串列表")
        hint = filters.get("date_hint", "").strip()
        start, end = date_window(hint, now) if hint else (None, None)
        return cls(filters.get("sender", "").strip().lower(),
                   frozenset(label.strip().lower() for label in labels), start, end)

    @property
    def active(self):
        return bool(self.sender or self.labels or self.start is not None)

    def matches(self, metadata: dict) -> bool:
        if not isinstance(metadata, dict):
            return False
        sender = " ".join(value for value in (metadata.get("sender", ""), metadata.get("sender_name", "")) if isinstance(value, str))
        if self.sender and (not isinstance(sender, str) or self.sender not in sender.lower()):
            return False
        if self.labels:
            actual = metadata.get("labels", [])
            try:
                actual = json.loads(actual) if isinstance(actual, str) else actual
                names = metadata.get("label_names", [])
                names = json.loads(names) if isinstance(names, str) else names
                if isinstance(actual, list) and isinstance(names, list):
                    actual = actual + names
            except (ValueError, TypeError):
                return False
            if not isinstance(actual, list) or not self.labels.issubset({item.strip().lower() for item in actual if isinstance(item, str)}):
                return False
        if self.start is not None:
            value = metadata.get("date", "")
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    first = dt.replace(tzinfo=self.start.tzinfo, fold=0)
                    second = dt.replace(tzinfo=self.start.tzinfo, fold=1)
                    if (first.utcoffset() != second.utcoffset() or
                            first.astimezone(timezone.utc).astimezone(self.start.tzinfo).replace(tzinfo=None) != dt):
                        raise FilterCoverageError("Index contains an ambiguous or nonexistent local date; normalize its timezone before filtering")
                    dt = dt.replace(tzinfo=self.start.tzinfo)
                if not self.start <= dt < self.end:
                    return False
            except FilterCoverageError:
                raise
            except (ValueError, TypeError, AttributeError):
                return False
        return True
