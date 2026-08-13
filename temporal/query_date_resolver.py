from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal


TemporalGranularity = Literal[
    "day",
    "month",
    "year",
    "current",
    "default",
]


@dataclass(frozen=True)
class QueryDateResolution:
    """
    Kết quả nhận diện thời điểm pháp lý trong câu hỏi.
    """

    as_of: date
    explicit: bool
    granularity: TemporalGranularity
    matched_text: str | None
    source: str
    use_temporal_index: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["as_of"] = self.as_of.isoformat()
        return result


class QueryDateResolver:
    """
    Nhận diện thời điểm pháp lý từ câu hỏi tiếng Việt.

    Hỗ trợ:
    - YYYY-MM-DD
    - DD/MM/YYYY
    - DD-MM-YYYY
    - ngày D tháng M năm YYYY
    - tháng M năm YYYY
    - vào/trong/tại năm YYYY
    - hiện nay, hiện hành, hôm nay

    Khi câu hỏi không chứa thời điểm, hệ thống dùng ngày hiện tại
    và sử dụng index pháp luật hiện hành.
    """

    ISO_DATE_PATTERN = re.compile(
        r"(?<!\d)"
        r"(?P<year>\d{4})-"
        r"(?P<month>\d{1,2})-"
        r"(?P<day>\d{1,2})"
        r"(?!\d)"
    )

    DMY_DATE_PATTERN = re.compile(
        r"(?<!\d)"
        r"(?P<day>\d{1,2})[/-]"
        r"(?P<month>\d{1,2})[/-]"
        r"(?P<year>\d{4})"
        r"(?!\d)"
    )

    VIETNAMESE_DATE_PATTERN = re.compile(
        r"\bngày\s+"
        r"(?P<day>\d{1,2})\s+"
        r"tháng\s+"
        r"(?P<month>\d{1,2})\s+"
        r"năm\s+"
        r"(?P<year>\d{4})\b",
        flags=re.IGNORECASE,
    )

    MONTH_YEAR_PATTERN = re.compile(
        r"\btháng\s+"
        r"(?P<month>\d{1,2})\s+"
        r"năm\s+"
        r"(?P<year>\d{4})\b",
        flags=re.IGNORECASE,
    )

    YEAR_PATTERNS = (
        re.compile(
            r"\b(?:vào|trong|tại|đến|tính\s+đến)"
            r"\s+(?:thời\s+điểm\s+)?"
            r"năm\s+"
            r"(?P<year>\d{4})\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^\s*năm\s+"
            r"(?P<year>\d{4})\b",
            flags=re.IGNORECASE,
        ),
    )

    CURRENT_TERMS = (
        "hiện nay",
        "hiện hành",
        "bây giờ",
        "hôm nay",
        "ngày hôm nay",
        "thời điểm hiện tại",
    )

    MIN_YEAR = 1900
    MAX_YEAR = 2100

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            query.strip().casefold(),
        )

    @staticmethod
    def _coerce_today(
        value: date | datetime | None,
    ) -> date:
        if value is None:
            return date.today()

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        raise TypeError(
            "today phải là date, datetime hoặc None."
        )

    @classmethod
    def _validate_year(
        cls,
        year: int,
    ) -> None:
        if not cls.MIN_YEAR <= year <= cls.MAX_YEAR:
            raise ValueError(
                "Năm tra cứu phải nằm trong khoảng "
                f"{cls.MIN_YEAR}-{cls.MAX_YEAR}."
            )

    @classmethod
    def _build_date(
        cls,
        *,
        year: int,
        month: int,
        day: int,
    ) -> date:
        cls._validate_year(year)

        try:
            return date(
                year=year,
                month=month,
                day=day,
            )

        except ValueError as error:
            raise ValueError(
                "Ngày được phát hiện trong câu hỏi "
                f"không hợp lệ: {day:02d}/"
                f"{month:02d}/{year}."
            ) from error

    @staticmethod
    def _uses_temporal_index(
        resolved_date: date,
        today: date,
    ) -> bool:
        return resolved_date != today

    def _resolve_full_date_match(
        self,
        match: re.Match[str],
        *,
        today: date,
    ) -> QueryDateResolution:
        resolved_date = self._build_date(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        )

        return QueryDateResolution(
            as_of=resolved_date,
            explicit=True,
            granularity="day",
            matched_text=match.group(0),
            source="explicit_date",
            use_temporal_index=(
                self._uses_temporal_index(
                    resolved_date,
                    today,
                )
            ),
        )

    def _resolve_month_year(
        self,
        match: re.Match[str],
        *,
        today: date,
    ) -> QueryDateResolution:
        year = int(match.group("year"))
        month = int(match.group("month"))

        self._validate_year(year)

        if not 1 <= month <= 12:
            raise ValueError(
                f"Tháng không hợp lệ: {month}."
            )

        last_day = calendar.monthrange(
            year,
            month,
        )[1]

        period_end = date(
            year,
            month,
            last_day,
        )

        if (
            year == today.year
            and month == today.month
            and period_end > today
        ):
            resolved_date = today
            warning = (
                "Câu hỏi chỉ nêu tháng hiện tại; "
                "hệ thống dùng ngày hôm nay."
            )

        else:
            resolved_date = period_end
            warning = (
                "Câu hỏi chỉ nêu tháng; hệ thống "
                "dùng ngày cuối cùng của tháng."
            )

        return QueryDateResolution(
            as_of=resolved_date,
            explicit=True,
            granularity="month",
            matched_text=match.group(0),
            source="month_year",
            use_temporal_index=(
                self._uses_temporal_index(
                    resolved_date,
                    today,
                )
            ),
            warning=warning,
        )

    def _resolve_year(
        self,
        match: re.Match[str],
        *,
        today: date,
    ) -> QueryDateResolution:
        year = int(match.group("year"))
        self._validate_year(year)

        if year == today.year:
            resolved_date = today
            warning = (
                "Câu hỏi chỉ nêu năm hiện tại; "
                "hệ thống dùng ngày hôm nay."
            )

        else:
            resolved_date = date(
                year,
                12,
                31,
            )
            warning = (
                "Câu hỏi chỉ nêu năm; hệ thống "
                "dùng ngày 31/12 của năm đó."
            )

        return QueryDateResolution(
            as_of=resolved_date,
            explicit=True,
            granularity="year",
            matched_text=match.group(0),
            source="year",
            use_temporal_index=(
                self._uses_temporal_index(
                    resolved_date,
                    today,
                )
            ),
            warning=warning,
        )

    def resolve(
        self,
        query: str,
        *,
        today: date | datetime | None = None,
    ) -> QueryDateResolution:
        if not isinstance(query, str):
            raise TypeError(
                "query phải là chuỗi."
            )

        clean_query = self._normalize_query(
            query
        )

        if not clean_query:
            raise ValueError(
                "query không được để trống."
            )

        current_date = self._coerce_today(
            today
        )

        for pattern in (
            self.VIETNAMESE_DATE_PATTERN,
            self.ISO_DATE_PATTERN,
            self.DMY_DATE_PATTERN,
        ):
            match = pattern.search(clean_query)

            if match is not None:
                return (
                    self._resolve_full_date_match(
                        match,
                        today=current_date,
                    )
                )

        month_year_match = (
            self.MONTH_YEAR_PATTERN.search(
                clean_query
            )
        )

        if month_year_match is not None:
            return self._resolve_month_year(
                month_year_match,
                today=current_date,
            )

        for pattern in self.YEAR_PATTERNS:
            year_match = pattern.search(
                clean_query
            )

            if year_match is not None:
                return self._resolve_year(
                    year_match,
                    today=current_date,
                )

        for term in self.CURRENT_TERMS:
            if term in clean_query:
                return QueryDateResolution(
                    as_of=current_date,
                    explicit=True,
                    granularity="current",
                    matched_text=term,
                    source="current_term",
                    use_temporal_index=False,
                )

        return QueryDateResolution(
            as_of=current_date,
            explicit=False,
            granularity="default",
            matched_text=None,
            source="default_today",
            use_temporal_index=False,
        )
