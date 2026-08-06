from __future__ import annotations

from collections import OrderedDict
from typing import Any


class ContextBuilder:
    """
    Tạo context pháp lý từ kết quả retrieval.

    Quy tắc:
    - Loại bỏ chunk_id trùng.
    - Giữ nhóm Điều có kết quả đứng cao hơn ở trước.
    - Sắp xếp các Khoản và phần chunk trong cùng Điều.
    - Đưa trích dẫn pháp lý vào context.
    """

    def __init__(
        self,
        *,
        max_chars: int = 14000,
        separator: str = "\n\n"
        + "=" * 70
        + "\n\n",
        include_score: bool = True,
    ) -> None:
        if max_chars < 1000:
            raise ValueError(
                "max_chars phải lớn hơn hoặc bằng 1000."
            )

        self.max_chars = max_chars
        self.separator = separator
        self.include_score = include_score

    @staticmethod
    def _number_sort_key(
        value: Any,
    ) -> tuple[int, str]:
        if value is None:
            return 0, ""

        text = str(value).strip()

        try:
            return int(text), text
        except ValueError:
            return 10**9, text.casefold()

    @staticmethod
    def _article_group_key(
        item: dict[str, Any],
    ) -> tuple[str, str]:
        metadata = item.get("metadata", {})

        return (
            str(metadata.get("law_id", "")),
            str(metadata.get("article_number", "")),
        )

    def _sort_article_chunks(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                self._number_sort_key(
                    item.get(
                        "metadata",
                        {},
                    ).get("clause_number")
                ),
                self._number_sort_key(
                    item.get(
                        "metadata",
                        {},
                    ).get("chunk_part")
                ),
            ),
        )

    def _format_item(
        self,
        item: dict[str, Any],
    ) -> str:
        metadata = item.get("metadata", {})

        citation = str(
            metadata.get("citation")
            or "Không rõ trích dẫn"
        )

        law_id = str(
            metadata.get("law_id", "")
        )

        chunk_id = str(
            metadata.get("chunk_id", "")
        )

        text = str(
            item.get("text", "")
        ).strip()

        header_lines = [
            f"[TRÍCH DẪN: {citation}]",
        ]

        details: list[str] = []

        if law_id:
            details.append(
                f"law_id={law_id}"
            )

        if chunk_id:
            details.append(
                f"chunk_id={chunk_id}"
            )

        if self.include_score:
            score = item.get("score")

            if score is not None:
                details.append(
                    f"score={float(score):.6f}"
                )

        if details:
            header_lines.append(
                "[" + " | ".join(details) + "]"
            )

        return (
            "\n".join(header_lines)
            + "\n"
            + text
        ).strip()

    def build(
        self,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        if not retrieved_docs:
            return ""

        unique_items: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for item in retrieved_docs:
            metadata = item.get(
                "metadata",
                {},
            )

            chunk_id = str(
                metadata.get("chunk_id", "")
            )

            if chunk_id:
                if chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)

            if not str(
                item.get("text", "")
            ).strip():
                continue

            unique_items.append(item)

        grouped: OrderedDict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = OrderedDict()

        for item in unique_items:
            group_key = self._article_group_key(
                item
            )

            grouped.setdefault(
                group_key,
                [],
            ).append(item)

        blocks: list[str] = []
        current_length = 0

        for article_items in grouped.values():
            sorted_items = self._sort_article_chunks(
                article_items
            )

            for item in sorted_items:
                block = self._format_item(item)

                added_length = len(block)

                if blocks:
                    added_length += len(
                        self.separator
                    )

                if (
                    blocks
                    and current_length + added_length
                    > self.max_chars
                ):
                    return self.separator.join(
                        blocks
                    )

                blocks.append(block)
                current_length += added_length

        return self.separator.join(blocks)