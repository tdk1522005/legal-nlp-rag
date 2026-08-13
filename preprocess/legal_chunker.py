import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


class LegalChunker:
    """
    Chuyển cây văn bản pháp luật thành các legal chunk.

    Quy tắc chính:
    - Điều không có Khoản: tạo chunk ARTICLE.
    - Điều có Khoản: mỗi Khoản tạo chunk CLAUSE.
    - Điểm nằm trong Khoản tương ứng.
    - Nội dung dài được chia thành nhiều chunk nhỏ,
      nhưng vẫn lặp lại ngữ cảnh Điều.
    """

    STRUCTURE_TYPES = {
        "PART",
        "CHAPTER",
        "SECTION",
        "SUBSECTION",
    }

    def __init__(
        self,
        max_chars: int = 1800,
        min_chars: int = 40,
    ) -> None:
        if max_chars < 500:
            raise ValueError(
                "max_chars nên lớn hơn hoặc bằng 500."
            )

        self.max_chars = max_chars
        self.min_chars = min_chars

    # =====================================================
    # 1. CHUẨN HÓA VĂN BẢN
    # =====================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(text),
        ).strip()

    @staticmethod
    def normalize_path(path: str) -> str:
        return str(path).replace("\\", "/")

    # =====================================================
    # 2. DUYỆT CÁC ĐIỀU TRONG CÂY
    # =====================================================

    def iter_articles(
        self,
        node: dict[str, Any],
        ancestors: list[dict[str, Any]] | None = None,
    ) -> Iterator[
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
        ]
    ]:
        """
        Duyệt cây và trả về:

        article:
            Node Điều.

        ancestors:
            Danh sách Phần, Chương, Mục, Tiểu mục
            chứa Điều đó.
        """
        if ancestors is None:
            ancestors = []

        node_type = node.get("node_type")

        current_ancestors = list(ancestors)

        if node_type in self.STRUCTURE_TYPES:
            current_ancestors.append(node)

        if node_type == "ARTICLE":
            yield node, ancestors

        for child in node.get("children", []):
            yield from self.iter_articles(
                node=child,
                ancestors=current_ancestors,
            )

    # =====================================================
    # 3. TẠO BREADCRUMB
    # =====================================================

    def format_structure_node(
        self,
        node: dict[str, Any],
    ) -> str:
        heading = self.normalize_text(
            node.get("heading", "")
        )

        title = self.normalize_text(
            node.get("title", "")
        )

        if heading and title:
            if title.casefold() not in heading.casefold():
                return f"{heading} - {title}"

        return heading or title

    def build_breadcrumb(
        self,
        ancestors: list[dict[str, Any]],
    ) -> list[str]:
        breadcrumb: list[str] = []

        for node in ancestors:
            label = self.format_structure_node(
                node
            )

            if label:
                breadcrumb.append(label)

        return breadcrumb

    # =====================================================
    # 4. TÁCH VĂN BẢN QUÁ DÀI
    # =====================================================

    def split_oversized_text(
        self,
        text: str,
        limit: int,
    ) -> list[str]:
        """
        Tách một paragraph quá dài theo câu.

        Nếu một câu vẫn quá dài thì tiếp tục tách
        theo từ.
        """
        text = self.normalize_text(text)

        if len(text) <= limit:
            return [text]

        sentences = re.split(
            r"(?<=[.;:!?])\s+",
            text,
        )

        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = self.normalize_text(
                sentence
            )

            if not sentence:
                continue

            if len(sentence) > limit:
                if current:
                    pieces.append(current)
                    current = ""

                words = sentence.split()
                word_group = ""

                for word in words:
                    candidate = (
                        f"{word_group} {word}".strip()
                    )

                    if (
                        word_group
                        and len(candidate) > limit
                    ):
                        pieces.append(word_group)
                        word_group = word
                    else:
                        word_group = candidate

                if word_group:
                    pieces.append(word_group)

                continue

            candidate = (
                f"{current} {sentence}".strip()
            )

            if current and len(candidate) > limit:
                pieces.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            pieces.append(current)

        return pieces

    def split_units(
        self,
        header_lines: list[str],
        content_units: list[str],
    ) -> list[list[str]]:
        """
        Gom các đơn vị nội dung thành nhiều nhóm,
        bảo đảm tổng độ dài gần max_chars.

        Header không nằm trong kết quả vì sẽ được
        lặp lại cho từng chunk.
        """
        clean_headers = [
            self.normalize_text(line)
            for line in header_lines
            if self.normalize_text(line)
        ]

        header_length = sum(
            len(line) + 1
            for line in clean_headers
        )

        available_length = max(
            self.max_chars - header_length - 20,
            300,
        )

        expanded_units: list[str] = []

        for unit in content_units:
            unit = self.normalize_text(unit)

            if not unit:
                continue

            expanded_units.extend(
                self.split_oversized_text(
                    text=unit,
                    limit=available_length,
                )
            )

        groups: list[list[str]] = []
        current_group: list[str] = []
        current_length = 0

        for unit in expanded_units:
            unit_length = len(unit) + 1

            if (
                current_group
                and current_length + unit_length
                > available_length
            ):
                groups.append(current_group)
                current_group = [unit]
                current_length = unit_length
            else:
                current_group.append(unit)
                current_length += unit_length

        if current_group:
            groups.append(current_group)

        return groups

    # =====================================================
    # 5. THÔNG TIN TRÍCH DẪN
    # =====================================================

    @staticmethod
    def build_citation(
        law_title: str,
        article_number: str,
        clause_number: str | None = None,
    ) -> str:
        if clause_number is not None:
            return (
                f"Khoản {clause_number} "
                f"Điều {article_number} "
                f"{law_title}"
            )

        return (
            f"Điều {article_number} "
            f"{law_title}"
        )

    # =====================================================
    # 6. TẠO HEADER CHO CHUNK
    # =====================================================

    def build_header_lines(
        self,
        law_metadata: dict[str, Any],
        breadcrumb: list[str],
        article: dict[str, Any],
    ) -> list[str]:
        law_title = self.normalize_text(
            law_metadata.get("title", "")
        )

        law_number = self.normalize_text(
            law_metadata.get("law_number", "")
        )

        article_heading = self.normalize_text(
            article.get("heading", "")
        )

        header_lines: list[str] = []

        if law_title and law_number:
            header_lines.append(
                f"{law_title} ({law_number})"
            )
        elif law_title:
            header_lines.append(law_title)

        if breadcrumb:
            header_lines.append(
                " > ".join(breadcrumb)
            )

        if article_heading:
            header_lines.append(article_heading)

        return header_lines

    # =====================================================
    # 7. TẠO MỘT CHUNK
    # =====================================================

    def create_chunk(
        self,
        *,
        law_metadata: dict[str, Any],
        parsed_document: dict[str, Any],
        article: dict[str, Any],
        breadcrumb: list[str],
        chunk_type: str,
        content_units: list[str],
        node_id: str,
        clause_number: str | None,
        point_numbers: list[str],
        part_index: int,
        part_total: int,
    ) -> dict[str, Any]:
        law_id = law_metadata["law_id"]
        law_title = law_metadata["title"]
        article_number = str(
            article.get("number", "")
        )

        header_lines = self.build_header_lines(
            law_metadata=law_metadata,
            breadcrumb=breadcrumb,
            article=article,
        )

        clean_content = [
            self.normalize_text(unit)
            for unit in content_units
            if self.normalize_text(unit)
        ]

        text_lines = header_lines + clean_content
        text = "\n".join(text_lines).strip()

        if chunk_type == "CLAUSE":
            base_chunk_id = (
                f"{law_id}/article/{article_number}"
                f"/clause/{clause_number}"
            )
        else:
            base_chunk_id = (
                f"{law_id}/article/{article_number}"
            )

        chunk_id = (
            f"{base_chunk_id}/chunk/{part_index}"
        )

        source_file = self.normalize_path(
            parsed_document.get(
                "source_file",
                "",
            )
        )

        citation = self.build_citation(
            law_title=law_title,
            article_number=article_number,
            clause_number=clause_number,
        )

        if part_total > 1:
            citation = (
                f"{citation}, phần {part_index}/{part_total}"
            )

        return {
            "chunk_id": chunk_id,
            "law_id": law_id,
            "law_number": law_metadata.get(
                "law_number"
            ),
            "law_title": law_title,
            "short_title": law_metadata.get(
                "short_title"
            ),
            "document_type": law_metadata.get(
                "document_type"
            ),
            "document_role": law_metadata.get(
                "document_role"
            ),
            "legal_domains": law_metadata.get(
                "legal_domains",
                [],
            ),
            "topics": law_metadata.get(
                "topics",
                [],
            ),
            "status": law_metadata.get("status"),
            "status_scope": law_metadata.get(
                "status_scope"
            ),
            "effective_from": law_metadata.get(
                "effective_from"
            ),
            "effective_to": law_metadata.get(
                "effective_to"
            ),
            "default_retrieval": law_metadata.get(
                "default_retrieval",
                False,
            ),
            "source_file": source_file,
            "is_consolidated": (
                "/consolidated/" in source_file
            ),
            "node_id": node_id,
            "chunk_type": chunk_type,
            "article_number": article_number,
            "article_title": article.get(
                "title",
                "",
            ),
            "clause_number": clause_number,
            "point_numbers": point_numbers,
            "breadcrumb": breadcrumb,
            "paragraph_index": article.get(
                "paragraph_index"
            ),
            "chunk_part": part_index,
            "chunk_part_total": part_total,
            "citation": citation,
            "content": "\n".join(
                clean_content
            ).strip(),
            "text": text,
            "char_count": len(text),
        }

    # =====================================================
    # 8. CHUNK MỘT ĐIỀU
    # =====================================================

    def chunk_article(
        self,
        *,
        article: dict[str, Any],
        ancestors: list[dict[str, Any]],
        law_metadata: dict[str, Any],
        parsed_document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        breadcrumb = self.build_breadcrumb(
            ancestors
        )

        article_paragraphs = [
            self.normalize_text(paragraph)
            for paragraph in article.get(
                "paragraphs",
                [],
            )
            if self.normalize_text(paragraph)
        ]

        clauses = [
            child
            for child in article.get(
                "children",
                [],
            )
            if child.get("node_type") == "CLAUSE"
        ]

        direct_points = [
            child
            for child in article.get(
                "children",
                [],
            )
            if child.get("node_type") == "POINT"
        ]

        header_lines = self.build_header_lines(
            law_metadata=law_metadata,
            breadcrumb=breadcrumb,
            article=article,
        )

        # -------------------------------------------------
        # Điều không có Khoản
        # -------------------------------------------------

        if not clauses:
            content_units = list(
                article_paragraphs
            )

            point_numbers: list[str] = []

            for point in direct_points:
                point_number = str(
                    point.get("number", "")
                )

                point_numbers.append(
                    point_number
                )

                content_units.extend(
                    point.get("paragraphs", [])
                )

            groups = self.split_units(
                header_lines=header_lines,
                content_units=content_units,
            )

            if not groups:
                groups = [[]]

            for part_index, group in enumerate(
                groups,
                start=1,
            ):
                chunk = self.create_chunk(
                    law_metadata=law_metadata,
                    parsed_document=parsed_document,
                    article=article,
                    breadcrumb=breadcrumb,
                    chunk_type="ARTICLE",
                    content_units=group,
                    node_id=article["node_id"],
                    clause_number=None,
                    point_numbers=point_numbers,
                    part_index=part_index,
                    part_total=len(groups),
                )

                if (
                    chunk["char_count"]
                    >= self.min_chars
                ):
                    chunks.append(chunk)

            return chunks

        # -------------------------------------------------
        # Điều có Khoản
        # -------------------------------------------------

        for clause in clauses:
            clause_number = str(
                clause.get("number", "")
            )

            content_units = list(
                article_paragraphs
            )

            content_units.extend(
                clause.get("paragraphs", [])
            )

            points = [
                child
                for child in clause.get(
                    "children",
                    [],
                )
                if child.get("node_type") == "POINT"
            ]

            point_numbers: list[str] = []

            for point in points:
                point_number = str(
                    point.get("number", "")
                )

                point_numbers.append(
                    point_number
                )

                content_units.extend(
                    point.get("paragraphs", [])
                )

            groups = self.split_units(
                header_lines=header_lines,
                content_units=content_units,
            )

            if not groups:
                groups = [[]]

            for part_index, group in enumerate(
                groups,
                start=1,
            ):
                chunk = self.create_chunk(
                    law_metadata=law_metadata,
                    parsed_document=parsed_document,
                    article=article,
                    breadcrumb=breadcrumb,
                    chunk_type="CLAUSE",
                    content_units=group,
                    node_id=clause["node_id"],
                    clause_number=clause_number,
                    point_numbers=point_numbers,
                    part_index=part_index,
                    part_total=len(groups),
                )

                if (
                    chunk["char_count"]
                    >= self.min_chars
                ):
                    chunks.append(chunk)

        return chunks

    # =====================================================
    # 9. CHUNK TOÀN BỘ VĂN BẢN
    # =====================================================

    def build_chunks(
        self,
        parsed_document: dict[str, Any],
        law_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tree = parsed_document.get("tree")

        if not isinstance(tree, dict):
            raise ValueError(
                "parsed_document không có tree hợp lệ."
            )

        if (
            parsed_document.get("law_id")
            != law_metadata.get("law_id")
        ):
            raise ValueError(
                "law_id của parsed document "
                "không khớp metadata."
            )

        chunks: list[dict[str, Any]] = []

        for article, ancestors in self.iter_articles(
            tree
        ):
            article_chunks = self.chunk_article(
                article=article,
                ancestors=ancestors,
                law_metadata=law_metadata,
                parsed_document=parsed_document,
            )

            chunks.extend(article_chunks)

        self.validate_chunks(chunks)

        return chunks

    # =====================================================
    # 10. KIỂM TRA CHUNK
    # =====================================================

    def validate_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        chunk_ids: set[str] = set()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Phát hiện chunk không có chunk_id."
                )

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"chunk_id bị trùng: {chunk_id}"
                )

            chunk_ids.add(chunk_id)

            if not chunk.get("text"):
                raise ValueError(
                    f"{chunk_id}: text đang rỗng."
                )

            if not chunk.get("article_number"):
                raise ValueError(
                    f"{chunk_id}: thiếu article_number."
                )

    # =====================================================
    # 11. THỐNG KÊ
    # =====================================================

    @staticmethod
    def summary(
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        type_counts = Counter(
            chunk.get(
                "chunk_type",
                "UNKNOWN",
            )
            for chunk in chunks
        )

        lengths = [
            chunk.get("char_count", 0)
            for chunk in chunks
        ]

        return {
            "chunk_count": len(chunks),
            "type_counts": dict(type_counts),
            "min_chars": (
                min(lengths)
                if lengths
                else 0
            ),
            "max_chars": (
                max(lengths)
                if lengths
                else 0
            ),
            "average_chars": (
                round(
                    sum(lengths) / len(lengths),
                    2,
                )
                if lengths
                else 0
            ),
        }

    # =====================================================
    # 12. LƯU JSONL
    # =====================================================

    @staticmethod
    def save_jsonl(
        chunks: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            for chunk in chunks:
                file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )