import argparse
import re
import sys
from pathlib import Path

from docx import Document


# =========================================================
# 1. THƯ MỤC PROJECT
# =========================================================

RAG_MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RAG_MODEL_DIR / "data"


# =========================================================
# 2. CÁC MẪU CẤU TRÚC PHÁP LUẬT
# =========================================================

PART_PATTERN = re.compile(
    r"^\s*PHẦN\s+",
    flags=re.IGNORECASE,
)

CHAPTER_PATTERN = re.compile(
    r"^\s*CHƯƠNG\s+",
    flags=re.IGNORECASE,
)

SECTION_PATTERN = re.compile(
    r"^\s*MỤC\s+\d+",
    flags=re.IGNORECASE,
)

SUBSECTION_PATTERN = re.compile(
    r"^\s*TIỂU MỤC\s+\d+",
    flags=re.IGNORECASE,
)

ARTICLE_PATTERN = re.compile(
    r"^\s*Điều\s+\d+[a-zA-Z]?\s*[.:]?",
    flags=re.IGNORECASE,
)

CLAUSE_PATTERN = re.compile(
    r"^\s*\d+\.\s+",
)

POINT_PATTERN = re.compile(
    r"^\s*[a-zA-ZđĐ]\)\s+",
)


# =========================================================
# 3. PHÂN LOẠI PARAGRAPH
# =========================================================

def classify_paragraph(text: str) -> str:
    """
    Xác định một paragraph thuộc loại cấu trúc nào.

    Kết quả có thể là:
        PART
        CHAPTER
        SECTION
        SUBSECTION
        ARTICLE
        CLAUSE
        POINT
        TEXT
        EMPTY
    """
    text = text.strip()

    if not text:
        return "EMPTY"

    if PART_PATTERN.match(text):
        return "PART"

    if CHAPTER_PATTERN.match(text):
        return "CHAPTER"

    if SUBSECTION_PATTERN.match(text):
        return "SUBSECTION"

    if SECTION_PATTERN.match(text):
        return "SECTION"

    if ARTICLE_PATTERN.match(text):
        return "ARTICLE"

    if CLAUSE_PATTERN.match(text):
        return "CLAUSE"

    if POINT_PATTERN.match(text):
        return "POINT"

    return "TEXT"


# =========================================================
# 4. RÚT GỌN VĂN BẢN KHI IN
# =========================================================

def shorten_text(
    text: str,
    max_length: int = 160,
) -> str:
    """
    Rút gọn nội dung paragraph để terminal dễ đọc.
    """
    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(normalized) <= max_length:
        return normalized

    return normalized[:max_length] + "..."


# =========================================================
# 5. KIỂM TRA FILE DOCX
# =========================================================

def inspect_docx(
    file_path: Path,
    start: int = 0,
    limit: int = 120,
    show_empty: bool = False,
) -> None:
    """
    In paragraph index, loại cấu trúc, style Word
    và nội dung paragraph.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    if file_path.suffix.lower() != ".docx":
        raise ValueError(
            f"Chỉ hỗ trợ .docx, nhận được: {file_path.suffix}"
        )

    document = Document(file_path)

    paragraphs = document.paragraphs
    end = min(
        start + limit,
        len(paragraphs),
    )

    type_counts: dict[str, int] = {}

    print("=" * 100)
    print("KIỂM TRA CẤU TRÚC DOCX")
    print("=" * 100)
    print(f"File: {file_path}")
    print(f"Tổng paragraph: {len(paragraphs)}")
    print(f"Đang hiển thị: {start} → {end - 1}")
    print("=" * 100)

    for index in range(start, end):
        paragraph = paragraphs[index]
        text = paragraph.text.strip()

        paragraph_type = classify_paragraph(text)

        type_counts[paragraph_type] = (
            type_counts.get(paragraph_type, 0) + 1
        )

        if paragraph_type == "EMPTY" and not show_empty:
            continue

        style_name = (
            paragraph.style.name
            if paragraph.style is not None
            else "Không rõ"
        )

        display_text = shorten_text(text)

        print(
            f"[{index:04d}] "
            f"[{paragraph_type:<10}] "
            f"[Style: {style_name:<20}] "
            f"{display_text}"
        )

    print("\n" + "=" * 100)
    print("THỐNG KÊ TRONG PHẠM VI ĐÃ HIỂN THỊ")
    print("=" * 100)

    for paragraph_type, count in sorted(
        type_counts.items()
    ):
        print(
            f"{paragraph_type:<12}: {count}"
        )


# =========================================================
# 6. XỬ LÝ THAM SỐ TERMINAL
# =========================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra paragraph và style "
            "của văn bản pháp luật DOCX."
        )
    )

    parser.add_argument(
        "file",
        type=str,
        help=(
            "Đường dẫn file tính từ rag_model/data, "
            "hoặc đường dẫn tuyệt đối."
        ),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Paragraph bắt đầu.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Số paragraph cần hiển thị.",
    )

    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Hiển thị cả paragraph rỗng.",
    )

    return parser.parse_args()


def resolve_file_path(
    file_argument: str,
) -> Path:
    """
    Nếu người dùng truyền đường dẫn tương đối,
    đường dẫn sẽ được tính từ rag_model/data.
    """
    path = Path(file_argument)

    if path.is_absolute():
        return path

    return DATA_DIR / path


# =========================================================
# 7. CHƯƠNG TRÌNH CHÍNH
# =========================================================

def main() -> None:
    arguments = parse_arguments()

    file_path = resolve_file_path(
        arguments.file
    )

    try:
        inspect_docx(
            file_path=file_path,
            start=arguments.start,
            limit=arguments.limit,
            show_empty=arguments.show_empty,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    except Exception as error:
        print(
            "[ERROR] Không thể đọc file DOCX:"
        )
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()