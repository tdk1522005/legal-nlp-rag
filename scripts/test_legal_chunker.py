import json
import sys
from pathlib import Path


RAG_MODEL_DIR = Path(__file__).resolve().parents[1]

if str(RAG_MODEL_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(RAG_MODEL_DIR),
    )


from preprocess.legal_chunker import LegalChunker


DATA_DIR = RAG_MODEL_DIR / "data"


def load_json(
    file_path: Path,
) -> dict:
    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_law_metadata(
    laws_data: dict,
    law_id: str,
) -> dict:
    for law in laws_data.get("laws", []):
        if law.get("law_id") == law_id:
            return law

    raise ValueError(
        f"Không tìm thấy law_id: {law_id}"
    )


def main() -> None:
    parsed_path = (
        DATA_DIR
        / "parsed"
        / "civil_code_2015.json"
    )

    laws_path = (
        DATA_DIR
        / "metadata"
        / "laws.json"
    )

    output_path = (
        DATA_DIR
        / "chunks"
        / "civil_code_2015.jsonl"
    )

    if not parsed_path.exists():
        print(
            f"Không tìm thấy parsed document: "
            f"{parsed_path}"
        )
        sys.exit(1)

    parsed_document = load_json(
        parsed_path
    )

    laws_data = load_json(
        laws_path
    )

    law_metadata = find_law_metadata(
        laws_data=laws_data,
        law_id="civil_code_2015",
    )

    chunker = LegalChunker(
        max_chars=1800,
        min_chars=40,
    )

    chunks = chunker.build_chunks(
        parsed_document=parsed_document,
        law_metadata=law_metadata,
    )

    chunker.save_jsonl(
        chunks=chunks,
        output_path=output_path,
    )

    summary = chunker.summary(chunks)

    print("=" * 75)
    print("KIỂM TRA LEGAL CHUNKER")
    print("=" * 75)

    print(
        f"Tổng số chunk: "
        f"{summary['chunk_count']}"
    )

    print(
        f"Loại chunk: "
        f"{summary['type_counts']}"
    )

    print(
        f"Độ dài nhỏ nhất: "
        f"{summary['min_chars']}"
    )

    print(
        f"Độ dài lớn nhất: "
        f"{summary['max_chars']}"
    )

    print(
        f"Độ dài trung bình: "
        f"{summary['average_chars']}"
    )

    # =====================================================
    # KIỂM TRA ĐIỀU 117
    # =====================================================

    article_117_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("article_number") == "117"
    ]

    print("\n" + "=" * 75)
    print("CHUNK CỦA ĐIỀU 117")
    print("=" * 75)

    print(
        f"Số chunk Điều 117: "
        f"{len(article_117_chunks)}"
    )

    for chunk in article_117_chunks:
        print("\n" + "-" * 75)

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Loại: {chunk['chunk_type']}"
        )

        print(
            f"Khoản: {chunk['clause_number']}"
        )

        print(
            f"Các điểm: {chunk['point_numbers']}"
        )

        print(
            f"Trích dẫn: {chunk['citation']}"
        )

        print(
            f"Số ký tự: {chunk['char_count']}"
        )

        print("\nNội dung:")
        print(chunk["text"])

    # =====================================================
    # KIỂM TRA CHUNK QUÁ DÀI
    # =====================================================

    oversized_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("char_count", 0) > 1800
    ]

    print("\n" + "=" * 75)
    print("KIỂM TRA ĐỘ DÀI")
    print("=" * 75)

    if oversized_chunks:
        print(
            f"Có {len(oversized_chunks)} chunk "
            f"dài hơn 1800 ký tự."
        )

        for chunk in oversized_chunks[:10]:
            print(
                f"- {chunk['chunk_id']}: "
                f"{chunk['char_count']}"
            )
    else:
        print(
            "Không có chunk nào dài quá "
            "1800 ký tự."
        )

    # =====================================================
    # KIỂM TRA CHUNK ID
    # =====================================================

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    duplicate_count = (
        len(chunk_ids)
        - len(set(chunk_ids))
    )

    print("\n" + "=" * 75)
    print("KIỂM TRA CHUNK ID")
    print("=" * 75)

    print(
        f"Số chunk_id trùng: "
        f"{duplicate_count}"
    )

    print("\n" + "=" * 75)
    print(f"Đã lưu tại: {output_path}")


if __name__ == "__main__":
    main()