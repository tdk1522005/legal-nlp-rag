import sys
from pathlib import Path


RAG_MODEL_DIR = Path(__file__).resolve().parents[1]

if str(RAG_MODEL_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(RAG_MODEL_DIR),
    )


from preprocess.legal_parser import LegalDocxParser


DATA_DIR = RAG_MODEL_DIR / "data"


def main() -> None:
    source_path = (
        DATA_DIR
        / "raw"
        / "current"
        / "civil_guidance"
        / "21_2021_ND_CP_secured_obligations.docx"
    )

    output_path = (
        DATA_DIR
        / "parsed"
        / "secured_obligations_decree_21_2021.json"
    )

    if not source_path.exists():
        print(f"Không tìm thấy file: {source_path}")
        sys.exit(1)

    parser = LegalDocxParser()

    parsed_document = parser.parse(
        file_path=source_path,
        law_id="secured_obligations_decree_21_2021",
        document_title=(
            "Nghị định 21/2021/NĐ-CP quy định thi hành "
            "Bộ luật Dân sự về bảo đảm thực hiện nghĩa vụ"
        ),
    )

    tree = parsed_document["tree"]

    parts = parser.find_nodes(tree, "PART")
    chapters = parser.find_nodes(tree, "CHAPTER")
    sections = parser.find_nodes(tree, "SECTION")
    subsections = parser.find_nodes(tree, "SUBSECTION")
    articles = parser.find_nodes(tree, "ARTICLE")
    clauses = parser.find_nodes(tree, "CLAUSE")
    points = parser.find_nodes(tree, "POINT")

    print("=" * 70)
    print("KIỂM TRA LEGAL PARSER")
    print("=" * 70)

    print(f"Số Phần: {len(parts)}")
    print(f"Số Chương: {len(chapters)}")
    print(f"Số Mục: {len(sections)}")
    print(f"Số Tiểu mục: {len(subsections)}")
    print(f"Số Điều: {len(articles)}")
    print(f"Số Khoản: {len(clauses)}")
    print(f"Số Điểm: {len(points)}")

    article_1 = parser.find_article(tree, "1")

    print("\n" + "=" * 70)
    print("KIỂM TRA ĐIỀU 1")
    print("=" * 70)

    if article_1 is None:
        print("Không tìm thấy Điều 1.")
        sys.exit(1)

    print(f"Node ID: {article_1['node_id']}")
    print(f"Tiêu đề: {article_1['title']}")
    print(f"Paragraph index: {article_1['paragraph_index']}")

    clauses_1 = [
        child
        for child in article_1["children"]
        if child["node_type"] == "CLAUSE"
    ]

    print(f"Số khoản: {len(clauses_1)}")

    for clause in clauses_1:
        print(f"\nKhoản {clause['number']}:")

        for paragraph in clause["paragraphs"]:
            print(f"  {paragraph}")

        point_nodes = [
            child
            for child in clause["children"]
            if child["node_type"] == "POINT"
        ]

        for point in point_nodes:
            print(
                f"    Điểm {point['number']}: "
                f"{point['paragraphs'][0]}"
            )

    parser.save_json(
        parsed_document=parsed_document,
        output_path=output_path,
    )

    print("\n" + "=" * 70)
    print(f"Đã lưu JSON tại: {output_path}")
    print("Parser test hoàn tất.")


if __name__ == "__main__":
    main()
