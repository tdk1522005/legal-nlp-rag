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
        / "civil"
        / "91_2015_QH13_civil_code_2015.docx"
    )

    output_path = (
        DATA_DIR
        / "parsed"
        / "civil_code_2015.json"
    )

    parser = LegalDocxParser()

    parsed_document = parser.parse(
        file_path=source_path,
        law_id="civil_code_2015",
        document_title="Bộ luật Dân sự 2015",
    )

    tree = parsed_document["tree"]

    parts = parser.find_nodes(
        tree,
        "PART",
    )

    chapters = parser.find_nodes(
        tree,
        "CHAPTER",
    )

    sections = parser.find_nodes(
        tree,
        "SECTION",
    )

    subsections = parser.find_nodes(
        tree,
        "SUBSECTION",
    )

    articles = parser.find_nodes(
        tree,
        "ARTICLE",
    )

    clauses = parser.find_nodes(
        tree,
        "CLAUSE",
    )

    points = parser.find_nodes(
        tree,
        "POINT",
    )

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

    article_117 = parser.find_article(
        tree,
        "117",
    )

    print("\n" + "=" * 70)
    print("KIỂM TRA ĐIỀU 117")
    print("=" * 70)

    if article_117 is None:
        print("Không tìm thấy Điều 117.")
        sys.exit(1)

    print(
        f"Node ID: {article_117['node_id']}"
    )

    print(
        f"Tiêu đề: {article_117['title']}"
    )

    print(
        f"Paragraph index: "
        f"{article_117['paragraph_index']}"
    )

    clauses_117 = [
        child
        for child in article_117["children"]
        if child["node_type"] == "CLAUSE"
    ]

    print(
        f"Số khoản: {len(clauses_117)}"
    )

    for clause in clauses_117:
        print(
            f"\nKhoản {clause['number']}:"
        )

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

    if len(articles) != 689:
        print(
            "[WARNING] Số Điều không bằng 689. "
            "Cần kiểm tra lại parser."
        )
    else:
        print(
            "Đã nhận diện đủ 689 Điều."
        )


if __name__ == "__main__":
    main()