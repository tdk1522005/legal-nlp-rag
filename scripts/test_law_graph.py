import sys
from pathlib import Path


# Đưa thư mục rag_model vào Python path.
RAG_MODEL_DIR = Path(__file__).resolve().parents[1]

if str(RAG_MODEL_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(RAG_MODEL_DIR),
    )


from graph.law_graph import LawGraph


def print_law_list(
    title: str,
    laws: list[dict],
    law_key: str,
) -> None:
    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)

    if not laws:
        print("Không tìm thấy.")
        return

    for index, item in enumerate(
        laws,
        start=1,
    ):
        law = item[law_key]

        print(
            f"{index}. {law['title']} "
            f"({law['law_number']})"
        )

        print(
            f"   Quan hệ: {item['relation']}"
        )


def main() -> None:
    law_graph = LawGraph()
    law_graph.load()

    # =====================================================
    # 1. THỐNG KÊ
    # =====================================================

    summary = law_graph.summary()

    print("=" * 65)
    print("LAW GRAPH TEST")
    print("=" * 65)

    print(
        f"Số node: {summary['node_count']}"
    )

    print(
        f"Số edge: {summary['edge_count']}"
    )

    print(
        f"Các loại quan hệ: "
        f"{summary['relation_counts']}"
    )

    # =====================================================
    # 2. TÌM LUẬT THEO SỐ HIỆU
    # =====================================================

    law = law_graph.get_law(
        "91/2015/QH13"
    )

    print("\n" + "-" * 65)
    print("Tìm theo số hiệu 91/2015/QH13")
    print("-" * 65)

    if law:
        print(f"law_id: {law['law_id']}")
        print(f"Tên: {law['title']}")
        print(f"Trạng thái: {law['status']}")

    # =====================================================
    # 3. LUẬT THAY THẾ BLDS 2005
    # =====================================================

    replacements = (
        law_graph.get_replacement_for(
            "civil_code_2005"
        )
    )

    print_law_list(
        title=(
            "Văn bản thay thế "
            "Bộ luật Dân sự 2005"
        ),
        laws=replacements,
        law_key="source_law",
    )

    # =====================================================
    # 4. CÁC LUẬT LIÊN QUAN BLDS 2015
    # =====================================================

    related_laws = (
        law_graph.get_related_laws(
            "civil_code_2015"
        )
    )

    print("\n" + "-" * 65)
    print(
        "Các luật liên quan "
        "Bộ luật Dân sự 2015"
    )
    print("-" * 65)

    for index, item in enumerate(
        related_laws,
        start=1,
    ):
        related_law = item["law"]

        print(
            f"{index}. {related_law['title']} "
            f"({related_law['law_number']})"
        )

        topics = item["edge"].get(
            "topics",
            [],
        )

        if topics:
            print(
                "   Chủ đề: "
                + ", ".join(topics)
            )

    # =====================================================
    # 5. LUẬT 43/2024/QH15 SỬA LUẬT NÀO?
    # =====================================================

    amended_laws = (
        law_graph.get_amended_laws(
            "43/2024/QH15"
        )
    )

    print_law_list(
        title=(
            "Các luật bị Luật "
            "43/2024/QH15 sửa đổi"
        ),
        laws=amended_laws,
        law_key="target_law",
    )

    # =====================================================
    # 6. LUẬT NÀO SỬA BLTTDS 2015?
    # =====================================================

    amending_laws = (
        law_graph.get_amending_laws(
            "civil_procedure_code_2015"
        )
    )

    print_law_list(
        title=(
            "Các luật sửa đổi "
            "Bộ luật Tố tụng dân sự 2015"
        ),
        laws=amending_laws,
        law_key="source_law",
    )


if __name__ == "__main__":
    main()