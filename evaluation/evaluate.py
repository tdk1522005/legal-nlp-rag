from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# =========================================================
# 1. ĐƯỜNG DẪN PROJECT
# =========================================================

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 2. IMPORT PIPELINE CỦA PROJECT
# =========================================================

from chat import (
    CURRENT_INDEX_DIR,
    TEMPORAL_INDEX_DIR,
    CURRENT_CANDIDATE_K,
    TEMPORAL_CANDIDATE_K,
    load_manifest,
    load_vector_store,
    validate_manifest_compatibility,
)

from models.qwen_embedding import QwenEmbedding
from retrieval.retriever import Retriever
from retrieval.temporal_router import TemporalRetrievalRouter
from temporal.query_date_resolver import QueryDateResolver
from validity.validity_resolver import ValidityResolver


# =========================================================
# 3. CẤU HÌNH
# =========================================================

TEST_FILE = EVALUATION_DIR / "test_questions.json"

# Chatbot hiện lấy tối đa 8 kết quả để kiểm tra
MAX_RESULTS = 8

# Ranking metrics
METRIC_KS = (1, 3, 5, 8)


# =========================================================
# 4. ĐỌC CÂU HỎI KIỂM THỬ
# =========================================================

def load_test_questions() -> list[dict[str, Any]]:
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {TEST_FILE}"
        )

    with TEST_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "test_questions.json phải chứa một danh sách."
        )

    if not data:
        raise ValueError(
            "Bộ câu hỏi kiểm thử đang rỗng."
        )

    return data


# =========================================================
# 5. TẢI Qwen3-Embedding-0.6B + FAISS
# =========================================================

def load_retrieval_pipeline() -> TemporalRetrievalRouter:
    print()
    print("Đang khởi động hệ thống...")

    print("- Đọc current index...")
    current_manifest = load_manifest(
        CURRENT_INDEX_DIR
    )

    print("- Đọc temporal index...")
    temporal_manifest = load_manifest(
        TEMPORAL_INDEX_DIR
    )

    validate_manifest_compatibility(
        current_manifest,
        temporal_manifest,
    )

    print(
        "- Tải mô hình Qwen3-Embedding-0.6B: "
        f"{current_manifest['model_name']}..."
    )

    embedding_model = QwenEmbedding(
        model_name=str(
            current_manifest["model_name"]
        ),
        device="cpu",
        dimension=int(
            current_manifest["dimension"]
        ),
        batch_size=1,
    )

    print("- Tải current FAISS index...")

    current_store = load_vector_store(
        CURRENT_INDEX_DIR,
        current_manifest,
    )

    print("- Tải temporal FAISS index...")

    temporal_store = load_vector_store(
        TEMPORAL_INDEX_DIR,
        temporal_manifest,
    )

    current_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=current_store,
    )

    temporal_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=temporal_store,
    )

    router = TemporalRetrievalRouter(
        current_retriever=current_retriever,
        temporal_retriever=temporal_retriever,
        validity_resolver=ValidityResolver(),
        query_date_resolver=QueryDateResolver(),
        current_candidate_k=CURRENT_CANDIDATE_K,
        temporal_candidate_k=TEMPORAL_CANDIDATE_K,
    )

    print()
    print("Hệ thống đã sẵn sàng.")

    return router


# =========================================================
# 6. KIỂM TRA ĐÚNG LUẬT + ĐÚNG ĐIỀU
# =========================================================

def is_correct_result(
    result: dict[str, Any],
    expected_law_id: str,
    expected_article: str,
) -> bool:

    metadata = result.get(
        "metadata",
        {},
    )

    law_id = str(
        metadata.get("law_id", "")
    ).strip()

    article_number = str(
        metadata.get("article_number", "")
    ).strip()

    return (
        law_id == expected_law_id
        and article_number == expected_article
    )


# =========================================================
# 7. TÌM VỊ TRÍ ĐIỀU LUẬT ĐÚNG
# =========================================================

def find_correct_position(
    results: list[dict[str, Any]],
    expected_law_id: str,
    expected_article: str,
) -> int | None:

    for position, result in enumerate(
        results,
        start=1,
    ):
        if is_correct_result(
            result=result,
            expected_law_id=expected_law_id,
            expected_article=expected_article,
        ):
            return position

    return None


# =========================================================
# 8. HIỂN THỊ KẾT QUẢ ĐẦU TIÊN
# =========================================================

def get_result_label(
    result: dict[str, Any],
) -> str:

    metadata = result.get(
        "metadata",
        {},
    )

    law_id = str(
        metadata.get("law_id", "?")
    )

    article = str(
        metadata.get("article_number", "?")
    )

    score = result.get("score")

    if score is None:
        return (
            f"{law_id} - Điều {article}"
        )

    return (
        f"{law_id} - Điều {article} "
        f"(độ tương đồng: {float(score):.4f})"
    )


# =========================================================
# 9. CHẠY ĐÁNH GIÁ
# =========================================================

def main() -> None:
    questions = load_test_questions()

    print()
    print("=" * 72)
    print("ĐÁNH GIÁ KHẢ NĂNG TÌM ĐÚNG ĐIỀU LUẬT CỦA CHATBOT")
    print("=" * 72)

    print(
        f"Số câu hỏi kiểm thử: {len(questions)}"
    )

    router = load_retrieval_pipeline()

    # -----------------------------------------------------
    # Các biến thống kê
    # -----------------------------------------------------

    correct_first = 0

    found_but_not_first = 0

    not_found = 0

    error_count = 0

    hit_counts = {
        k: 0
        for k in METRIC_KS
    }

    reciprocal_rank_sum = 0.0

    # -----------------------------------------------------
    # Chạy từng câu
    # -----------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question = str(
            item["question"]
        ).strip()

        expected_law_id = str(
            item["expected_law_id"]
        ).strip()

        expected_article = str(
            item["expected_article"]
        ).strip()

        print()
        print("-" * 72)

        print(
            f"CÂU {index}/{len(questions)}"
        )

        print(
            f"Câu hỏi: {question}"
        )

        print(
            "Đáp án mong đợi: "
            f"{expected_law_id} - "
            f"Điều {expected_article}"
        )

        try:
            route_output = router.retrieve(
                query=question,
                top_k=MAX_RESULTS,
            )

            results = route_output[
                "results"
            ]

            position = find_correct_position(
                results=results,
                expected_law_id=expected_law_id,
                expected_article=expected_article,
            )

            if position is not None:
                reciprocal_rank_sum += (
                    1.0 / position
                )

                for k in METRIC_KS:
                    if position <= k:
                        hit_counts[k] += 1

            print(
                f"Chỉ mục: "
                f"{route_output['index_name']}"
            )

            print(
                f"Kiểu truy xuất: "
                f"{route_output['retrieval_mode']}"
            )

            if results:
                print(
                    "Kết quả được xếp đầu tiên: "
                    f"{get_result_label(results[0])}"
                )

            else:
                print(
                    "Hệ thống không trả về kết quả."
                )

            # -------------------------------------------------
            # Trường hợp 1: đúng ngay đầu tiên
            # -------------------------------------------------

            if position == 1:
                correct_first += 1

                print(
                    "Đánh giá: TỐT"
                )

                print(
                    "=> Hệ thống tìm đúng Điều luật "
                    "ngay ở kết quả đầu tiên."
                )

            # -------------------------------------------------
            # Trường hợp 2: tìm được nhưng không đứng đầu
            # -------------------------------------------------

            elif position is not None:
                found_but_not_first += 1

                print(
                    "Đánh giá: TÌM ĐƯỢC"
                )

                print(
                    "=> Điều luật đúng được tìm thấy "
                    f"ở vị trí thứ {position}."
                )

            # -------------------------------------------------
            # Trường hợp 3: không tìm thấy
            # -------------------------------------------------

            else:
                not_found += 1

                print(
                    "Đánh giá: KHÔNG TÌM ĐƯỢC"
                )

                print(
                    f"=> Điều luật đúng không xuất hiện "
                    f"trong {MAX_RESULTS} kết quả được lấy ra."
                )

        except Exception as error:
            error_count += 1

            print(
                "Đánh giá: LỖI"
            )

            print(
                f"=> {type(error).__name__}: {error}"
            )

    # =====================================================
    # 10. TỔNG HỢP
    # =====================================================

    total = len(questions)

    found_total = (
        correct_first
        + found_but_not_first
    )

    correct_first_rate = (
        correct_first
        / total
        * 100
    )

    found_rate = (
        found_total
        / total
        * 100
    )
    accuracy = found_rate

    hit_at = {
        k: hit_counts[k] / total
        for k in METRIC_KS
    }

    # The current dataset contains exactly one
    # relevant law/article target per question.
    # Therefore Recall@K equals Hit@K.
    recall_at = {
        k: hit_at[k]
        for k in METRIC_KS
    }

    mrr_at_8 = (
        reciprocal_rank_sum / total
    )

    not_found_rate = (
        not_found
        / total
        * 100
    )

    print()
    print()
    print("=" * 72)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 72)

    print()

    print(
        f"Tổng số câu hỏi               : "
        f"{total} câu"
    )

    print()

    print(
        "Tìm đúng ngay kết quả đầu tiên : "
        f"{correct_first}/{total} câu "
        f"({correct_first_rate:.2f}%)"
    )

    print(
        "Tìm được nhưng không đứng đầu  : "
        f"{found_but_not_first}/{total} câu"
    )

    print(
        "Không tìm được Điều luật đúng  : "
        f"{not_found}/{total} câu "
        f"({not_found_rate:.2f}%)"
    )

    if error_count > 0:
        print(
            "Câu xảy ra lỗi                 : "
            f"{error_count}/{total} câu"
        )

    print()
    print("-" * 72)
    print("CHỈ SỐ ĐÁNH GIÁ")
    print("-" * 72)
    print()

    print(
        "Accuracy                      : "
        f"{accuracy:.2f}%"
    )

    print(
        "Số câu tìm đúng               : "
        f"{found_total}/{total} câu"
    )

    print(
        "Đúng ngay kết quả đầu tiên    : "
        f"{correct_first}/{total} câu "
        f"({correct_first_rate:.2f}%)"
    )

    print(
        "Đúng nhưng không đứng đầu     : "
        f"{found_but_not_first}/{total} câu"
    )

    print(
        "Không tìm được                : "
        f"{not_found}/{total} câu "
        f"({not_found_rate:.2f}%)"
    )

    print()

    print(
        f"{found_total}/{total} câu "
        f"= {found_rate:.2f}%"
    )

    print()
    print("-" * 72)
    print("KẾT LUẬN")
    print("-" * 72)

    print()

    print(
        f"Hệ thống tìm được Điều luật đúng "
        f"cho {found_total}/{total} câu hỏi, "
        f"đạt tỷ lệ {found_rate:.2f}%."
    )

    print(
        f"Trong đó có {correct_first}/{total} câu "
        f"({correct_first_rate:.2f}%) "
        "mà Điều luật đúng được xếp "
        "ngay ở vị trí đầu tiên."
    )

    print()
    print()
    print("-" * 72)
    print("RANKING METRICS")
    print("-" * 72)
    print()

    for k in METRIC_KS:
        print(
            f"Hit@{k:<2}    : "
            f"{hit_at[k] * 100:.2f}% "
            f"({hit_counts[k]}/{total})"
        )

    print()

    for k in METRIC_KS:
        print(
            f"Recall@{k:<2} : "
            f"{recall_at[k] * 100:.2f}%"
        )

    print()

    print(
        f"MRR@{MAX_RESULTS:<2}    : "
        f"{mrr_at_8:.4f}"
    )

    print()

    print(
        "Note: one relevant law/article target "
        "per question => Recall@K = Hit@K."
    )

    print(
        f"Accuracy = Hit@{MAX_RESULTS} "
        "for the current evaluation setup."
    )

    print()

    print("=" * 72)


if __name__ == "__main__":
    main()