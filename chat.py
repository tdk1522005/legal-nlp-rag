from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from context.context_builder import ContextBuilder
from llm.qwen_llm import QwenLLM
from models.qwen_embedding import QwenEmbedding
from prompt.prompt_builder import PromptBuilder
from retrieval.retriever import Retriever
from retrieval.temporal_router import TemporalRetrievalRouter
from temporal.query_date_resolver import QueryDateResolver
from validity.validity_resolver import ValidityResolver
from vectorstore.faiss_store import FaissStore


PROJECT_ROOT = Path(__file__).resolve().parent

CURRENT_INDEX_DIR = (
    PROJECT_ROOT
    / "index"
    / "legal_dense_qwen"
)

TEMPORAL_INDEX_DIR = (
    PROJECT_ROOT
    / "index"
    / "legal_temporal_qwen"
)

TOP_K = 8
CONTEXT_MAX_CHARS = 14000

CURRENT_CANDIDATE_K = 100
TEMPORAL_CANDIDATE_K = 1000


def load_manifest(
    index_dir: Path,
) -> dict[str, Any]:
    path = (
        index_dir
        / "index_manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        manifest = json.load(file)

    required_fields = {
        "model_name",
        "dimension",
        "embedding_provider",
        "vector_count",
    }

    missing_fields = (
        required_fields.difference(
            manifest
        )
    )

    if missing_fields:
        raise ValueError(
            "Manifest thiếu các trường: "
            + ", ".join(
                sorted(missing_fields)
            )
        )

    return manifest


def validate_manifest_compatibility(
    current_manifest: dict[str, Any],
    temporal_manifest: dict[str, Any],
) -> None:
    fields = (
        "model_name",
        "dimension",
        "embedding_provider",
    )

    for field in fields:
        current_value = (
            current_manifest.get(field)
        )

        temporal_value = (
            temporal_manifest.get(field)
        )

        if current_value != temporal_value:
            raise RuntimeError(
                "Hai index không tương thích tại "
                f"trường {field}: "
                f"{current_value} != "
                f"{temporal_value}"
            )


def load_vector_store(
    index_dir: Path,
    manifest: dict[str, Any],
) -> FaissStore:
    files = dict(
        manifest.get("files", {})
        or {}
    )

    index_path = (
        index_dir
        / str(
            files.get(
                "index",
                "legal_dense.index",
            )
        )
    )

    metadata_path = (
        index_dir
        / str(
            files.get(
                "metadata",
                "chunk_metadata.jsonl",
            )
        )
    )

    vector_store = FaissStore(
        dimension=int(
            manifest["dimension"]
        )
    )

    vector_store.load(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    loaded_count = len(
        vector_store.documents
    )

    expected_count = int(
        manifest["vector_count"]
    )

    if loaded_count != expected_count:
        raise RuntimeError(
            "Số metadata đã tải không khớp "
            f"manifest: {loaded_count} != "
            f"{expected_count}"
        )

    return vector_store


def print_sources(
    results: list[dict[str, Any]],
) -> None:
    citations: list[str] = []
    seen: set[str] = set()

    for result in results:
        metadata = result.get(
            "metadata",
            {},
        )

        citation = str(
            metadata.get(
                "citation",
                "",
            )
        ).strip()

        if not citation:
            continue

        if citation in seen:
            continue

        seen.add(citation)
        citations.append(citation)

    if not citations:
        return

    print("\nNguồn truy xuất:")

    for citation in citations:
        print(f"- {citation}")


def print_temporal_report(
    route_output: dict[str, Any],
) -> None:
    resolution = route_output[
        "date_resolution"
    ]

    print("\nThời điểm pháp lý:")
    print(
        "- Ngày áp dụng: "
        f"{route_output['as_of'].isoformat()}"
    )
    print(
        "- Chỉ mục được sử dụng: "
        f"{route_output['index_name']}"
    )

    if resolution.matched_text:
        print(
            "- Cụm thời gian nhận diện: "
            f"{resolution.matched_text}"
        )

    if resolution.warning:
        print(
            f"- Lưu ý: {resolution.warning}"
        )


def print_validity_report(
    report: dict[str, Any],
) -> None:
    evaluations = report.get(
        "law_evaluations",
        {},
    )

    displayed = False

    for evaluation in evaluations.values():
        validity_state = evaluation.get(
            "validity_state"
        )

        amending_laws = evaluation.get(
            "amending_laws",
            [],
        )

        replacements = evaluation.get(
            "replacements",
            [],
        )

        has_legal_change = (
            validity_state != "effective"
            or bool(amending_laws)
            or bool(replacements)
        )

        if not has_legal_change:
            continue

        if not displayed:
            print("\nTình trạng hiệu lực:")
            displayed = True

        title = (
            evaluation.get("title")
            or evaluation.get("law_id")
        )

        law_number = evaluation.get(
            "law_number"
        )

        label = str(title)

        if law_number:
            label += f" ({law_number})"

        print(
            f"- {label}: "
            f"{validity_state} "
            f"tại ngày {evaluation['as_of']}"
        )

        for amendment in amending_laws:
            amendment_title = (
                amendment.get("title")
                or amendment.get("law_id")
            )

            amendment_number = (
                amendment.get("law_number")
            )

            text = (
                "  + Được sửa đổi bởi "
                f"{amendment_title}"
            )

            if amendment_number:
                text += (
                    f" ({amendment_number})"
                )

            print(text)

        for replacement in replacements:
            if not replacement.get(
                "is_effective"
            ):
                continue

            replacement_title = (
                replacement.get("title")
                or replacement.get("law_id")
            )

            print(
                "  + Văn bản thay thế: "
                f"{replacement_title}"
            )

        for warning in evaluation.get(
            "warnings",
            [],
        ):
            print(f"  ! {warning}")

    excluded_count = len(
        report.get(
            "excluded_results",
            [],
        )
    )

    if excluded_count:
        if not displayed:
            print("\nTình trạng hiệu lực:")

        print(
            "- Đã loại "
            f"{excluded_count} kết quả "
            "không còn hiệu lực."
        )


SEMANTIC_MIN_SCORE = 0.65
ARTICLE_HEADING_MIN_SCORE = 0.90

RELEVANCE_FALLBACK_MESSAGE = (
    "Tôi chưa tìm thấy đủ căn cứ "
    "trong kho văn bản pháp luật "
    "hiện có để trả lời chính xác "
    "câu hỏi này."
)


def is_semantic_result_relevant(
    route_output: dict[str, Any],
) -> bool:
    """
    Decide whether retrieval has enough evidence
    to continue to context building and the LLM.

    Accepted evidence:
    1. Exact legal reference.
    2. Strong Qwen semantic similarity.
    3. Strong article-heading match for
       no-accent Vietnamese queries.
    """
    retrieval_mode = str(
        route_output.get(
            "retrieval_mode",
            "semantic",
        )
    )

    # Exact document/article lookup does not use
    # cosine similarity.
    if retrieval_mode == "exact_reference":
        return True

    law_scope = route_output.get(
        "law_scope"
    )

    # Explicitly detected unsupported domain.
    if (
        law_scope is not None
        and bool(
            getattr(
                law_scope,
                "is_out_of_scope",
                False,
            )
        )
    ):
        return False

    results = route_output.get(
        "results",
        [],
    )

    if not results:
        return False

    top_result = results[0]

    # -------------------------------------------------
    # Evidence type 1:
    # Article-heading retrieval.
    #
    # Do not treat heading_score as cosine similarity.
    # It has its own independent threshold.
    # -------------------------------------------------
    retrieval_source = str(
        top_result.get(
            "retrieval_source",
            "",
        )
    )

    heading_score = top_result.get(
        "heading_score"
    )

    if (
        retrieval_source
        == "article_heading"
        and heading_score is not None
        and float(heading_score)
        >= ARTICLE_HEADING_MIN_SCORE
    ):
        return True

    # -------------------------------------------------
    # Evidence type 2:
    # Qwen semantic similarity.
    # -------------------------------------------------
    top_score = top_result.get(
        "score"
    )

    if top_score is None:
        return False

    return (
        float(top_score)
        >= SEMANTIC_MIN_SCORE
    )


def select_context_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reduce retrieval noise before sending context
    to a small local LLM.

    If the first two ranked results belong to the
    same law and article, treat that article as the
    primary answer source and keep all retrieved
    chunks from that article.

    Otherwise keep the original retrieval results.
    """
    if len(results) < 2:
        return results

    first_metadata = results[0].get(
        "metadata",
        {},
    )

    second_metadata = results[1].get(
        "metadata",
        {},
    )

    first_key = (
        str(first_metadata.get("law_id", "")),
        str(
            first_metadata.get(
                "article_number",
                "",
            )
        ),
    )

    second_key = (
        str(second_metadata.get("law_id", "")),
        str(
            second_metadata.get(
                "article_number",
                "",
            )
        ),
    )

    if (
        not first_key[0]
        or not first_key[1]
        or first_key != second_key
    ):
        return results

    focused_results = [
        result
        for result in results
        if (
            str(
                result.get(
                    "metadata",
                    {},
                ).get(
                    "law_id",
                    "",
                )
            ),
            str(
                result.get(
                    "metadata",
                    {},
                ).get(
                    "article_number",
                    "",
                )
            ),
        )
        == first_key
    ]

    return (
        focused_results
        if focused_results
        else results
    )


def main() -> None:
    print("=" * 70)
    print("KHỞI ĐỘNG LEGAL RAG CHATBOT")
    print("=" * 70)

    print(
        "Đang đọc current index manifest..."
    )

    current_manifest = load_manifest(
        CURRENT_INDEX_DIR
    )

    print(
        "Đang đọc temporal index manifest..."
    )

    temporal_manifest = load_manifest(
        TEMPORAL_INDEX_DIR
    )

    validate_manifest_compatibility(
        current_manifest,
        temporal_manifest,
    )

    print(
        "Đang tải Qwen3-Embedding-0.6B "
        f"({current_manifest['model_name']})..."
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

    print(
        "Đang tải current FAISS index..."
    )

    current_store = load_vector_store(
        CURRENT_INDEX_DIR,
        current_manifest,
    )

    print(
        "Đang tải temporal FAISS index..."
    )

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

    validity_resolver = ValidityResolver()
    query_date_resolver = QueryDateResolver()

    temporal_router = (
        TemporalRetrievalRouter(
            current_retriever=(
                current_retriever
            ),
            temporal_retriever=(
                temporal_retriever
            ),
            validity_resolver=(
                validity_resolver
            ),
            query_date_resolver=(
                query_date_resolver
            ),
            current_candidate_k=(
                CURRENT_CANDIDATE_K
            ),
            temporal_candidate_k=(
                TEMPORAL_CANDIDATE_K
            ),
        )
    )

    context_builder = ContextBuilder(
        max_chars=CONTEXT_MAX_CHARS,
        include_score=False,
    )

    prompt_builder = PromptBuilder()

    llm = QwenLLM(
        base_url="http://127.0.0.1:8080",
        model_name="qwen",
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        max_output_tokens=1024,
        timeout=180,
        disable_thinking=True,
    )

    current_count = len(
        current_store.documents
    )

    temporal_count = len(
        temporal_store.documents
    )

    print("-" * 70)
    print(
        "Đã tải current index: "
        f"{current_count} legal chunk."
    )
    print(
        "Đã tải temporal index: "
        f"{temporal_count} legal chunk."
    )
    print("Chatbot đã sẵn sàng.")
    print(
        "Nhập 'exit' hoặc 'quit' để thoát."
    )
    print("-" * 70)

    while True:
        try:
            question = input(
                "\nBạn: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print("\nĐã dừng chatbot.")
            break

        if question.casefold() in {
            "exit",
            "quit",
        }:
            print("Đã dừng chatbot.")
            break

        if not question:
            print(
                "Vui lòng nhập câu hỏi."
            )
            continue

        try:
            route_output = (
                temporal_router.retrieve(
                    query=question,
                    top_k=TOP_K,
                )
            )

            results = route_output[
                "results"
            ]

            validity_report = (
                route_output[
                    "validity_report"
                ]
            )

            if not results:
                print_temporal_report(
                    route_output
                )

                print(
                    "\nTrợ lý: "
                    "Không tìm thấy tài liệu "
                    "có hiệu lực phù hợp với "
                    "thời điểm được hỏi."
                )
                continue

            if not is_semantic_result_relevant(
                route_output
            ):
                print_temporal_report(
                    route_output
                )

                print()
                print(
                    "Tr\u1ee3 l\u00fd:",
                    RELEVANCE_FALLBACK_MESSAGE,
                )
                continue

            context_results = (
                select_context_results(
                    results
                )
            )

            context = context_builder.build(
                context_results
            )

            if not context:
                print(
                    "\nTrợ lý: "
                    "Không tạo được context "
                    "từ kết quả truy xuất."
                )
                continue

            temporal_question = (
                f"{question}\n\n"
                "Thời điểm pháp lý cần áp dụng: "
                f"{route_output['as_of'].isoformat()}."
            )

            prompt = prompt_builder.build(
                question=temporal_question,
                context=context,
            )

            answer = llm.generate(
                prompt
            )

            print(
                "\n"
                + "=" * 70
            )
            print("TRỢ LÝ")
            print("=" * 70)
            print(answer)

            print_temporal_report(
                route_output
            )

            print_validity_report(
                validity_report
            )

            print_sources(
                context_results
            )

        except Exception as error:
            print(
                "\nĐã xảy ra lỗi khi xử lý "
                f"câu hỏi: {error}"
            )


if __name__ == "__main__":
    main()
