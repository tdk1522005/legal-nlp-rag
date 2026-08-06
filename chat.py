from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from context.context_builder import ContextBuilder
from llm.gemini_llm import GeminiLLM
from models.bge_embedding import BGEEmbedding
from prompt.prompt_builder import PromptBuilder
from retrieval.retriever import Retriever
from vectorstore.faiss_store import FaissStore


PROJECT_ROOT = Path(__file__).resolve().parent

INDEX_DIR = (
    PROJECT_ROOT
    / "index"
    / "legal_dense"
)

INDEX_PATH = (
    INDEX_DIR
    / "legal_dense.index"
)

METADATA_PATH = (
    INDEX_DIR
    / "chunk_metadata.jsonl"
)

MANIFEST_PATH = (
    INDEX_DIR
    / "index_manifest.json"
)

TOP_K = 8
CONTEXT_MAX_CHARS = 14000


def load_manifest(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    required_fields = {
        "model_name",
        "dimension",
        "max_length",
        "vector_count",
    }

    missing_fields = required_fields.difference(
        manifest
    )

    if missing_fields:
        raise ValueError(
            "Manifest thiếu các trường: "
            + ", ".join(
                sorted(missing_fields)
            )
        )

    return manifest


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


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "Không tìm thấy GEMINI_API_KEY "
            "trong file .env."
        )

    print("=" * 70)
    print("KHỞI ĐỘNG LEGAL RAG CHATBOT")
    print("=" * 70)

    print("Đang đọc index manifest...")

    manifest = load_manifest(
        MANIFEST_PATH
    )

    print(
        "Đang tải BGE-M3 "
        f"({manifest['model_name']})..."
    )

    embedding_model = BGEEmbedding(
        model_name=str(
            manifest["model_name"]
        ),
        use_fp16=False,
        batch_size=1,
        max_length=int(
            manifest["max_length"]
        ),
    )

    print("Đang tải FAISS index...")

    vector_store = FaissStore(
        dimension=int(
            manifest["dimension"]
        )
    )

    vector_store.load(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    loaded_count = len(
        vector_store.documents
    )

    expected_count = int(
        manifest["vector_count"]
    )

    if loaded_count != expected_count:
        raise RuntimeError(
            "Số metadata đã tải không khớp manifest: "
            f"{loaded_count} != {expected_count}"
        )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder(
        max_chars=CONTEXT_MAX_CHARS,
        include_score=False,
    )

    prompt_builder = PromptBuilder()

    llm = GeminiLLM(
        api_key=api_key
    )

    print("-" * 70)
    print(
        f"Đã tải {loaded_count} legal chunk."
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
            results = retriever.retrieve(
                query=question,
                top_k=TOP_K,
            )

            if not results:
                print(
                    "\nTrợ lý: "
                    "Không tìm thấy tài liệu phù hợp."
                )
                continue

            context = context_builder.build(
                results
            )

            if not context:
                print(
                    "\nTrợ lý: "
                    "Không tạo được context "
                    "từ kết quả truy xuất."
                )
                continue

            prompt = prompt_builder.build(
                question=question,
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

            print_sources(results)

        except Exception as error:
            print(
                "\nĐã xảy ra lỗi khi xử lý "
                f"câu hỏi: {error}"
            )


if __name__ == "__main__":
    main()