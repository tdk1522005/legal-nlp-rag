from __future__ import annotations

from typing import Any


class Retriever:
    """
    Dense retriever sử dụng BGE-M3 và FAISS.

    Luồng xử lý:
    query
        → encode_query()
        → FAISS search
        → trả về các legal chunk gần nhất
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
    ) -> None:
        if embedding_model is None:
            raise ValueError(
                "embedding_model không được để trống."
            )

        if vector_store is None:
            raise ValueError(
                "vector_store không được để trống."
            )

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        filters: dict[str, Any] | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tìm các legal chunk liên quan nhất.

        filters ví dụ:
            {"law_id": "civil_code_2015"}

            {"law_id": [
                "civil_code_2015",
                "land_law_2024",
            ]}

        candidate_k:
            Số ứng viên FAISS lấy ra trước khi lọc
            metadata.
        """
        clean_query = str(query).strip()

        if not clean_query:
            raise ValueError(
                "Câu hỏi không được để trống."
            )

        if top_k < 1:
            raise ValueError(
                "top_k phải lớn hơn hoặc bằng 1."
            )

        if candidate_k is not None:
            if candidate_k < top_k:
                raise ValueError(
                    "candidate_k phải lớn hơn "
                    "hoặc bằng top_k."
                )

        query_embedding = (
            self.embedding_model.encode_query(
                clean_query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            candidate_k=candidate_k,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rank"] = rank

        return results