from __future__ import annotations

from typing import Any

from temporal.query_date_resolver import (
    QueryDateResolver,
)
from validity.validity_resolver import (
    ValidityResolver,
)


class TemporalRetrievalRouter:
    """
    Điều phối retrieval theo thời điểm pháp lý.

    Luồng xử lý:
        câu hỏi
        → nhận diện ngày áp dụng
        → chọn index hiện hành hoặc temporal
        → lấy danh sách luật có hiệu lực
        → lọc FAISS theo law_id
        → kiểm tra hiệu lực lần cuối
    """

    def __init__(
        self,
        *,
        current_retriever: Any,
        temporal_retriever: Any,
        validity_resolver: ValidityResolver,
        query_date_resolver: QueryDateResolver,
        current_candidate_k: int,
        temporal_candidate_k: int,
    ) -> None:
        if current_retriever is None:
            raise ValueError(
                "current_retriever không được để trống."
            )

        if temporal_retriever is None:
            raise ValueError(
                "temporal_retriever không được để trống."
            )

        if validity_resolver is None:
            raise ValueError(
                "validity_resolver không được để trống."
            )

        if query_date_resolver is None:
            raise ValueError(
                "query_date_resolver không được để trống."
            )

        if current_candidate_k < 1:
            raise ValueError(
                "current_candidate_k phải lớn hơn "
                "hoặc bằng 1."
            )

        if temporal_candidate_k < 1:
            raise ValueError(
                "temporal_candidate_k phải lớn hơn "
                "hoặc bằng 1."
            )

        self.current_retriever = current_retriever
        self.temporal_retriever = temporal_retriever
        self.validity_resolver = validity_resolver
        self.query_date_resolver = (
            query_date_resolver
        )

        self.current_candidate_k = (
            current_candidate_k
        )

        self.temporal_candidate_k = (
            temporal_candidate_k
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> dict[str, Any]:
        clean_query = str(query).strip()

        if not clean_query:
            raise ValueError(
                "Câu hỏi không được để trống."
            )

        if top_k < 1:
            raise ValueError(
                "top_k phải lớn hơn hoặc bằng 1."
            )

        date_resolution = (
            self.query_date_resolver.resolve(
                clean_query
            )
        )

        effective_law_ids = (
            self.validity_resolver
            .get_effective_law_ids(
                as_of=date_resolution.as_of,
            )
        )

        use_temporal_index = (
            date_resolution.use_temporal_index
        )

        if use_temporal_index:
            selected_retriever = (
                self.temporal_retriever
            )

            candidate_k = (
                self.temporal_candidate_k
            )

            index_name = "legal_temporal"

        else:
            selected_retriever = (
                self.current_retriever
            )

            candidate_k = (
                self.current_candidate_k
            )

            index_name = "legal_dense"

        if not effective_law_ids:
            raw_results: list[
                dict[str, Any]
            ] = []

        else:
            raw_results = (
                selected_retriever.retrieve(
                    query=clean_query,
                    top_k=top_k,
                    filters={
                        "law_id": (
                            effective_law_ids
                        )
                    },
                    candidate_k=candidate_k,
                )
            )

        validity_report = (
            self.validity_resolver
            .resolve_results(
                raw_results,
                as_of=date_resolution.as_of,
            )
        )

        return {
            "query": clean_query,
            "as_of": (
                date_resolution.as_of
            ),
            "date_resolution": (
                date_resolution
            ),
            "index_name": index_name,
            "effective_law_ids": (
                effective_law_ids
            ),
            "raw_results": raw_results,
            "validity_report": (
                validity_report
            ),
            "results": validity_report[
                "valid_results"
            ],
            "excluded_results": (
                validity_report[
                    "excluded_results"
                ]
            ),
        }
