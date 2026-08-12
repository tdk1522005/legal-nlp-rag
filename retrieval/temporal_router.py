from __future__ import annotations

from typing import Any

from retrieval.exact_reference import (
    resolve_exact_legal_reference,
)
from retrieval.law_scope import (
    detect_law_scope,
    filter_effective_law_ids_by_scope,
)
from retrieval.article_heading_retriever import (
    rank_article_headings,
)
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

        law_scope = detect_law_scope(
            clean_query
        )

        retrieval_law_ids = (
            filter_effective_law_ids_by_scope(
                law_scope,
                effective_law_ids,
            )
        )

        article_heading_match = None

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

        exact_reference = (
            resolve_exact_legal_reference(
                clean_query,
                allowed_law_ids=effective_law_ids,
            )
        )

        exact_results: list[
            dict[str, Any]
        ] = []

        if exact_reference is not None:
            # Exact reference is an explicit document lookup.
            # If the requested law is historical and is not
            # present in the current index, use temporal index.

            current_documents = getattr(
                self.current_retriever.vector_store,
                "documents",
                [],
            )

            temporal_documents = getattr(
                self.temporal_retriever.vector_store,
                "documents",
                [],
            )

            current_has_law = any(
                str(document.get("law_id", ""))
                == exact_reference.law_id
                for document in current_documents
            )

            temporal_has_law = any(
                str(document.get("law_id", ""))
                == exact_reference.law_id
                for document in temporal_documents
            )

            if current_has_law:
                selected_retriever = (
                    self.current_retriever
                )
                index_name = "legal_dense"

            elif temporal_has_law:
                selected_retriever = (
                    self.temporal_retriever
                )
                index_name = "legal_temporal"

            documents = getattr(
                selected_retriever.vector_store,
                "documents",
                [],
            )

            for index_position, document in enumerate(
                documents
            ):
                law_id = str(
                    document.get(
                        "law_id",
                        "",
                    )
                )

                article_number = str(
                    document.get(
                        "article_number",
                        "",
                    )
                )

                if (
                    law_id
                    != exact_reference.law_id
                ):
                    continue

                if (
                    article_number
                    != exact_reference.article_number
                ):
                    continue

                document_text = str(
                    document.get(
                        "text",
                        "",
                    )
                ).strip()

                if not document_text:
                    continue

                metadata = {
                    key: value
                    for key, value
                    in document.items()
                    if key != "text"
                }

                exact_results.append(
                    {
                        "text": document_text,
                        "metadata": metadata,
                        "score": None,
                        "faiss_id": (
                            index_position
                        ),
                        "rank": (
                            len(exact_results)
                            + 1
                        ),
                    }
                )

                if (
                    len(exact_results)
                    >= top_k
                ):
                    break

        if exact_results:
            raw_results = exact_results
            retrieval_mode = (
                "exact_reference"
            )

        elif not retrieval_law_ids:
            raw_results = []
            retrieval_mode = "semantic"

        else:
            raw_results = (
                selected_retriever.retrieve(
                    query=clean_query,
                    top_k=top_k,
                    filters={
                        "law_id": (
                            retrieval_law_ids
                        )
                    },
                    candidate_k=candidate_k,
                )
            )

            # -------------------------------------------------
            # Article Heading Retrieval
            #
            # Chi ho tro cau hoi tieng Viet khong dau
            # khi Law Scope da duoc xac dinh.
            #
            # Qwen semantic van la retrieval chinh.
            # Heading retrieval chi cuu cac truong hop
            # query khong dau lam semantic ranking giam manh.
            # -------------------------------------------------

            use_article_heading = (
                clean_query.isascii()
                and law_scope.law_id is not None
                and not law_scope.is_out_of_scope
                and bool(retrieval_law_ids)
            )

            if use_article_heading:
                store_documents = getattr(
                    selected_retriever.vector_store,
                    "documents",
                    [],
                )

                heading_candidates = (
                    rank_article_headings(
                        query=clean_query,
                        documents=store_documents,
                        allowed_law_ids=(
                            retrieval_law_ids
                        ),
                        scope_phrase=(
                            law_scope.matched_phrase
                        ),
                        top_k=1,
                    )
                )

                if heading_candidates:
                    best_heading = (
                        heading_candidates[0]
                    )

                    heading_score = float(
                        best_heading.get(
                            "heading_score",
                            0.0,
                        )
                    )

                    # High precision gate.
                    # Chi promote khi tieu de Dieu
                    # khop rat manh voi query.
                    if heading_score >= 0.90:
                        article_heading_match = (
                            best_heading
                        )

                        heading_article = str(
                            best_heading.get(
                                "article_number",
                                "",
                            )
                        )

                        heading_results = (
                            selected_retriever.retrieve(
                                query=clean_query,
                                top_k=top_k,
                                filters={
                                    "law_id": (
                                        retrieval_law_ids
                                    ),
                                    "article_number": (
                                        heading_article
                                    ),
                                },
                                candidate_k=candidate_k,
                            )
                        )

                        for result in heading_results:
                            result[
                                "heading_score"
                            ] = heading_score

                            result[
                                "retrieval_source"
                            ] = (
                                "article_heading"
                            )

                        # -----------------------------------------
                        # Heading article len truoc,
                        # semantic candidates theo sau.
                        # Loai trung theo FAISS id / chunk id.
                        # -----------------------------------------

                        merged_results = []
                        seen_result_keys = set()

                        for result in (
                            heading_results
                            + raw_results
                        ):
                            metadata = result.get(
                                "metadata",
                                {},
                            )

                            faiss_id = result.get(
                                "faiss_id"
                            )

                            chunk_id = metadata.get(
                                "chunk_id"
                            )

                            if faiss_id is not None:
                                result_key = (
                                    "faiss",
                                    int(faiss_id),
                                )

                            elif chunk_id:
                                result_key = (
                                    "chunk",
                                    str(chunk_id),
                                )

                            else:
                                result_key = (
                                    str(
                                        metadata.get(
                                            "law_id",
                                            "",
                                        )
                                    ),
                                    str(
                                        metadata.get(
                                            "article_number",
                                            "",
                                        )
                                    ),
                                    str(
                                        metadata.get(
                                            "clause_number",
                                            "",
                                        )
                                    ),
                                    str(
                                        result.get(
                                            "text",
                                            "",
                                        )
                                    )[:200],
                                )

                            if (
                                result_key
                                in seen_result_keys
                            ):
                                continue

                            seen_result_keys.add(
                                result_key
                            )

                            merged_results.append(
                                result
                            )

                        raw_results = (
                            merged_results[:top_k]
                        )

            retrieval_mode = "semantic"

        if retrieval_mode == "exact_reference":
            law_validity = None

            if exact_reference is not None:
                try:
                    law_validity = (
                        self.validity_resolver.evaluate_law(
                            exact_reference.law_id,
                            as_of=date_resolution.as_of,
                        )
                    )
                except Exception:
                    law_validity = None

            validity_report = {
                "valid_results": raw_results,
                "excluded_results": [],
                "reference_lookup": True,
                "referenced_law_validity": law_validity,
            }

        else:
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
            "retrieval_mode": (
                retrieval_mode
            ),
            "exact_reference": (
                exact_reference
            ),
            "effective_law_ids": (
                effective_law_ids
            ),
            "retrieval_law_ids": (
                retrieval_law_ids
            ),
            "law_scope": law_scope,
            "article_heading_match": (
                article_heading_match
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
