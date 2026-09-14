from __future__ import annotations

from typing import Any


class RerankedTemporalRetrievalRouter:
    """Wrap the Branch 8 router with a reranking stage.

    The existing routing, temporal logic, law-scope filtering, exact-reference
    lookup, article-heading retrieval and validity filtering are preserved.
    The reranker only re-orders semantic candidates before final Top-K output.
    """

    def __init__(
        self,
        base_router: Any,
        reranker: Any,
        *,
        candidate_k: int = 20,
    ) -> None:
        if base_router is None:
            raise ValueError("base_router không được để trống.")
        if reranker is None:
            raise ValueError("reranker không được để trống.")
        if candidate_k < 1:
            raise ValueError("candidate_k phải lớn hơn hoặc bằng 1.")

        self.base_router = base_router
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn hoặc bằng 1.")

        route_output = self.base_router.retrieve(
            query=query,
            top_k=max(top_k, self.candidate_k),
        )

        # Explicit legal references are already exact document lookups.
        if route_output.get("retrieval_mode") == "exact_reference":
            route_output["reranker_used"] = False
            route_output["rerank_candidate_k"] = None
            return route_output

        candidates = list(route_output.get("results", []))
        if not candidates:
            route_output["reranker_used"] = False
            route_output["rerank_candidate_k"] = 0
            return route_output

        reranked = self.reranker.rerank(
            query=str(query).strip(),
            results=candidates,
        )[:top_k]

        for rank, result in enumerate(reranked, start=1):
            result["rank"] = rank

        route_output["raw_results"] = reranked
        route_output["results"] = reranked
        route_output["reranker_used"] = True
        route_output["rerank_candidate_k"] = len(candidates)

        validity_report = route_output.get("validity_report")
        if isinstance(validity_report, dict):
            validity_report = dict(validity_report)
            validity_report["valid_results"] = reranked
            route_output["validity_report"] = validity_report

        return route_output
