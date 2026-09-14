from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"


class VietnameseReranker:
    """Cross-encoder reranker for Vietnamese query/document pairs."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 768,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        pairs = [
            [str(query), str(result.get("text", ""))]
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["rerank_score"] = float(score)
            item["retrieval_source"] = "reranker"
            reranked.append(item)

        reranked.sort(
            key=lambda item: float(item["rerank_score"]),
            reverse=True,
        )

        for rank, result in enumerate(reranked, start=1):
            result["rank"] = rank

        return reranked
