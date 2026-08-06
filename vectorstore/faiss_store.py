from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FaissStore:
    """
    FAISS IndexFlatIP + metadata JSONL.

    Vector tài liệu và query đều được chuẩn hóa L2,
    vì vậy inner product chính là cosine similarity.
    """

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension < 1:
            raise ValueError("dimension phải lớn hơn hoặc bằng 1.")

        self.dimension = dimension
        self.index = (
            faiss.IndexFlatIP(dimension)
            if dimension is not None
            else None
        )
        self.documents: list[dict[str, Any]] = []

    @staticmethod
    def _to_numpy(vectors: Any) -> np.ndarray:
        if torch is not None and isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError("Vector phải là mảng 1D hoặc ma trận 2D.")

        if matrix.shape[0] == 0:
            raise ValueError("Không có vector để xử lý.")

        matrix = np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

        faiss.normalize_L2(matrix)
        return matrix

    @staticmethod
    def _document_to_dict(document: Any) -> dict[str, Any]:
        if isinstance(document, dict):
            item = dict(document)
        elif hasattr(document, "page_content"):
            item = dict(getattr(document, "metadata", {}) or {})
            item["text"] = str(getattr(document, "page_content", ""))
        else:
            raise TypeError(
                "Document phải là dict hoặc đối tượng có page_content và metadata."
            )

        text = str(
            item.get("text")
            or item.get("content")
            or ""
        ).strip()

        if not text:
            raise ValueError("Document không có text/content hợp lệ.")

        item["text"] = text
        return item

    def add(self, embeddings: Any, documents: list[Any]) -> None:
        matrix = self._to_numpy(embeddings)

        if len(documents) != matrix.shape[0]:
            raise ValueError(
                "Số document không khớp số embedding: "
                f"{len(documents)} != {matrix.shape[0]}"
            )

        if self.index is None:
            self.dimension = int(matrix.shape[1])
            self.index = faiss.IndexFlatIP(self.dimension)

        if matrix.shape[1] != self.dimension:
            raise ValueError(
                "Sai số chiều embedding: "
                f"{matrix.shape[1]} != {self.dimension}"
            )

        clean_documents = [
            self._document_to_dict(document)
            for document in documents
        ]

        self.index.add(matrix)
        self.documents.extend(clean_documents)

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "Số vector trong FAISS không khớp metadata."
            )

    @staticmethod
    def _matches_filters(
        document: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = document.get(key)

            if isinstance(expected, (list, tuple, set)):
                expected_values = set(expected)

                if isinstance(actual, list):
                    if not expected_values.intersection(actual):
                        return False
                elif actual not in expected_values:
                    return False
            elif isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False

        return True

    def search(
        self,
        query_embedding: Any,
        top_k: int = 5,
        *,
        filters: dict[str, Any] | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISS index chưa được khởi tạo.")

        if top_k < 1:
            raise ValueError("top_k phải lớn hơn hoặc bằng 1.")

        if self.index.ntotal == 0:
            return []

        query_matrix = self._to_numpy(query_embedding)

        if query_matrix.shape[0] != 1:
            raise ValueError("search() chỉ nhận một query embedding.")

        if query_matrix.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding sai số chiều: "
                f"{query_matrix.shape[1]} != {self.dimension}"
            )

        if candidate_k is None:
            candidate_k = top_k * 10 if filters else top_k

        candidate_k = max(candidate_k, top_k)
        candidate_k = min(candidate_k, int(self.index.ntotal))

        scores, indices = self.index.search(query_matrix, candidate_k)

        results: list[dict[str, Any]] = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            document = self.documents[int(index_position)]

            if not self._matches_filters(document, filters):
                continue

            text = str(document.get("text", ""))
            metadata = {
                key: value
                for key, value in document.items()
                if key != "text"
            }

            results.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": float(score),
                    "faiss_id": int(index_position),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def save(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        if self.index is None:
            raise RuntimeError("Không có FAISS index để lưu.")

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "Không thể lưu vì vector và metadata không đồng bộ."
            )

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with metadata_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for document in self.documents:
                file.write(
                    json.dumps(document, ensure_ascii=False) + "\n"
                )

    def load(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy metadata: {metadata_path}"
            )

        self.index = faiss.read_index(str(index_path))
        self.dimension = int(self.index.d)

        documents: list[dict[str, Any]] = []

        with metadata_path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Metadata JSONL lỗi tại dòng "
                        f"{line_number}: {error}"
                    ) from error

                documents.append(self._document_to_dict(item))

        if self.index.ntotal != len(documents):
            raise RuntimeError(
                "Số vector và metadata không khớp: "
                f"{self.index.ntotal} != {len(documents)}"
            )

        self.documents = documents
