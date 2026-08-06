from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from models.base_embedding import BaseEmbedding


class BGEEmbedding(BaseEmbedding):
    """
    Dense embedding wrapper cho BAAI/bge-m3.

    - Chỉ lấy dense_vecs ở giai đoạn hiện tại.
    - Chuẩn hóa L2 để dot product tương đương cosine similarity.
    - Giữ encode() và batch_encode() để tương thích code cũ.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        use_fp16: bool = False,
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size phải lớn hơn hoặc bằng 1.")

        if max_length < 64:
            raise ValueError("max_length phải lớn hơn hoặc bằng 64.")

        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
        )

    @staticmethod
    def _clean_texts(texts: Sequence[str]) -> list[str]:
        clean_texts: list[str] = []

        for index, text in enumerate(texts):
            value = str(text).strip()

            if not value:
                raise ValueError(
                    f"Văn bản tại vị trí {index} đang rỗng."
                )

            clean_texts.append(value)

        if not clean_texts:
            raise ValueError("Danh sách văn bản đang rỗng.")

        return clean_texts

    @staticmethod
    def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(
                "Embedding phải là vector 1 chiều hoặc ma trận 2 chiều."
            )

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        if np.any(norms <= 0):
            raise ValueError("Phát hiện embedding có norm bằng 0.")

        matrix = matrix / norms

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    def _model_encode(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts = self._clean_texts(texts)

        encode_kwargs = {
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }

        try:
            output = self.model.encode(
                clean_texts,
                **encode_kwargs,
            )
        except TypeError:
            # Tương thích một số bản FlagEmbedding cũ
            # không nhận các cờ return_*.
            output = self.model.encode(
                clean_texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )

        if isinstance(output, dict):
            dense_vectors = output.get("dense_vecs")
        else:
            dense_vectors = output

        if dense_vectors is None:
            raise RuntimeError(
                "BGE-M3 không trả về trường dense_vecs."
            )

        matrix = np.asarray(
            dense_vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise RuntimeError(
                "Kết quả BGE-M3 không phải ma trận 2 chiều."
            )

        if matrix.shape[0] != len(clean_texts):
            raise RuntimeError(
                "Số embedding không khớp số văn bản đầu vào."
            )

        return self.normalize_l2(matrix)

    def encode_document(self, text: str) -> np.ndarray:
        return self._model_encode([text])[0]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._model_encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        # BGE-M3 không yêu cầu query instruction mặc định.
        return self._model_encode([query])[0]

    # Tương thích interface cũ.
    def encode(self, text: str) -> np.ndarray:
        return self.encode_query(text)

    def batch_encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents(texts)

    def similarity(self, text1: str, text2: str) -> float:
        vector1 = self.encode_document(text1)
        vector2 = self.encode_document(text2)

        return float(np.dot(vector1, vector2))
