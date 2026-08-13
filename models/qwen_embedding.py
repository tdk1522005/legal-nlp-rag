from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from models.base_embedding import BaseEmbedding


class QwenEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        device: str = "cpu",
        dimension: int = 1024,
        batch_size: int = 2,
        query_instruction: str = (
            "Instruct: Given a Vietnamese legal question, "
            "retrieve relevant passages from Vietnamese legal "
            "documents that answer the question\n"
            "Query:"
        ),
    ) -> None:

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1."
            )

        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self.batch_size = batch_size
        self.query_instruction = query_instruction

        print(
            f"Loading {self.model_name} on {self.device}..."
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        actual_dimension = (
            self.model.get_embedding_dimension()
        )

        if actual_dimension != self.dimension:
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{actual_dimension} != {self.dimension}"
            )

    @staticmethod
    def _clean_text(text: str) -> str:
        value = str(text).strip()

        if not value:
            raise ValueError(
                "Text must not be empty."
            )

        return value

    @staticmethod
    def normalize_l2(
        embeddings: np.ndarray,
    ) -> np.ndarray:

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(
                "Embedding must be a vector or matrix."
            )

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        if np.any(norms <= 0):
            raise ValueError(
                "Embedding contains zero norm."
            )

        matrix = matrix / norms

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        prompt: str | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:

        clean_texts = [
            self._clean_text(text)
            for text in texts
        ]

        if not clean_texts:
            raise ValueError(
                "Text list must not be empty."
            )

        vectors = self.model.encode(
            clean_texts,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
            precision="float32",
        )

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.shape[1] != self.dimension:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{matrix.shape[1]} != {self.dimension}"
            )

        return self.normalize_l2(
            matrix
        )

    def encode_document(
        self,
        text: str,
    ) -> np.ndarray:

        return self._encode_texts(
            [text],
        )[0]

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        return self._encode_texts(
            texts,
            show_progress=True,
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        return self._encode_texts(
            [query],
            prompt=self.query_instruction,
        )[0]

    def encode(
        self,
        text: str,
    ) -> np.ndarray:

        return self.encode_query(
            text
        )

    def batch_encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        return self.encode_documents(
            texts
        )

    def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:

        vector1 = self.encode_document(
            text1
        )

        vector2 = self.encode_document(
            text2
        )

        return float(
            np.dot(
                vector1,
                vector2,
            )
        )
