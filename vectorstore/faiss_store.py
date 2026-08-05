import faiss
import numpy as np
import pickle
import torch
from pathlib import Path

class FaissStore:
    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents= []

    def add(self, embeddings, documents):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=5):
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )
        results = []

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.documents[idx].page_content,
                "metadata": {
                    **self.documents[idx].metadata,
                    "source": Path(
                        self.documents[idx].metadata.get("source", "")
                    ).name
                },
                "score": float(distance)
            })
        return results
    def save(self, path):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        faiss.write_index(
            self.index,
            str(path.with_suffix(".index"))
        )
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(
                self.documents,
                f
            )
    def load(self, path):
        path = Path(path)

        self.index = faiss.read_index(
            str(path.with_suffix(".index"))
        )
        self.dimension = self.index.d

        with open(path.with_suffix(".pkl"), "rb") as f:
            self.documents = pickle.load(f)