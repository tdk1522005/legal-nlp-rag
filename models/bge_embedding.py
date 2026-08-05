from FlagEmbedding import BGEM3FlagModel
import numpy as np
from models.base_embedding import BaseEmbedding

class BGEEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

    def encode(self, text):
        embedding = self.model.encode(
            [text]
        )["dense_vecs"]


        return embedding[0]
    def batch_encode(self, texts):
        embeddings = self.model.encode(
            texts
        )["dense_vecs"]
        return embeddings

    def similarity(self, text1, text2):
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)

        similarity = np.dot(vec1, vec2)

        return float(similarity)
