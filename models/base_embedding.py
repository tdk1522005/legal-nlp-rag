from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, text: str):
        pass
    @abstractmethod
    def batch_encode(self, texts):
        pass