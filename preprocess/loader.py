from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
)


class DocumentLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load(self):
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.docx",
            loader_cls= Docx2txtLoader,
            show_progress=True,
            use_multithreading=False
        )
        return loader.load()