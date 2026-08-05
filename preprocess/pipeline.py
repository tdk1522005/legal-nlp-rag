from langchain_core.documents import Document
from preprocess.cleaner import TextCleaner
from preprocess.loader import DocumentLoader
from preprocess.chunker import DocumentChunker

class PreprocessPipeline:
    def __init__(self, data_dir):
        self.loader = DocumentLoader(data_dir)
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()

    def run(self):
        docs = self.loader.load()

        processed = []

        for doc in docs:
            text = self.cleaner.clean(doc.page_content)
            processed.append(
                Document(
                    page_content= text,
                    metadata = doc.metadata
                )
            )
        chunks = self.chunker.split(processed)
        return chunks