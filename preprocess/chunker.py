from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self):
        separators = [
            "\n\n",
            "\n",
            ". ",
            "; ",
            ": ",
            " ",
            ""
        ]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=separators,
            add_start_index=True,
            strip_whitespace=True,
        )
    def split(self, documents):
        return self.splitter.split_documents(documents)