from typing import List



class ContextBuilder:

    def __init__(self, separator="\n\n-------------------------\n\n", include_metadata=True):
        self.separator = separator
        self.include_metadata = include_metadata

    def build(self, retrieved_docs: List[dict])->str:
        contexts = []

        for item in retrieved_docs:
            text =""
            if self.include_metadata:
                metadata = item["metadata"]
                source = metadata.get("source", "Không rõ nguồn")
                article = metadata.get("article", "")
                text += f"[Nguồn: {source}"
                if article:
                    text += f" - {article}"
                text += "]\n\n"
            text += item["text"]
            contexts.append(text)

        return self.separator.join(contexts)