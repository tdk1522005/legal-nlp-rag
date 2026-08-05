from pathlib import Path
from preprocess.pipeline import PreprocessPipeline
from models.bge_embedding import BGEEmbedding
from vectorstore.faiss_store import FaissStore

DATA_DIR = "data"
INDEX_PATH = "index/faiss"

def main():
    print("="*50)
    print("Building FAISS Index....")
    print("="*50)

    pipeline = PreprocessPipeline(DATA_DIR)
    chunks = pipeline.run()

    print(f"Loaded {len(chunks)} chunks")

    texts = [doc.page_content for doc in chunks]

    model = BGEEmbedding()
    embeddings = model.batch_encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    store = FaissStore(
        dimension=embeddings.shape[1]
    )
    store.add(
        embeddings,
        chunks
    )

    Path("index").mkdir(exist_ok=True)

    store.save(INDEX_PATH)

    print("="*50)
    print("Index saved successfully!")
    print(f"Chunks  : {len(chunks)}")
    print(f"PAth    : {INDEX_PATH}")
    print("="*50)
if __name__ == "__main__":
    main()