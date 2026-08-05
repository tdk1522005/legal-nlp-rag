from models.bge_embedding import BGEEmbedding
from vectorstore.faiss_store import FaissStore
from retrieval.retriever import Retriever
from context.context_builder import ContextBuilder
from prompt.prompt_builder import PromptBuilder
from dotenv import load_dotenv
import os

from llm.gemini_llm import GeminiLLM

INDEX_PATH = "index/faiss"
TOP_K = 10

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Không tìm thấy GEMINI_API_KEY")
    llm = GeminiLLM(
        api_key=api_key
    )



    print("="*50)
    print("Loading index...")
    print("="*50)



    embedding_model = BGEEmbedding()

    store = FaissStore(dimension=1024)

    store.load(INDEX_PATH)

    print(f"Loaded {len(store.documents)} chunks.")

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=store
    )

    context_builder = ContextBuilder()

    prompt_builder = PromptBuilder()

    print("\nChatbot sẵn sàng")
    print("Nhập 'exit' để thoát.\n")

    while True:
        question = input("Bạn: ").strip()

        if question.lower() == "exit":
            break
        results = retriever.retrieve(
            question,
            top_k=TOP_K
        )
        '''
        print("\nTop documents:\n")
        
        for i, item in enumerate(results, start=1):
            print(f"{i}. Score: {item['score']:.4f}")
            print(f"   Source: {item['metadata']['source']}")
            print(f"   {item['text'][:150]}...")
            print()
        '''
        if not results:
            print("\nKhông tìm thấy tài liệu liên quan.\n")
            continue

        context = context_builder.build(results)

        prompt = prompt_builder.build(
            question= question,
            context= context
        )
        answer = llm.generate(prompt)
        print("\n================ ANSWER ================\n")
        print(answer)
        print("\n========================================\n")
if __name__ == "__main__":
    main()