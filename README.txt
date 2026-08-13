LEGAL NLP RAG - CHATBOT TƯ VẤN PHÁP LUẬT DÂN SỰ

1. Clone project

git clone https://github.com/tdk1522005/legal-nlp-rag.git
cd legal-nlp-rag
git switch stage-7-streamlit-web


2. Tạo môi trường Python

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


3. Chuẩn bị dữ liệu pháp luật

data/
└── raw/
    ├── current/
    ├── historical/
    ├── amendments/
    └── consolidated/


4. Kiểm tra metadata

python .\scripts\validate_legal_metadata.py


5. Xây dựng Legal Corpus

python .\scripts\build_legal_corpus.py

Quy trình:

DOCX
→ Chuẩn hóa văn bản
→ LegalDocxParser
→ Legal Tree
→ Legal Chunker
→ Metadata
→ JSONL Corpus


6. Tạo embedding

Mô hình:

Qwen/Qwen3-Embedding-0.6B

Mỗi legal chunk được chuyển thành vector 1024 chiều
và chuẩn hóa L2.

Cache embedding chính:

data/embeddings/qwen3_embedding_0_6b_1024


7. Xây dựng FAISS Index

Current Index:

python .\build_index.py `
    --corpus ".\data\chunks\default_retrieval_corpus.jsonl" `
    --output-dir ".\index\legal_dense_qwen"

Temporal Index:

python .\build_index.py `
    --corpus ".\data\chunks\legal_corpus.jsonl" `
    --output-dir ".\index\legal_temporal_qwen"

Hai index chính:

index/
├── legal_dense_qwen
└── legal_temporal_qwen

FAISS sử dụng IndexFlatIP.
Vector đã được chuẩn hóa L2 nên Inner Product
tương đương Cosine Similarity.


8. Một số lệnh kiểm tra

Parser:

python .\scripts\test_legal_parser.py

Legal Chunker:

python .\scripts\test_legal_chunker.py

Legal Graph:

python .\scripts\test_law_graph.py

FAISS + Qwen Embedding:

python .\scripts\test_dense_index.py

Đánh giá hệ thống:

$env:PYTHONPATH = (Get-Location).Path
python .\evaluation\evaluate_extended.py


9. Chạy chatbot bằng Terminal

python .\chat.py


10. Chạy giao diện Streamlit

Có thể chạy trực tiếp:

python -m streamlit run .\web_app.py `
    --server.fileWatcherType none

Hoặc dùng launcher:

.\start_chatbot.ps1


11. Kiến trúc hệ thống

OFFLINE:

DOCX
→ Tiền xử lý
→ Legal Parser
→ Legal Tree
→ Legal Chunk
→ Metadata
→ Qwen3-Embedding-0.6B
→ L2 Normalization
→ FAISS


ONLINE:

Câu hỏi người dùng
→ Xác định thời điểm pháp lý
→ Xác định văn bản có hiệu lực
→ Exact Retrieval hoặc Semantic Retrieval
→ Qwen Query Embedding
→ FAISS
→ Context Builder
→ Prompt Builder
→ Qwen3-4B
→ Câu trả lời
→ Streamlit


12. Mô hình sử dụng

Embedding:
Qwen/Qwen3-Embedding-0.6B

LLM:
Qwen3-4B

Vector Database:
FAISS

Giao diện:
Streamlit


13. Hai kho truy xuất

legal_dense_qwen:
Dùng cho hệ thống pháp luật hiện hành.

legal_temporal_qwen:
Dùng khi câu hỏi đề cập đến một thời điểm trong quá khứ.


14. Lưu ý

Project không sử dụng BGE-M3 hoặc Gemini trong pipeline hiện tại.

Không cần GEMINI_API_KEY.

Qwen3-Embedding-0.6B được sử dụng để tạo vector.

Qwen3-4B được chạy local thông qua llama.cpp để sinh câu trả lời.