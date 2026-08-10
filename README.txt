1. Clone project từ GitHub

git clone https://github.com/tdk1522005/legal-nlp-rag.git
cd legal-nlp-rag

Phiên bản mới nhất hiện đang nằm ở branch: git switch stage-7-streamlit-web

2. Tạo môi trường ảo Python

python -m venv .venv

Kích hoạt môi trường: .\.venv\Scripts\Activate.ps1

3. Cài đặt thư viện

python -m pip install -r requirements.txt

python -m pip install python-docx networkx

4. Tạo file .env

Tại thư mục gốc của project:

legal-nlp-rag/

tạo file: .env

Nội dung: GEMINI_API_KEY=YOUR_GEMINI_API_KEY

5. Chuẩn bị dữ liệu pháp luật

data/
└── raw/
    ├── current(Văn bản hiện hành)/
    ├── historical(Văn bản lịch sử/đã hết hiệu lực)/
    ├── amendments(Văn bản sửa đổi, bổ sung)/
    └── consolidated(Văn bản hợp nhất)/

6. Kiểm tra metadata pháp luật

Kiểm tra các file: python .\scripts\validate_legal_metadata.py

7. Xây dựng Legal Corpus

python .\scripts\build_legal_corpus.py

[thực hiện DOCX -> LegalDocxParser(Phân tích cấu trúc luật) -> Legal Tree(Cây phần => Chương => Điều => Khoản => Điểm) -> Legal Chunker(Chia thành các legal chunk) -> JSONL Corpus]


8. Build FAISS Index cho pháp luật hiện hành

python .\build_index.py --corpus ".\data\chunks\default_retrieval_corpus.jsonl" --output-dir ".\index\legal_dense"

[default_retrieval_corpus.jsonl -> BAAI/bge-m3 -> Dense Embedding 1024 chiều -> L2 Normalization -> FAISS IndexFlatIP -> index/legal_dense]


9. Build FAISS Index theo thời gian

Để chatbot có thể trả lời câu hỏi legal trong qua khứ build thêm temporal index:

python .\build_index.py --corpus ".\data\chunks\legal_corpus.jsonl" --output-dir ".\index\legal_temporal"

10. Build lại index khi index đã tồn tại

Để ghi đè thêm index

ví dụ Current index:
python .\build_index.py --corpus ".\data\chunks\default_retrieval_corpus.jsonl" --output-dir ".\index\legal_dense" --force

Temporal index:
python .\build_index.py --corpus ".\data\chunks\legal_corpus.jsonl" --output-dir ".\index\legal_temporal" --force

11. Một số lệnh kiểm tra hệ thống

Kiểm tra parser: python .\scripts\test_legal_parser.py
Kiểm tra Legal Chunker: python .\scripts\test_legal_chunker.py
Kiểm tra Graph quan hệ pháp luật: python .\scripts\test_law_graph.py
Kiểm tra FAISS Dense Index: python .\scripts\test_dense_index.py

12. Chạy chatbot trên Terminal

python .\chat.py

[Đọc current index manifest -> Đọc temporal index manifest -> load BGE-M3 -> Load current FAISS index -> Load temporal FAISS index -> Khởi tạo Retriever -> Khởi tạo ValidityResolver -> Khởi tạo QueryDateResolver -> Khởi tạo TemporalRetrievalRouter -> Khởi tạo ContextBuilder -> Khởi tạo PromptBuilder -> Khởi tạo Gemini ->Chatbot sẳn sàng]

13. Chạy giao diện Web bằng Streamlit

python -m streamlit run .\web_app.py



TÓM TẮT LỆNH TỪ ĐẦU ĐẾN CUỐI

# 1. Tạo môi trường ảo

python -m venv .venv


# 2. Kích hoạt môi trường ảo

.\.venv\Scripts\Activate.ps1


# 3. Cài thư viện

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m pip install python-docx networkx


# 4. Kiểm tra metadata pháp luật

python .\scripts\validate_legal_metadata.py


# 5. Parse + Chunk + Build Legal Corpus
# DOCX
# → phân tích cấu trúc luật
# → tạo Legal Tree
# → chia Legal Chunk
# → tạo corpus JSONL

python .\scripts\build_legal_corpus.py


# 6. Tokenize + Embedding + Build Current FAISS Index
# default_retrieval_corpus.jsonl
# → tokenizer nội bộ của BGE-M3
# → BAAI/bge-m3
# → Dense Embedding 1024 chiều
# → L2 Normalization
# → FAISS IndexFlatIP
# → index/legal_dense

python .\build_index.py --corpus ".\data\chunks\default_retrieval_corpus.jsonl" --output-dir ".\index\legal_dense"


# 7. Tokenize + Embedding + Build Temporal FAISS Index
# legal_corpus.jsonl
# → tokenizer nội bộ của BGE-M3
# → Dense Embedding
# → L2 Normalization
# → FAISS IndexFlatIP
# → index/legal_temporal

python .\build_index.py --corpus ".\data\chunks\legal_corpus.jsonl" --output-dir ".\index\legal_temporal"


# 8. Chạy chatbot bằng Terminal

python .\chat.py


# 9. Hoặc chạy chatbot bằng giao diện Web Streamlit

python -m streamlit run .\web_app.py



 