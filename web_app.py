from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from chat import (
    CONTEXT_MAX_CHARS,
    CURRENT_CANDIDATE_K,
    CURRENT_INDEX_DIR,
    TEMPORAL_CANDIDATE_K,
    TEMPORAL_INDEX_DIR,
    TOP_K,
    load_manifest,
    load_vector_store,
    validate_manifest_compatibility,
)
from context.context_builder import ContextBuilder
from llm.gemini_llm import GeminiLLM
from models.bge_embedding import BGEEmbedding
from prompt.prompt_builder import PromptBuilder
from retrieval.retriever import Retriever
from retrieval.temporal_router import (
    TemporalRetrievalRouter,
)
from temporal.query_date_resolver import (
    QueryDateResolver,
)
from validity.validity_resolver import (
    ValidityResolver,
)


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(
    PROJECT_ROOT / ".env"
)

st.set_page_config(
    page_title="Trợ lý Pháp luật Dân sự",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(
    show_spinner=False
)
def load_chatbot() -> dict[str, Any]:
    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "Không tìm thấy GEMINI_API_KEY "
            "trong file .env."
        )

    current_manifest = load_manifest(
        CURRENT_INDEX_DIR
    )

    temporal_manifest = load_manifest(
        TEMPORAL_INDEX_DIR
    )

    validate_manifest_compatibility(
        current_manifest,
        temporal_manifest,
    )

    embedding_model = BGEEmbedding(
        model_name=str(
            current_manifest["model_name"]
        ),
        use_fp16=False,
        batch_size=1,
        max_length=int(
            current_manifest["max_length"]
        ),
    )

    current_store = load_vector_store(
        CURRENT_INDEX_DIR,
        current_manifest,
    )

    temporal_store = load_vector_store(
        TEMPORAL_INDEX_DIR,
        temporal_manifest,
    )

    current_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=current_store,
    )

    temporal_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=temporal_store,
    )

    router = TemporalRetrievalRouter(
        current_retriever=(
            current_retriever
        ),
        temporal_retriever=(
            temporal_retriever
        ),
        validity_resolver=(
            ValidityResolver()
        ),
        query_date_resolver=(
            QueryDateResolver()
        ),
        current_candidate_k=(
            CURRENT_CANDIDATE_K
        ),
        temporal_candidate_k=(
            TEMPORAL_CANDIDATE_K
        ),
    )

    return {
        "router": router,
        "context_builder": ContextBuilder(
            max_chars=CONTEXT_MAX_CHARS,
            include_score=False,
        ),
        "prompt_builder": PromptBuilder(),
        "llm": GeminiLLM(
            api_key=api_key
        ),
        "current_count": len(
            current_store.documents
        ),
        "temporal_count": len(
            temporal_store.documents
        ),
    }


def collect_sources(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for result in results:
        metadata = result.get(
            "metadata",
            {},
        )

        citation = str(
            metadata.get("citation")
            or ""
        ).strip()

        if not citation:
            continue

        if citation in seen:
            continue

        seen.add(citation)

        score = result.get("score")

        sources.append(
            {
                "citation": citation,
                "score": (
                    float(score)
                    if score is not None
                    else None
                ),
            }
        )

    return sources


def collect_validity_notes(
    report: dict[str, Any],
) -> list[str]:
    notes: list[str] = []

    evaluations = report.get(
        "law_evaluations",
        {},
    )

    for evaluation in evaluations.values():
        title = str(
            evaluation.get("title")
            or evaluation.get("law_id")
            or "Văn bản"
        )

        state = str(
            evaluation.get(
                "validity_state",
                "",
            )
        )

        if state and state != "effective":
            notes.append(
                f"{title}: {state}."
            )

        for amendment in evaluation.get(
            "amending_laws",
            [],
        ):
            amendment_title = str(
                amendment.get("title")
                or amendment.get("law_id")
                or "văn bản sửa đổi"
            )

            notes.append(
                f"{title} được sửa đổi bởi "
                f"{amendment_title}."
            )

        for warning in evaluation.get(
            "warnings",
            [],
        ):
            text = str(warning).strip()

            if text:
                notes.append(text)

    return list(
        dict.fromkeys(notes)
    )


def answer_question(
    question: str,
    chatbot: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    route_output = chatbot[
        "router"
    ].retrieve(
        query=question,
        top_k=TOP_K,
    )

    results = route_output[
        "results"
    ]

    resolution = route_output[
        "date_resolution"
    ]

    metadata = {
        "as_of": (
            route_output[
                "as_of"
            ].isoformat()
        ),
        "index_name": (
            route_output[
                "index_name"
            ]
        ),
        "matched_text": (
            resolution.matched_text
        ),
        "warning": (
            resolution.warning
        ),
        "sources": collect_sources(
            results
        ),
        "validity_notes": (
            collect_validity_notes(
                route_output[
                    "validity_report"
                ]
            )
        ),
    }

    if not results:
        return (
            "Không tìm thấy tài liệu pháp luật "
            "có hiệu lực phù hợp với câu hỏi "
            "và thời điểm được yêu cầu.",
            metadata,
        )

    context = chatbot[
        "context_builder"
    ].build(results)

    if not context:
        return (
            "Không thể tạo ngữ cảnh từ các "
            "văn bản pháp luật đã truy xuất.",
            metadata,
        )

    temporal_question = (
        f"{question}\n\n"
        "Thời điểm pháp lý cần áp dụng: "
        f"{metadata['as_of']}."
    )

    prompt = chatbot[
        "prompt_builder"
    ].build(
        question=temporal_question,
        context=context,
    )

    answer = chatbot[
        "llm"
    ].generate(prompt)

    return str(answer), metadata


def render_metadata(
    metadata: dict[str, Any],
) -> None:
    st.caption(
        "Ngày áp dụng: "
        f"{metadata['as_of']} · "
        "Chỉ mục: "
        f"{metadata['index_name']}"
    )

    with st.expander(
        "Xem nguồn và thông tin truy xuất"
    ):
        if metadata.get(
            "matched_text"
        ):
            st.write(
                "**Cụm thời gian nhận diện:** "
                f"{metadata['matched_text']}"
            )

        if metadata.get("warning"):
            st.warning(
                metadata["warning"]
            )

        validity_notes = metadata.get(
            "validity_notes",
            [],
        )

        if validity_notes:
            st.markdown(
                "**Tình trạng hiệu lực:**"
            )

            for note in validity_notes:
                st.write(f"- {note}")

        sources = metadata.get(
            "sources",
            [],
        )

        if sources:
            st.markdown(
                "**Nguồn pháp luật:**"
            )

            for source in sources:
                citation = source[
                    "citation"
                ]

                score = source.get(
                    "score"
                )

                if score is None:
                    st.write(
                        f"- {citation}"
                    )
                else:
                    st.write(
                        f"- {citation} "
                        f"— score: {score:.4f}"
                    )
        else:
            st.write(
                "Không có nguồn truy xuất."
            )


st.title(
    "⚖️ Trợ lý Pháp luật Dân sự"
)

st.caption(
    "Hỏi đáp dựa trên văn bản pháp luật "
    "và thời điểm có hiệu lực."
)

try:
    with st.spinner(
        "Đang tải mô hình và dữ liệu pháp luật..."
    ):
        chatbot = load_chatbot()

except Exception as error:
    st.error(
        "Không thể khởi động chatbot: "
        f"{error}"
    )
    st.stop()


with st.sidebar:
    st.header(
        "Thông tin hệ thống"
    )

    st.metric(
        "Chunk hiện hành",
        chatbot["current_count"],
    )

    st.metric(
        "Chunk theo thời gian",
        chatbot["temporal_count"],
    )

    st.write(
        "Mô hình embedding: BGE-M3"
    )

    st.write(
        "Mô hình trả lời: Gemini"
    )

    if st.button(
        "Xóa lịch sử trò chuyện"
    ):
        st.session_state.messages = []
        st.rerun()

    st.warning(
        "Kết quả chỉ phục vụ mục đích "
        "tham khảo và học tập, không thay "
        "thế ý kiến tư vấn pháp lý chuyên môn."
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bạn có thể hỏi về pháp luật "
                "dân sự hiện hành hoặc tại một "
                "thời điểm trong quá khứ."
            ),
            "metadata": None,
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        metadata = message.get(
            "metadata"
        )

        if metadata:
            render_metadata(
                metadata
            )


question = st.chat_input(
    "Nhập câu hỏi pháp luật..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "metadata": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message(
        "assistant"
    ):
        try:
            with st.spinner(
                "Đang tra cứu văn bản pháp luật..."
            ):
                answer, metadata = (
                    answer_question(
                        question,
                        chatbot,
                    )
                )

            st.markdown(answer)

            render_metadata(
                metadata
            )

        except Exception as error:
            answer = (
                "Đã xảy ra lỗi khi xử lý "
                f"câu hỏi: {error}"
            )

            metadata = None

            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "metadata": metadata,
        }
    )
