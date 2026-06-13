import streamlit as st


def initialize_session() -> None:
    defaults = {
        "current_page": "Chat",
        "temperature": 0.2,
        "max_tokens": 1024,
        "language": "中文",
        "answer_style": "Detailed",
        "chat_sessions": {
            "新建会话 1": [],
        },
        "active_chat": "新建会话 1",
        "uploaded_documents": [],
        "uploaded_text_files": [],
        "uploaded_image_files": [],
        "retrieved_evidence": [],
        "last_response": None,
        "last_backend_health": None,
        "corpus_chip": "Global corpus",
        "last_query_meta": {},
        "query_suggestions": [
            "Summarize the most relevant evidence for this question.",
            "Which page most directly supports the answer?",
            "这张图里最关键的信息是什么？",
        ],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
