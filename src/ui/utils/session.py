import streamlit as st


def initialize_session():

    defaults = {
        "current_page": "Chat",
        "temperature": 0.2,
        "max_tokens": 1024,

        "language": "中文",
        "answer_style": "Academic",

        # 多会话系统
        "chat_sessions": {
            "新建会话": []
        },

        "active_chat": "新建会话",

        # 后续文档系统预留
        "uploaded_documents": [],

        # 后续检索证据预留
        "retrieved_evidence": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value