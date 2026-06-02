import streamlit as st


def initialize_session():

    defaults = {
        "messages": [],
        "current_page": "Chat",
        "temperature": 0.2,
        "max_tokens": 1024,

        # NEW
        "language": "English",
        "answer_style": "Academic",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value