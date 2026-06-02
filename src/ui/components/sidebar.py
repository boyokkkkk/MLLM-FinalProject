import streamlit as st
from utils.locales import TEXT


def render_sidebar():

    lang = st.session_state.language
    ui = TEXT[lang]

    with st.sidebar:

        st.markdown("# 📚 MLLM Assistant")

        st.caption(ui["subtitle"])

        st.divider()

        st.subheader(ui["navigation"])

        if st.button(ui["chat"], use_container_width=True):
            st.session_state.current_page = "Chat"

        if st.button(ui["documents"], use_container_width=True):
            st.session_state.current_page = "Documents"

        if st.button(ui["benchmark"], use_container_width=True):
            st.session_state.current_page = "Benchmark"

        st.divider()

        st.subheader(ui["language"])

        st.session_state.language = st.selectbox(
            ui["language"],
            ["English", "中文"],
            index=0 if st.session_state.language == "English" else 1,
        )

        st.divider()

        st.subheader(ui["generation"])

        st.session_state.answer_style = st.selectbox(
            ui["style"],
            [
                "Concise",
                "Detailed",
                "Academic"
            ]
        )

        st.session_state.temperature = st.slider(
            ui["temperature"],
            0.0,
            1.0,
            st.session_state.temperature,
            0.1,
        )

        st.session_state.max_tokens = st.slider(
            ui["max_tokens"],
            128,
            4096,
            st.session_state.max_tokens,
            128,
        )

        st.divider()

        if st.button(ui["clear"], use_container_width=True):
            st.session_state.messages = []
            st.rerun()