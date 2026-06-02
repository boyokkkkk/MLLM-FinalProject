import streamlit as st

from pages.chat_page import render_chat_page
from components.sidebar import render_sidebar
from utils.session import initialize_session
from utils.locales import TEXT

st.set_page_config(
    page_title="MLLM Assistant",
    page_icon="📚",
    layout="wide",
)

initialize_session()

render_sidebar()

lang = st.session_state.language
ui = TEXT[lang]

page = st.session_state.get("current_page", "Chat")

if page == "Chat":

    render_chat_page()

elif page == "Documents":

    st.title(ui["documents"])

    if lang == "中文":

        st.info("文档管理模块将在第二阶段实现。")

    else:

        st.info("Document Management Module Coming Soon.")

elif page == "Benchmark":

    st.title(ui["benchmark"])

    if lang == "中文":

        st.info("评测模块将在第二阶段实现。")

    else:

        st.info("Benchmark Module Coming Soon.")