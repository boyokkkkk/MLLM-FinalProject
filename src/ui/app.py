from pathlib import Path

import streamlit as st

from components.header import render_header
from components.navbar import render_navbar
from pages.benchmark_page import render_benchmark_page
from pages.chat_page import render_chat_page
from pages.documents_page import render_documents_page
from utils.session import initialize_session

st.set_page_config(
    page_title="MLLM Research Workspace",
    page_icon="📚",
    layout="wide",
)


def load_css() -> None:
    css_file = Path(__file__).parent / "styles" / "newsprint.css"
    with css_file.open(encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


initialize_session()
load_css()

page = st.session_state.get("current_page", "Chat")

render_header()
render_navbar()

if page == "Chat":
    render_chat_page()
elif page == "Documents":
    render_documents_page()
elif page == "Benchmark":
    render_benchmark_page()
else:
    st.warning(f"Unknown page: {page}")
