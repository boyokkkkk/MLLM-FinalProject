import streamlit as st
from pathlib import Path

from pages.chat_page import render_chat_page
from utils.session import initialize_session
from utils.locales import TEXT

from components.header import render_header
from components.ticker import render_ticker
from components.document_panel import render_document_panel
from components.evidence_panel import render_evidence_panel
from components.navbar import render_navbar


st.set_page_config(
    page_title="MLLM Assistant",
    page_icon="📚",
    layout="wide",
)


def load_css():

    css_file = (
        Path(__file__).parent
        / "styles"
        / "newsprint.css"
    )

    with open(css_file, encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


initialize_session()
load_css()

lang = st.session_state.language
ui = TEXT[lang]

page = st.session_state.get(
    "current_page",
    "Chat"
)

# ==========================
# CHAT PAGE
# ==========================

if page == "Chat":

    render_header()

    render_navbar()

    render_ticker()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(
        [2, 6],
        gap="large"
    )

    with col1:

        render_document_panel()

    with col2:

        render_chat_page()

    st.markdown("<br>", unsafe_allow_html=True)

    render_evidence_panel()

# ==========================
# DOCUMENT PAGE
# ==========================

elif page == "Documents":

    render_header()
    render_navbar()

    st.markdown(
        """
        <h2 class='section-title'>
        DOCUMENT ARCHIVE
        </h2>
        """,
        unsafe_allow_html=True
    )

    if lang == "中文":

        st.info(
            "文档管理模块将在第二阶段实现。"
        )

    else:

        st.info(
            "Document Management Module Coming Soon."
        )

# ==========================
# BENCHMARK PAGE
# ==========================

elif page == "Benchmark":

    render_header()
    render_navbar()

    st.markdown(
        """
        <h2 class='section-title'>
        BENCHMARK CENTER
        </h2>
        """,
        unsafe_allow_html=True
    )

    if lang == "中文":

        st.info(
            "评测模块将在第二阶段实现。"
        )

    else:

        st.info(
            "Benchmark Module Coming Soon."
        )