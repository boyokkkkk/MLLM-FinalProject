from __future__ import annotations

import streamlit as st


def render_markdown_text(content: str) -> None:
    st.markdown(content, unsafe_allow_html=False)
