from __future__ import annotations

import streamlit as st

from utils.locales import TEXT


def render_document_panel() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    uploaded_text_files = st.file_uploader(
        ui["upload_text_files"],
        type=["txt", "md", "csv", "json"],
        accept_multiple_files=True,
        key="uploaded_text_files_widget",
    )
    uploaded_images = st.file_uploader(
        ui["upload_images"],
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="uploaded_image_files_widget",
    )
    st.text_area(
        ui["paste_context"],
        placeholder=ui["paste_context_placeholder"],
        height=150,
        key="pasted_context_widget",
    )

    st.session_state.uploaded_text_files = uploaded_text_files or []
    st.session_state.uploaded_image_files = uploaded_images or []
    st.session_state.uploaded_documents = [
        *[file.name for file in st.session_state.uploaded_text_files],
        *[image.name for image in st.session_state.uploaded_image_files],
    ]

    text_count = len(st.session_state.uploaded_text_files)
    image_count = len(st.session_state.uploaded_image_files)
    pasted_context = st.session_state.get("pasted_context_widget", "")
    context_chars = len(pasted_context.strip()) if isinstance(pasted_context, str) else 0

    st.markdown(
        f"""
        <div class="panel-shell">
            <div class="panel-title">{ui["current_document"]}</div>
            <div class="stat-grid compact">
                <div class="stat-card">
                    <div class="stat-label">{ui["text_assets"]}</div>
                    <div class="stat-value">{text_count}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{ui["image_assets"]}</div>
                    <div class="stat-value">{image_count}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{ui["pasted_context_label"]}</div>
                    <div class="stat-value">{context_chars}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if text_count:
        with st.expander(ui["text_assets"], expanded=False):
            for file in st.session_state.uploaded_text_files:
                st.markdown(f"- `{file.name}`")

    if image_count:
        with st.expander(ui["image_assets"], expanded=False):
            cols = st.columns(2, gap="small")
            for idx, image in enumerate(st.session_state.uploaded_image_files):
                with cols[idx % 2]:
                    st.image(image, caption=image.name, use_container_width=True)

    if not st.session_state.uploaded_documents and context_chars == 0:
        st.caption(ui["no_document"])
