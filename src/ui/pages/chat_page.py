from __future__ import annotations

import base64

import streamlit as st

from components.chat_window import add_assistant_message, add_user_message, clear_current_chat, render_chat_history
from services.api_client import APIClient
from utils.locales import TEXT


def _read_text_files(files: list) -> list[str]:
    chunks: list[str] = []
    for file in files:
        try:
            raw = file.getvalue()
            text = raw.decode("utf-8", errors="ignore").strip()
            if text:
                chunks.append(f"[file:{file.name}]\n{text}")
        except Exception as exc:
            chunks.append(f"[file:{file.name}] read_failed: {exc}")
    return chunks


def _encode_images(files: list) -> list[str]:
    data_urls: list[str] = []
    for file in files:
        raw = file.getvalue()
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = file.type or "image/png"
        data_urls.append(f"data:{mime};base64,{b64}")
    return data_urls


def _render_suggestions(ui: dict[str, str]) -> None:
    st.markdown(f"<div class='panel-title'>{ui['prompt_suggestions']}</div>", unsafe_allow_html=True)
    suggestions = st.session_state.get("query_suggestions", [])
    cols = st.columns(len(suggestions) or 1, gap="small")
    for idx, prompt in enumerate(suggestions):
        with cols[idx]:
            if st.button(prompt, key=f"suggestion_{idx}", use_container_width=True):
                st.session_state.query_textarea = prompt
                st.rerun()


def _render_workspace_expander(ui: dict[str, str]) -> None:
    with st.expander(ui["context_assets"], expanded=False):
        text_files = st.file_uploader(
            ui["upload_text_files"],
            type=["txt", "md", "csv", "json"],
            accept_multiple_files=True,
            key="uploaded_text_files_widget",
        )
        image_files = st.file_uploader(
            ui["upload_images"],
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="uploaded_image_files_widget",
        )
        st.text_area(
            ui["paste_context"],
            placeholder=ui["paste_context_placeholder"],
            height=140,
            key="pasted_context_widget",
        )
        st.caption(ui["workspace_notes_hint"])

        st.session_state.uploaded_text_files = text_files or []
        st.session_state.uploaded_image_files = image_files or []
        st.session_state.uploaded_documents = [
            *[file.name for file in st.session_state.uploaded_text_files],
            *[image.name for image in st.session_state.uploaded_image_files],
        ]


def _build_scope_note(corpus_mode: str, ui: dict[str, str]) -> str:
    mapping = {
        ui["chat_mode_all"]: ui["scope_all"],
        ui["chat_mode_upload_first"]: ui["scope_upload_first"],
        ui["chat_mode_context_only"]: ui["scope_context_only"],
    }
    return mapping.get(corpus_mode, ui["scope_all"])


def _build_style_note(answer_style: str, ui: dict[str, str]) -> str:
    mapping = {
        "Concise": ui["style_note_concise"],
        "Detailed": ui["style_note_detailed"],
        "Academic": ui["style_note_academic"],
    }
    return mapping.get(answer_style, ui["style_note_detailed"])


def _render_backend_console(ui: dict[str, str], visible_citations: int, corpus_mode: str) -> None:
    health = st.session_state.get("last_backend_health") or {}
    online = isinstance(health, dict) and bool(health)
    version = health.get("version", "-") if online else "-"
    last_response = st.session_state.get("last_response") or {}
    model = last_response.get("model", "-")
    text_count = len(st.session_state.get("uploaded_text_files", []))
    image_count = len(st.session_state.get("uploaded_image_files", []))

    st.markdown(
        f"""
        <div class="panel-shell">
            <div class="panel-title">{ui["backend_console"]}</div>
            <div class="stat-grid compact two-col">
                <div class="stat-card">
                    <div class="stat-label">{ui["service_snapshot"]}</div>
                    <div class="stat-value small">{ui["service_online"] if online else ui["service_offline"]}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{ui["service_version"]}</div>
                    <div class="stat-value small">v{version}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{ui["latest_model"]}</div>
                    <div class="stat-value small">{model}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">{ui["evidence_limit"]}</div>
                    <div class="stat-value small">{visible_citations}</div>
                </div>
            </div>
            <div class="mini-divider"></div>
            <div class="capability-list">
                <div class="capability-chip">{ui["multimodal_ready"]}</div>
                <div class="capability-chip">{ui["retrieval_ready"]}</div>
                <div class="capability-chip">{ui["grounding_ready"]}</div>
            </div>
            <div class="mini-divider"></div>
            <div class="asset-summary">
                <div class="asset-summary-row"><span>{ui["uploaded_text_count"]}</span><strong>{text_count}</strong></div>
                <div class="asset-summary-row"><span>{ui["uploaded_image_count"]}</span><strong>{image_count}</strong></div>
                <div class="asset-summary-row"><span>{ui["active_scope"]}</span><strong>{corpus_mode}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_answer_card(ui: dict[str, str], visible_citations: int) -> None:
    last_response = st.session_state.get("last_response") or {}
    evidences = st.session_state.get("retrieved_evidence", [])[:visible_citations]
    answer = str(last_response.get("answer") or "").strip()

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(f"<div class='panel-shell answer-shell'><div class='panel-title'>{ui['answer_panel']}</div>", unsafe_allow_html=True)
        if answer:
            st.markdown(answer)
        else:
            st.caption(ui["no_evidence"])
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='panel-shell'><div class='panel-title'>{ui['citations_panel']}</div>", unsafe_allow_html=True)
        if not evidences:
            st.caption(ui["no_evidence"])
        else:
            for idx, evidence in enumerate(evidences, start=1):
                source = evidence.get("source") or evidence.get("source_ref") or "unknown"
                page = evidence.get("page")
                snippet = evidence.get("snippet", "")
                page_text = f"p.{page}" if page is not None else "-"
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-rank">#{idx}</div>
                        <div class="evidence-meta">{source}</div>
                        <div class="evidence-page">{page_text}</div>
                        <div class="evidence-snippet">{snippet or '-'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def render_chat_page() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    st.markdown(
        f"""
        <section class="workspace-intro">
            <div class="section-kicker">{ui["chat_workspace"]}</div>
            <h2 class="workspace-title">{ui["query_section_title"]}</h2>
            <p class="workspace-subtitle">{ui["query_section_subtitle"]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _render_suggestions(ui)

    left, right = st.columns([1.55, 0.95], gap="large")
    with left:
        st.markdown(f"<div class='composer-shell'><div class='panel-title'>{ui['query_controls']}</div>", unsafe_allow_html=True)
        with st.form("query_form", clear_on_submit=False):
            query = st.text_area(
                "",
                placeholder=ui["placeholder"],
                height=150,
                key="query_textarea",
                label_visibility="collapsed",
            )

            control_cols = st.columns(4, gap="small")
            with control_cols[0]:
                corpus_mode = st.selectbox(
                    ui["corpus_scope"],
                    [ui["chat_mode_all"], ui["chat_mode_upload_first"], ui["chat_mode_context_only"]],
                    index=0,
                )
            with control_cols[1]:
                visible_citations = st.selectbox(ui["evidence_limit"], [3, 5, 8], index=1)
            with control_cols[2]:
                st.session_state.temperature = st.selectbox(ui["temperature"], [0.0, 0.2, 0.4, 0.6], index=1)
            with control_cols[3]:
                st.session_state.max_tokens = st.selectbox(ui["max_tokens"], [512, 1024, 2048], index=1)

            _render_workspace_expander(ui)
            submitted = st.form_submit_button(ui["run_query"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        _render_backend_console(ui, visible_citations, corpus_mode)
        if st.session_state.chat_sessions.get(st.session_state.active_chat):
            with st.expander(ui["conversation"], expanded=False):
                render_chat_history()

    st.session_state.corpus_chip = corpus_mode

    if submitted and query.strip():
        uploaded_files = st.session_state.get("uploaded_text_files", [])
        uploaded_images = st.session_state.get("uploaded_image_files", [])
        pasted_context = st.session_state.get("pasted_context_widget", "")

        file_context = _read_text_files(uploaded_files)
        extra_context: list[str] = [
            f"[ui_scope] {_build_scope_note(corpus_mode, ui)}",
            f"[ui_answer_style] {_build_style_note(st.session_state.answer_style, ui)}",
        ]
        if isinstance(pasted_context, str) and pasted_context.strip():
            extra_context.append(pasted_context.strip())
        extra_context.extend(file_context)

        image_data_urls = _encode_images(uploaded_images)
        file_names = [file.name for file in uploaded_files]
        add_user_message(query, images=image_data_urls, files=file_names)

        st.session_state.last_query_meta = {
            "scope": corpus_mode,
            "visible_citations": visible_citations,
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
        }

        with st.spinner(ui["thinking"]):
            try:
                result = APIClient.chat(
                    query=query,
                    context=extra_context,
                    images=image_data_urls,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )
                answer = result.get("answer", "No answer returned.")
                citations = result.get("citations", [])[:visible_citations]
                result["citations"] = citations
                st.session_state.retrieved_evidence = citations
                st.session_state.last_response = result
            except Exception as exc:
                answer = f"{ui['request_failed']}: {exc}"
                st.session_state.retrieved_evidence = []
                st.session_state.last_response = {"answer": answer, "citations": [], "model": "-"}
            add_assistant_message(answer)
        st.rerun()

    if st.session_state.get("last_response") or st.session_state.get("retrieved_evidence"):
        _render_answer_card(ui, visible_citations)

    footer_left, footer_right = st.columns([1, 2], gap="large")
    with footer_left:
        if st.button(ui["clear_chat"], use_container_width=True):
            clear_current_chat()
            st.session_state.retrieved_evidence = []
            st.session_state.last_response = None
            st.rerun()
    with footer_right:
        st.caption(ui["status_helper"])
