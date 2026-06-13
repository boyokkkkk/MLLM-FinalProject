from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from services.api_client import APIClient
from utils.locales import TEXT


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def render_documents_page() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    docs_path = PROJECT_ROOT / "data" / "processed" / "documents" / "documents.jsonl"
    chunks_path = PROJECT_ROOT / "data" / "processed" / "retrieval" / "text_chunks.jsonl"
    processed_documents = _count_jsonl_rows(docs_path)
    indexed_chunks = _count_jsonl_rows(chunks_path)
    uploaded_text = len(st.session_state.get("uploaded_text_files", []))
    uploaded_image = len(st.session_state.get("uploaded_image_files", []))

    st.markdown(
        f"""
        <section class="workspace-intro">
            <div class="section-kicker">{ui["documents_title"]}</div>
            <h2 class="workspace-title">{ui["documents_title"]}</h2>
            <p class="workspace-subtitle">{ui["documents_subtitle"]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4, gap="small")
    metrics = [
        (ui["processed_documents"], processed_documents),
        (ui["indexed_chunks"], indexed_chunks),
        (ui["uploaded_text_count"], uploaded_text),
        (ui["uploaded_image_count"], uploaded_image),
    ]
    for col, (label, value) in zip(stat_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="stat-card large">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown(f"<div class='panel-shell'><div class='panel-title'>{ui['workspace_inventory']}</div>", unsafe_allow_html=True)
        uploaded_text_files = st.session_state.get("uploaded_text_files", [])
        uploaded_images = st.session_state.get("uploaded_image_files", [])
        if not uploaded_text_files and not uploaded_images:
            st.caption(ui["no_document"])
        else:
            for file in uploaded_text_files:
                st.markdown(f"<div class='asset-row'>TXT <strong>{file.name}</strong></div>", unsafe_allow_html=True)
            for file in uploaded_images:
                st.markdown(f"<div class='asset-row'>IMG <strong>{file.name}</strong></div>", unsafe_allow_html=True)

        pasted_context = st.session_state.get("pasted_context_widget", "")
        if isinstance(pasted_context, str) and pasted_context.strip():
            st.markdown(f"<div class='panel-title'>{ui['workspace_context_preview']}</div>", unsafe_allow_html=True)
            st.code(pasted_context[:1500], language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='panel-shell'><div class='panel-title'>{ui['document_inventory']}</div>", unsafe_allow_html=True)
        if docs_path.exists():
            preview_rows: list[dict[str, str]] = []
            with docs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(preview_rows) >= 8:
                        break
                    payload = json.loads(line)
                    preview_rows.append(
                        {
                            "document_id": str(payload.get("document_id", "")),
                            "source_path": str(payload.get("source_path", "")),
                            "pages": str(len(payload.get("pages", []))) if isinstance(payload.get("pages"), list) else "-",
                        }
                    )
            if preview_rows:
                st.dataframe(preview_rows, use_container_width=True, hide_index=True)
            else:
                st.caption(ui["empty_table"])
        else:
            st.caption("`data/processed/documents/documents.jsonl` not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f"<div class='panel-shell'><div class='panel-title'>{ui['api_endpoints']}</div>", unsafe_allow_html=True)
        endpoint_rows = [
            {
                ui["endpoint"]: capability["endpoint"],
                ui["purpose"]: capability["purpose"],
                ui["example_payload"]: capability["mapping"],
            }
            for capability in APIClient.capabilities()
        ]
        st.dataframe(endpoint_rows, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        health = st.session_state.get("last_backend_health") or {}
        version = health.get("version", "-") if isinstance(health, dict) else "-"
        st.markdown(
            f"""
            <div class="panel-shell">
                <div class="panel-title">{ui["service_snapshot"]}</div>
                <div class="metric-callout">
                    <div class="metric-run">{ui["api_base"]}</div>
                    <div class="metric-line">{APIClient.get_api_base()}</div>
                </div>
                <div class="mini-divider"></div>
                <div class="asset-summary">
                    <div class="asset-summary-row"><span>{ui["service_version"]}</span><strong>v{version}</strong></div>
                    <div class="asset-summary-row"><span>{ui["multimodal_ready"]}</span><strong>ON</strong></div>
                    <div class="asset-summary-row"><span>{ui["retrieval_ready"]}</span><strong>ON</strong></div>
                    <div class="asset-summary-row"><span>{ui["grounding_ready"]}</span><strong>ON</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
