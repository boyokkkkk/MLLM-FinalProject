from __future__ import annotations

import streamlit as st

from utils.locales import TEXT


def render_evidence_panel() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    evidences = st.session_state.retrieved_evidence
    last_response = st.session_state.get("last_response") or {}

    st.markdown(
        f"""
        <div class="panel-shell">
            <div class="panel-title">{ui["retrieved_evidence"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not evidences:
        st.caption(ui["no_evidence"])
        return

    summary_cols = st.columns([1, 1, 4], gap="small")
    with summary_cols[0]:
        st.metric(ui["citations_count"], len(evidences))
    with summary_cols[1]:
        model_name = str(last_response.get("model", "-"))
        st.metric(ui["model_label"], model_name)
    with summary_cols[2]:
        st.caption(ui["status_helper"])

    for idx, evidence in enumerate(evidences, start=1):
        source = evidence.get("source") or evidence.get("source_ref") or "unknown"
        page = evidence.get("page")
        snippet = evidence.get("snippet", "")
        score = evidence.get("score")

        meta_parts = [f"{ui['source_label']}: {source}"]
        if page is not None:
            meta_parts.append(f"{ui['page_label']}: {page}")
        if score is not None:
            meta_parts.append(f"{ui['score_label']}: {score}")

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-rank">#{idx}</div>
                <div class="evidence-meta">{' · '.join(meta_parts)}</div>
                <div class="evidence-snippet">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
