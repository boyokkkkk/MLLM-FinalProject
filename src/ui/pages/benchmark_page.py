from __future__ import annotations

import csv
import json
from pathlib import Path

import streamlit as st

from utils.locales import TEXT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_ASSETS_DIR = PROJECT_ROOT / "outputs" / "eval" / "final_assets"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_benchmark_page() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    manifest = _load_json(FINAL_ASSETS_DIR / "asset_manifest.json")
    main_table = _load_csv(FINAL_ASSETS_DIR / "table_closeout_main_results.csv")
    appendix_table = _load_csv(FINAL_ASSETS_DIR / "table_appendix_error_by_type.csv")
    retrieval = ((manifest.get("closeout_mainline") or {}).get("retrieval") or {})
    rag = ((manifest.get("closeout_mainline") or {}).get("rag") or {})
    repair_stats = manifest.get("repair_stats", {})
    current_error_summary = manifest.get("current_error_summary", {})

    st.markdown(
        f"""
        <section class="workspace-intro">
            <div class="section-kicker">{ui["benchmark_title"]}</div>
            <h2 class="workspace-title">{ui["benchmark_title"]}</h2>
            <p class="workspace-subtitle">{ui["benchmark_subtitle"]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(4, gap="small")
    summary_stats = [
        (ui["repair_pipeline"], repair_stats.get("original_total", "-")),
        ("Unique pages", repair_stats.get("repaired_unique", "-")),
        (ui["current_errors"], ((current_error_summary.get("category_counts") or {}).get("generation_issue", "-"))),
        (ui["last_updated"], manifest.get("generated_at", "-")),
    ]
    for col, (label, value) in zip(top_cols, summary_stats):
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

    callout_left, callout_right = st.columns(2, gap="large")
    with callout_left:
        metrics = retrieval.get("metrics", {})
        st.markdown(
            f"""
            <div class="panel-shell">
                <div class="panel-title">{ui["mainline_retrieval"]}</div>
                <div class="metric-callout">
                    <div class="metric-run">{retrieval.get('run', '-')}</div>
                    <div class="metric-line">Hit@5 {metrics.get('hit_at_k', '-')} · Precision@5 {metrics.get('precision_at_k', '-')} · Citation@1 {metrics.get('citation_accuracy', '-')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with callout_right:
        metrics = rag.get("metrics", {})
        st.markdown(
            f"""
            <div class="panel-shell">
                <div class="panel-title">{ui["mainline_rag"]}</div>
                <div class="metric-callout">
                    <div class="metric-run">{rag.get('run', '-')}</div>
                    <div class="metric-line">EM {metrics.get('exact_match', '-')} · ANLS {metrics.get('anls', '-')} · Token-F1 {metrics.get('token_f1', '-')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    overview_tab, figure_tab, table_tab, diag_tab = st.tabs(
        [
            ui["benchmark_overview"],
            ui["benchmark_figures"],
            ui["benchmark_tables"],
            ui["benchmark_diagnostics"],
        ]
    )

    with overview_tab:
        chart_paths = [
            FINAL_ASSETS_DIR / "fig_benchmark_repair.svg",
            FINAL_ASSETS_DIR / "fig_retrieval_progression.svg",
            FINAL_ASSETS_DIR / "fig_rag_closeout_comparison.svg",
            FINAL_ASSETS_DIR / "fig_grounding_vs_quality_tradeoff.svg",
        ]
        for path in chart_paths:
            if path.exists():
                st.image(str(path), use_container_width=True)

    with figure_tab:
        for name in [
            "fig_benchmark_repair.svg",
            "fig_retrieval_progression.svg",
            "fig_rag_closeout_comparison.svg",
            "fig_grounding_vs_quality_tradeoff.svg",
            "fig_appendix_question_type_mix.svg",
            "fig_appendix_error_overview.svg",
            "fig_appendix_error_by_question_type.svg",
        ]:
            path = FINAL_ASSETS_DIR / name
            if path.exists():
                st.image(str(path), use_container_width=True)

    with table_tab:
        if main_table:
            st.dataframe(main_table, use_container_width=True, hide_index=True)
        else:
            st.caption(ui["empty_table"])
        if appendix_table:
            st.markdown("<div class='mini-divider'></div>", unsafe_allow_html=True)
            st.dataframe(appendix_table, use_container_width=True, hide_index=True)

    with diag_tab:
        st.json(current_error_summary)
