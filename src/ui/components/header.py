from __future__ import annotations

import streamlit as st

from services.api_client import APIClient
from utils.locales import TEXT


def render_header() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    try:
        health = APIClient.health()
        st.session_state.last_backend_health = health
    except Exception:
        health = st.session_state.get("last_backend_health")

    online = isinstance(health, dict)
    version = health.get("version", "-") if online else "-"
    last_response = st.session_state.get("last_response") or {}
    latest_model = last_response.get("model", "-")
    active_scope = st.session_state.get("corpus_chip", ui["chat_mode_all"])
    status_text = ui["service_online"] if online else ui["service_offline"]
    status_class = "status-online" if online else "status-offline"

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-copy">
                <div class="section-kicker">{ui["workspace"]}</div>
                <h1 class="hero-title">{ui["title"]}</h1>
                <p class="hero-subtitle">{ui["workspace_subtitle"]}</p>
            </div>
            <div class="hero-metrics">
                <div class="hero-status-card">
                    <div class="hero-status-head">
                        <span class="hero-chip {status_class}">{status_text}</span>
                        <span class="hero-chip muted-chip">{ui["service_version"]}: v{version}</span>
                    </div>
                    <div class="hero-grid">
                        <div class="hero-grid-item">
                            <div class="hero-grid-label">{ui["active_scope"]}</div>
                            <div class="hero-grid-value">{active_scope}</div>
                        </div>
                        <div class="hero-grid-item">
                            <div class="hero-grid-label">{ui["latest_model"]}</div>
                            <div class="hero-grid-value">{latest_model}</div>
                        </div>
                        <div class="hero-grid-item wide">
                            <div class="hero-grid-label">{ui["api_base"]}</div>
                            <div class="hero-grid-value mono">{APIClient.get_api_base()}</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
