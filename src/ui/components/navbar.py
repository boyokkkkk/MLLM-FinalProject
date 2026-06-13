from __future__ import annotations

import streamlit as st

from utils.locales import TEXT


def _page_button(label: str, page: str, *, primary: bool = False) -> None:
    if st.button(
        label,
        use_container_width=True,
        type="primary" if primary else "secondary",
    ):
        st.session_state.current_page = page
        st.rerun()


def render_navbar() -> None:
    lang = st.session_state.language
    ui = TEXT[lang]

    chats = list(st.session_state.chat_sessions.keys())
    active_chat = st.session_state.active_chat

    rail_left, rail_right = st.columns([2.2, 1.1], gap="large")

    with rail_left:
        st.markdown(f"<div class='toolbar-label'>{ui['chat_history']}</div>", unsafe_allow_html=True)
        session_cols = st.columns([2.4, 1], gap="small")
        with session_cols[0]:
            selected = st.selectbox(
                ui["chat_history"],
                chats,
                index=chats.index(active_chat),
                label_visibility="collapsed",
            )
            st.session_state.active_chat = selected
        with session_cols[1]:
            if st.button(ui["new_chat"], use_container_width=True):
                name = f"{ui['new_chat']} {len(chats) + 1}"
                st.session_state.chat_sessions[name] = []
                st.session_state.active_chat = name
                st.session_state.retrieved_evidence = []
                st.session_state.last_response = None
                st.rerun()

        nav_cols = st.columns(3, gap="small")
        with nav_cols[0]:
            _page_button(ui["chat"], "Chat", primary=st.session_state.current_page == "Chat")
        with nav_cols[1]:
            _page_button(ui["documents"], "Documents", primary=st.session_state.current_page == "Documents")
        with nav_cols[2]:
            _page_button(ui["benchmark"], "Benchmark", primary=st.session_state.current_page == "Benchmark")

    with rail_right:
        st.markdown(f"<div class='toolbar-label'>{ui['language']}</div>", unsafe_allow_html=True)
        st.session_state.language = st.selectbox(
            ui["language"],
            ["中文", "English"],
            index=0 if st.session_state.language == "中文" else 1,
            label_visibility="collapsed",
        )
        st.markdown(f"<div class='toolbar-label toolbar-spacer'>{ui['style']}</div>", unsafe_allow_html=True)
        st.session_state.answer_style = st.selectbox(
            ui["style"],
            ["Concise", "Detailed", "Academic"],
            index=["Concise", "Detailed", "Academic"].index(st.session_state.answer_style),
            label_visibility="collapsed",
            help=ui["answer_style_helper"],
        )

    st.markdown("<div class='toolbar-divider'></div>", unsafe_allow_html=True)
