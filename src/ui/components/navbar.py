import streamlit as st

from utils.locales import TEXT


def render_navbar():

    lang = st.session_state.language
    ui = TEXT[lang]

    col1,col2,col3,col4 = st.columns(
        [2,1,1,1]
    )

    with col1:

        chats = list(
            st.session_state.chat_sessions.keys()
        )

        active = st.selectbox(
            ui["chat_history"],
            chats,
            index=chats.index(
                st.session_state.active_chat
            )
        )

        st.session_state.active_chat = active

        if st.button(
            ui["new_chat"]
        ):

            name = f"{ui['new_chat']} {len(chats)+1}"

            st.session_state.chat_sessions[
                name
            ] = []

            st.session_state.active_chat = name

            st.rerun()

    with col2:

        if st.button(
            ui["documents"],
            use_container_width=True
        ):

            st.session_state.current_page="Documents"

            st.rerun()

    with col3:

        if st.button(
            ui["benchmark"],
            use_container_width=True
        ):

            st.session_state.current_page="Benchmark"

            st.rerun()

    with col4:

        st.session_state.language = st.selectbox(
            "",
            ["English","中文"]
        )

    st.divider()