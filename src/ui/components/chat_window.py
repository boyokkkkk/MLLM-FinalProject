import streamlit as st


def get_current_messages():

    return st.session_state.chat_sessions[
        st.session_state.active_chat
    ]


def render_chat_history():

    messages = get_current_messages()

    for msg in messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


def add_user_message(content):

    st.session_state.chat_sessions[
        st.session_state.active_chat
    ].append(
        {
            "role": "user",
            "content": content
        }
    )


def add_assistant_message(content):

    st.session_state.chat_sessions[
        st.session_state.active_chat
    ].append(
        {
            "role": "assistant",
            "content": content
        }
    )


def clear_current_chat():

    st.session_state.chat_sessions[
        st.session_state.active_chat
    ] = []