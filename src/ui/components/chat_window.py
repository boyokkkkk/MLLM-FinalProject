import streamlit as st


def render_chat_history():

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


def add_user_message(content):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


def add_assistant_message(content):

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
        }
    )