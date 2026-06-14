import streamlit as st

from components.markdown import render_markdown_text


def get_current_messages() -> list[dict]:
    return st.session_state.chat_sessions[st.session_state.active_chat]


def render_chat_history() -> None:
    messages = get_current_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            render_markdown_text(msg["content"])
            for image in msg.get("images", []):
                st.image(image, width=220)
            files = msg.get("files", [])
            if files:
                st.caption("Files: " + ", ".join(files))


def add_user_message(content: str, *, images: list[str] | None = None, files: list[str] | None = None) -> None:
    st.session_state.chat_sessions[st.session_state.active_chat].append(
        {
            "role": "user",
            "content": content,
            "images": images or [],
            "files": files or [],
        }
    )


def add_assistant_message(content: str) -> None:
    st.session_state.chat_sessions[st.session_state.active_chat].append(
        {
            "role": "assistant",
            "content": content,
        }
    )


def clear_current_chat() -> None:
    st.session_state.chat_sessions[st.session_state.active_chat] = []
