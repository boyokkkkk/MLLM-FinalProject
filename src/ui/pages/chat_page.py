import streamlit as st

from services.api_client import APIClient

from components.chat_window import (
    render_chat_history,
    add_user_message,
    add_assistant_message,
    clear_current_chat,
)

from utils.locales import TEXT


def build_system_prompt():

    lang = st.session_state.language
    style = st.session_state.answer_style

    prompt = ""

    if lang == "中文":

        prompt += """
Please answer entirely in Simplified Chinese.
"""

    else:

        prompt += """
Please answer entirely in English.
"""

    if style == "Concise":

        prompt += """
Provide short and concise answers.
"""

    elif style == "Detailed":

        prompt += """
Provide detailed explanations.
"""

    elif style == "Academic":

        prompt += """
Provide academic and formal explanations.
"""

    return prompt


def render_chat_page():

    lang = st.session_state.language
    ui = TEXT[lang]

    # ===== 顶部会话栏 =====

    col1, col2 = st.columns([8, 1])

    with col1:

        st.subheader(
            st.session_state.active_chat
        )

    with col2:

        if st.button(
            "🗑",
            help=ui["clear_chat"]
        ):

            clear_current_chat()

            st.rerun()

    st.divider()

    render_chat_history()

    prompt = st.chat_input(
        ui["placeholder"]
    )

    if not prompt:

        return

    add_user_message(prompt)

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner(ui["thinking"]):

            try:

                system_prompt = build_system_prompt()

                enhanced_query = f"""
{system_prompt}

User Question:
{prompt}
"""

                result = APIClient.chat(
                    query=enhanced_query,
                    context=[],
                    images=[],
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )

                answer = result.get(
                    "answer",
                    "No answer returned."
                )

            except Exception:

                if lang == "中文":

                    answer = """
当前处于 Demo 模式。

后端接口尚未接入。
"""

                else:

                    answer = """
Demo Mode.

Backend API not connected yet.
"""

            st.markdown(answer)

            add_assistant_message(answer)