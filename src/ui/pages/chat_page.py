import streamlit as st

from services.api_client import APIClient
from components.chat_window import (
    render_chat_history,
    add_user_message,
    add_assistant_message,
)

from utils.locales import TEXT


def build_system_prompt():

    lang = st.session_state.language
    style = st.session_state.answer_style

    prompt = ""

    if lang == "中文":

        prompt += """
Please answer entirely in Simplified Chinese.
Keep technical terms accurate.
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
Use precise terminology.
"""

    return prompt


def render_chat_page():

    lang = st.session_state.language
    ui = TEXT[lang]

    st.title(ui["title"])

    st.caption(ui["subtitle"])

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

                st.markdown(answer)

                citations = result.get(
                    "citations",
                    []
                )

                if citations:

                    st.divider()

                    st.markdown(
                        f"##### {ui['sources']}"
                    )

                    for item in citations:

                        st.info(
                            f"{item['source_ref']}\n\n"
                            f"{item['snippet']}"
                        )

                add_assistant_message(answer)

            except Exception as e:

                error_msg = f"Error:\n\n{str(e)}"

                st.error(error_msg)

                add_assistant_message(error_msg)