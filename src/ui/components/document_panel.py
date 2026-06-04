import streamlit as st

from utils.locales import TEXT


def render_document_panel():

    lang = st.session_state.language
    ui = TEXT[lang]

    st.markdown(
        f"""
        <div class="news-card">
            <h3>{ui["current_document"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    documents = st.session_state.uploaded_documents

    if len(documents) == 0:

        st.info(
            ui["no_document"]
        )

        return

    st.markdown(
        f"### {ui['uploaded_files']}"
    )

    for doc in documents:

        st.markdown(
            f"📄 {doc}"
        )