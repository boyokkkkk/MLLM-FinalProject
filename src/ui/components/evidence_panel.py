import streamlit as st

from utils.locales import TEXT


def render_evidence_panel():

    lang = st.session_state.language
    ui = TEXT[lang]

    st.divider()

    st.markdown(
        f"""
        <h3>
        {ui["retrieved_evidence"]}
        </h3>
        """,
        unsafe_allow_html=True
    )

    evidences = st.session_state.retrieved_evidence

    if len(evidences) == 0:

        if lang == "中文":

            st.caption(
                "暂无检索证据"
            )

        else:

            st.caption(
                "No retrieved evidence"
            )

        return

    for evidence in evidences:

        st.info(evidence)