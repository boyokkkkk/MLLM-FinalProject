import streamlit as st


def render_ticker():

    st.markdown(
        """
        <div class="ticker">
            <div class="ticker-content">

            LATEST DOCUMENTS |
            Operating Systems Review.pdf |
            ANN Lecture 08.pdf |
            Computer Graphics Notes.pdf |
            ROS Tutorial.pdf |
            Multimedia Security Report.pdf

            </div>
        </div>
        """,
        unsafe_allow_html=True
    )