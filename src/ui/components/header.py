import streamlit as st
from datetime import datetime


def render_header():

    today = datetime.now().strftime("%B %d, %Y")

    st.markdown(
        f"""
        <div style="
            border-top:4px solid black;
            border-bottom:4px solid black;
            padding:20px;
            margin-bottom:20px;
        ">

        <h1 class="news-title"
        style="
        font-size:56px;
        margin:0;
        ">
        MULTIMODAL COURSE ASSISTANT
        </h1>

        <div class="news-subtitle">
        VOL.01 | SYSU EDITION | {today}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )