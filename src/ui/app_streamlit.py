from __future__ import annotations

import base64
import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api/v1")

st.set_page_config(page_title="Multimodal Doc RAG Demo", layout="wide")
st.title("Multimodal Doc RAG - Chat")
st.caption("Chat-style multimodal frontend: text, paste, image upload, and file upload.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 512

with st.sidebar:
    st.subheader("Generation Settings")
    st.session_state.temperature = st.slider(
        "temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1
    )
    st.session_state.max_tokens = st.slider(
        "max_tokens", min_value=64, max_value=2048, value=st.session_state.max_tokens, step=64
    )

    st.subheader("Optional Inputs")
    uploaded_images = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="uploaded_images",
    )
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=["txt", "md", "csv", "json"],
        accept_multiple_files=True,
        key="uploaded_files",
    )
    pasted_text = st.text_area(
        "Paste extra context",
        placeholder="Paste copied text here (optional).",
        height=140,
        key="pasted_text",
    )

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


def _read_text_files(files: list[Any] | None) -> list[str]:
    chunks: list[str] = []
    for file in files or []:
        try:
            raw = file.read()
            text = raw.decode("utf-8", errors="ignore").strip()
            if text:
                chunks.append(f"[file:{file.name}]\n{text}")
        except Exception as exc:
            chunks.append(f"[file:{file.name}] read_failed: {exc}")
    return chunks


def _encode_images(files: list[Any] | None) -> list[str]:
    data_urls: list[str] = []
    for file in files or []:
        raw = file.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = file.type or "image/png"
        data_urls.append(f"data:{mime};base64,{b64}")
    return data_urls


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("images"):
            for img in msg["images"]:
                st.image(img, width=220)
        if msg.get("files"):
            st.caption("Files: " + ", ".join(msg["files"]))

prompt = st.chat_input("Type your question, or paste directly here...")
if prompt:
    file_context = _read_text_files(uploaded_files)
    extra_context: list[str] = []
    if pasted_text and pasted_text.strip():
        extra_context.append(pasted_text.strip())
    extra_context.extend(file_context)

    image_data_urls = _encode_images(uploaded_images)

    user_msg = {
        "role": "user",
        "content": prompt,
        "images": image_data_urls,
        "files": [f.name for f in (uploaded_files or [])],
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.write(prompt)
        if image_data_urls:
            for img in image_data_urls:
                st.image(img, width=220)
        if uploaded_files:
            st.caption("Files: " + ", ".join([f.name for f in uploaded_files]))

    payload = {
        "query": prompt,
        "context": extra_context,
        "image_data_urls": image_data_urls,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=240)
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "")
                st.write(answer)

                citations = data.get("citations", [])
                if citations:
                    st.caption("Citations")
                    for c in citations:
                        st.write(f"- {c['source_ref']}: {c['snippet']}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                err = f"Request failed: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
