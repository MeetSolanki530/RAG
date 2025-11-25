import streamlit as st
import requests

SERVER_URL = "http://localhost:8000"

st.title("Client: Conversational RAG With PDF Upload + Chat History")

# Session ID
session_id = st.text_input("Session ID:", "default_session")

# Upload PDF
uploaded_files = st.file_uploader("Upload PDF files", type="pdf")

if uploaded_files:
    st.info("Uploading PDF...")
    files = {"file": uploaded_files}
    res = requests.post(f"{SERVER_URL}/upload", files=files)
    st.success(res.json())

st.write("---")

# Ask questions
user_input = st.text_input("Ask a question:")

if st.button("Send Question"):
    if not user_input:
        st.warning("Please enter a question.")
    else:
        res = requests.post(
            f"{SERVER_URL}/ask",
            data={"question": user_input, "session_id": session_id}
        )
        data = res.json()

        if "answer" in data:
            st.write("### Assistant:")
            st.write(data["answer"])

            st.write("### Chat History:")
            for msg in data["chat_history"]:
                st.write("- ", msg)
        else:
            st.error(data.get("error", "Unknown error"))


