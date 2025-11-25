import streamlit as st
import requests

UPLOAD_URL = "http://localhost:8000/upload"
ASK_URL = "http://localhost:8000/ask"

def upload_pdf(file):
    files = {"file": (file.name, file, "application/pdf")}
    response = requests.post(UPLOAD_URL, files=files)
    return response.json()

def ask_question(question):
    data = {"question": question}
    response = requests.post(ASK_URL, data=data)
    return response.json()


st.title("Simple AI RAG Application")
st.write("Upload a PDF document to create a vector store and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

if uploaded_file is not None:
    with st.spinner("Uploading and processing PDF..."):
        upload_response = upload_pdf(uploaded_file)
        st.success(upload_response.get("message", "Upload completed."))

    question = st.text_input("Ask a question about the document:")

    if st.button("Ask") and question:
        with st.spinner("Getting answer..."):
            answer_response = ask_question(question)
            st.success("Answer:")
            st.write(answer_response.get("answer", "No answer found."))