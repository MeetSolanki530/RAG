from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile, shutil, os

app = FastAPI(title="RAG Chat Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for chat histories per session
chat_store = {}

# Global RAG components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="gemma3:1b")
vectorstore = None
conversational_rag_chain = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]


# ----------------------------------------------------------
# 1. UPLOAD PDF → Create Vectorstore
# ----------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, conversational_rag_chain

    try:
        # save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_pdf_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # Load PDF & split
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(docs)

        # Build embeddings
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Build history-aware retriever
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Rewrite the user's question to be standalone, using chat history context."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        # Build QA chain
        answer_system_prompt = (
            "Use the retrieved context to answer the question. "
            "Be concise. If unknown, say you don't know.\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", answer_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        os.remove(temp_pdf_path)

        return {"message": "PDF uploaded and vectorstore created successfully"}

    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------
# 2. ASK QUESTION → RAG Answer With Chat History
# ----------------------------------------------------------
@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    session_id: str = Form(...)
):
    global conversational_rag_chain

    if conversational_rag_chain is None:
        return {"error": "No PDF uploaded yet."}

    try:
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        return {
            "answer": response["answer"],
            "chat_history": [msg.content for msg in get_session_history(session_id).messages]
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {"message": "RAG Server is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)