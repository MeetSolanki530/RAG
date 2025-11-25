from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI,UploadFile,File, Form
from fastapi.middleware.cors import CORSMiddleware
import os,shutil,tempfile


app = FastAPI(
    title="SIMPLE RAG WITH OLLAMA MODELS",
    description="A simple RAG application using Ollama models and LangChain",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORSTORE_PATH = "faiss_index"


def process_input_to_vectorstore(input_file_path):
    
    input_loaded_data = PyPDFLoader(file_path=input_file_path).load()
    
    splitted_data = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 200).split_documents(input_loaded_data)
    
    vectorstore = FAISS.from_documents(documents=splitted_data,embedding=OllamaEmbeddings(model="nomic-embed-text"))
    
    vectorstore.save_local(folder_path=VECTORSTORE_PATH)
    
    return True

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    
    try:
        #Save Temporary
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
            temp_pdf_path = tmp.name
            shutil.copyfileobj(file.file,tmp)

        process_input_to_vectorstore(temp_pdf_path)

        os.remove(temp_pdf_path)

        return {"message" : "PDF Upload and Vectorstore created Sucessfully."}
    
    except Exception as e:

        print(e)

        return {"message" : "Error Occured"}


@app.post("/ask")
async def ask_question(question:str = Form(...)):
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vectorstore = FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever()
            
        retrieved_docs = retriever.invoke(question)

        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        llm = ChatOllama(model="gemma3:1b")

        prompt_template = ChatPromptTemplate.from_template("""
    Respond User Question Based On Provided Document Context.

    ---
    Context: {context}
    User Question: {question}
    ---

    """)
        output_parser = StrOutputParser()

        chain = prompt_template | llm | output_parser

        response = chain.invoke({"context": context_text, "question": question})

        return {"answer" : response}
    
    except Exception as e:
        print(e)
        return {"answer" : "Sorry Something Was Wrong!!! Please Try again later."}


@app.get("/")
def home():
    return {"message" : "RAG FastAPI Server is Running!"}

if __name__  == "__main__":
    import uvicorn
    uvicorn.run(app=app,port=8000)