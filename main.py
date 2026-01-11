import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with your PDF")

# --------------------------------------------------
# Load API Key (NO UI INPUT)
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("---")
    st.markdown("**Model:** Llama-3.1-8B (Groq)")
    st.markdown("**Embeddings:** MiniLM")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --------------------------------------------------
# Document Processing
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

# --------------------------------------------------
# Build QA Chain
# --------------------------------------------------
def build_qa_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# --------------------------------------------------
# Load PDF
# --------------------------------------------------
if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("🔍 Indexing document..."):
        retriever = process_pdf(uploaded_file)
        st.session_state.qa_chain = build_qa_chain(retriever)
    st.success("✅ Document ready! Ask questions below.")

# --------------------------------------------------
# Chat UI
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.qa_chain:
    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(query)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
