import streamlit as st
import os
import tempfile
import hashlib

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Chat with PDFs",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with Multiple PDFs")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

# --------------------------------------------------
# Session State Init
# --------------------------------------------------
for key in ["messages", "retriever", "qa_chain", "file_hash"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑️ Clear Data (PDFs + Chat)"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.qa_chain = None
        st.session_state.file_hash = None
        st.session_state.uploaded_files = None  # optional
        st.success("✅ All data cleared. Upload new PDFs to start fresh!")

# --------------------------------------------------
# Utility: Hash uploaded files
# --------------------------------------------------
def hash_files(files):
    hasher = hashlib.md5()
    for f in files:
        hasher.update(f.name.encode())
        hasher.update(f.getbuffer())
    return hasher.hexdigest()

# --------------------------------------------------
# Process PDFs (NO CACHE!)
# --------------------------------------------------
def process_pdfs(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    splits = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# --------------------------------------------------
# Build RAG Chain
# --------------------------------------------------
def build_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. "
            "Use ONLY the provided context. "
            "If the answer is not in the context, say 'I don't know'."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# --------------------------------------------------
# Detect PDF changes & rebuild index
# --------------------------------------------------
if uploaded_files:
    new_hash = hash_files(uploaded_files)

    if new_hash != st.session_state.file_hash:
        with st.spinner("🔍 Indexing PDFs..."):
            retriever = process_pdfs(uploaded_files)
            qa_chain = build_chain(retriever)

            st.session_state.retriever = retriever
            st.session_state.qa_chain = qa_chain
            st.session_state.file_hash = new_hash
            st.session_state.messages = []

        st.success("✅ PDFs indexed successfully")

# --------------------------------------------------
# Chat History Display
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# Chat Input
# --------------------------------------------------
if st.session_state.qa_chain:
    query = st.chat_input("Ask a question about the PDFs...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_text = ""

            chat_history = st.session_state.messages[-6:]
            docs = st.session_state.retriever.invoke(query)

            result = st.session_state.qa_chain.invoke({
                "question": query,
                "chat_history": chat_history
            })

            for token in result.split():
                answer_text += token + " "
                placeholder.markdown(answer_text)

            sources = set(
                f"{d.metadata['source']} (page {d.metadata.get('page', '?')})"
                for d in docs
            )

            answer_text += "\n\n📌 **Sources:**\n" + "\n".join(
                f"- {s}" for s in sources
            )

            placeholder.markdown(answer_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text
        })
