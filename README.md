# 📄 Chat with Multiple PDFs

A **Streamlit app** that lets you upload multiple PDFs and chat with them using a **Groq LLM**, **HuggingFace embeddings**, and **FAISS vector database**.  

This project is designed for **portfolios or resumes** to showcase real-world document retrieval, conversational AI, and interactive data applications.

---

## 🚀 Features

- **Multi-PDF Support** – Upload and query multiple PDFs at once.  
- **Robust PDF Parsing** – Works with standard and scanned PDFs using `UnstructuredPDFLoader`.  
- **Vector Search** – Embeds text with HuggingFace embeddings and stores in FAISS for fast retrieval.  
- **Conversational Memory** – Maintains last 3 user-assistant turns for context-aware answers.  
- **Sources Displayed** – Shows file names and page numbers for retrieved answers.  
- **Streaming-style Answers** – Answers appear progressively for an interactive experience.  
- **Clear Data Button** – Reset chat history and vector database to start fresh.  

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)  
- **Embeddings:** [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **LLM:** [Groq ChatGroq `llama-3.1-8b-instant`](https://groq.com/)  
- **PDF Parsing:** [LangChain UnstructuredPDFLoader](https://www.langchain.com/)  

---

## 📂 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/chat-with-pdfs.git
cd chat-with-pdfs
```

2. **Create virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set your Groq API Key**
Create a .env file in the root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the app:

```bash
streamlit run app.py
```

---

Upload one or more PDFs using the sidebar.

Ask questions in the chat input.

View sources and answers streaming progressively.

Use “Clear Data” to reset chat and upload new PDFs.


**💡 Notes**

1.Works with scanned PDFs and complex fonts.

2.Maintains last 3 conversation turns to provide context-aware responses.

3.Sources are displayed for each answer to increase transparency.

---

**📈 Future Enhancements**

1.True token-by-token streaming from Groq for live answers.

2.Persistent FAISS database to avoid re-indexing PDFs.

3.Reranking retrieved documents for higher answer accuracy.

Multi-modal support for images or Excel files.

**📝 License**

MIT License – see LICENSE
 for details.

**👨‍💻 Author**

Prahas Hegde – AI developer
[LinkedIn](https://www.linkedin.com/in/prahaas-baburai-hegdae-4a2464255/) | [GitHub](https://github.com/PrahasHegde)
