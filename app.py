import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- 1. THE AGENTIC WORKFLOW COMPONENTS ---

class ResearchAgent:
    def __init__(self):
        # Using distilGPT2 as requested
        self.model_id = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        # Initialize the generation pipeline
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,
            temperature=0.7,
            device=-1 # Set to 0 if you have a GPU
        )
        
        # Sentence Transformers for Vector Search
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def process_pdf(self, file_path):
        """Agent specialized in PDF Extraction and Chunking"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # LangChain Chunking Strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(pages)

    def build_vector_store(self, chunks):
        """Agent specialized in RAG & FAISS indexing"""
        vector_db = FAISS.from_documents(chunks, self.embeddings)
        return vector_db

    def analyze_content(self, question, vector_db):
        """Agent specialized in Research Analysis & Q&A"""
        # Retrieve context from FAISS
        docs = vector_db.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        
        # Format Prompt for distilGPT2
        prompt = f"Answer the question based ONLY on the context below.\nContext: {context}\nQuestion: {question}\nAnswer:"
        
        response = self.llm_pipeline(prompt)
        return response[0]['generated_text'].split("Answer:")[-1].strip()

# --- 2. STREAMLIT INTERFACE ---

st.set_page_config(page_title="AI Research Paper Analyzer", page_icon="📑")
st.title("📑 Research Paper Analyzer Agent")
st.markdown("Analyze PDFs using **distilGPT2**, **FAISS**, and **Agentic RAG**.")

if 'agent' not in st.session_state:
    st.session_state.agent = ResearchAgent()

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    # Save temp file
    with open("temp_paper.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Extracting and Indexing Paper..."):
        chunks = st.session_state.agent.process_pdf("temp_paper.pdf")
        vector_db = st.session_state.agent.build_vector_store(chunks)
        st.success("Paper Indexed in FAISS Vector Store!")

    query = st.text_input("Ask the Research Agent about the paper:")
    
    if query:
        with st.spinner("Analyzing..."):
            answer = st.session_state.agent.analyze_content(query, vector_db)
            st.subheader("Agent's Analysis:")
            st.write(answer)

            with st.expander("View Source Context (FAISS Retrieval)"):
                sources = vector_db.similarity_search(query, k=2)
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**\n{doc.page_content}")
