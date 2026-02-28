import streamlit as st
import torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- 1. THE SPECIALIZED AGENT ARCHITECTURE ---

class PDFExtractionAgent:
    """Specialized in transforming raw PDF bytes into searchable atomic units."""
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def extract(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return self.splitter.split_documents(pages)

class RetrievalAgent:
    """Specialized in FAISS indexing and semantic similarity search."""
    def __init__(self):
        # Industrial standard for lightweight local embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_index(self, chunks):
        return FAISS.from_documents(chunks, self.embeddings)

class AnalysisAgent:
    """The 'Brain' - Handles intelligent Q&A using distilGPT2."""
    def __init__(self):
        self.model_id = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Ensure padding token is set for batch processing if needed
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def synthesize_answer(self, question, vector_db):
        # Retrieval Step
        docs = vector_db.similarity_search(question, k=2) # k=2 to save distilGPT2 context space
        context = " ".join([d.page_content for d in docs])
        
        # Crafting a tight prompt for a small model
        prompt = (
            f"Context: {context[:600]}\n" # Hard truncate to prevent overflow
            f"Question: {question}\n"
            f"Answer the question concisely based on the context above:\n"
        )
        
        output = self.gen_pipeline(
            prompt, 
            max_new_tokens=100, 
            temperature=0.3, # Low temp for factual consistency
            truncation=True
        )
        
        # Parsing logic to extract only the generated text
        full_text = output[0]['generated_text']
        return full_text.replace(prompt, "").strip()

# --- 2. STREAMLIT ORCHESTRATION ---

st.set_page_config(page_title="Agentic Research Analyzer", layout="wide")
st.title("📑 Research Paper Analyzer: Agentic Workflow")

# Initialize Agents in Session State
if "extractor" not in st.session_state:
    st.session_state.extractor = PDFExtractionAgent()
    st.session_state.retriever = RetrievalAgent()
    st.session_state.analyzer = AnalysisAgent()

uploaded_file = st.file_uploader("Upload Research PDF", type="pdf")

if uploaded_file:
    # Save to temp location for PyPDFLoader
    temp_path = "current_paper.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vector_db" not in st.session_state:
        with st.spinner("Agent 1 (Extraction) & Agent 2 (Indexing) at work..."):
            chunks = st.session_state.extractor.extract(temp_path)
            st.session_state.vector_db = st.session_state.retriever.create_index(chunks)
            st.success("Indexing Complete.")

    query = st.text_input("Ask the Analysis Agent a question:")
    
    if query:
        with st.spinner("Agent 3 (Analysis) generating response..."):
            answer = st.session_state.analyzer.synthesize_answer(
                query, st.session_state.vector_db
            )
            
            st.markdown("### 🤖 Analysis Output")
            st.info(answer if answer else "The agent could not formulate a confident answer.")

            with st.expander("Show Grounding Context"):
                sources = st.session_state.vector_db.similarity_search(query, k=2)
                for i, doc in enumerate(sources):
                    st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
