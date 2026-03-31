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
        # Lightweight and effective embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def create_index(self, chunks):
        return FAISS.from_documents(chunks, self.embeddings)


class AnalysisAgent:
    """The 'Brain' - Handles intelligent Q&A using Phi-4-mini-instruct (3.8B)."""
    def __init__(self):
        self.model_id = "microsoft/Phi-4-mini-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",          # Automatically uses GPU if available
            trust_remote_code=True
        )
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            trust_remote_code=True
        )

    def synthesize_answer(self, question, vector_db):
        # Retrieval Step
        docs = vector_db.similarity_search(question, k=3)   # Slightly increased for better context
        context = "\n\n".join([d.page_content for d in docs])

        # Better prompt for Phi-4-mini-instruct
        prompt = f"""You are a helpful research assistant. Answer the question concisely and accurately based only on the provided context.

Context:
{context[:1500]}   # Increased context window (Phi-4-mini handles it well)

Question: {question}

Answer:"""

        output = self.gen_pipeline(
            prompt,
            max_new_tokens=200,
            temperature=0.2,      # Lower temperature for more factual answers
            do_sample=True,
            top_p=0.95,
            truncation=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_text = output[0]['generated_text']
        
        # Clean up: remove the prompt part
        answer = full_text.replace(prompt, "").strip()
        return answer


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
    # Save uploaded file temporarily
    temp_path = "current_paper.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vector_db" not in st.session_state:
        with st.spinner("Agent 1 (Extraction) & Agent 2 (Indexing) at work..."):
            chunks = st.session_state.extractor.extract(temp_path)
            st.session_state.vector_db = st.session_state.retriever.create_index(chunks)
            st.success("✅ Indexing Complete. You can now ask questions.")

    query = st.text_input("Ask the Analysis Agent a question about the paper:")

    if query:
        with st.spinner("Agent 3 (Analysis with Phi-4-mini 3.8B) generating response..."):
            answer = st.session_state.analyzer.synthesize_answer(
                query, st.session_state.vector_db
            )
            
            st.markdown("### 🤖 Analysis Output")
            st.info(answer if answer else "The agent could not formulate a confident answer.")

            with st.expander("Show Grounding Context (Retrieved Chunks)"):
                sources = st.session_state.vector_db.similarity_search(query, k=3)
                for i, doc in enumerate(sources):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                    st.divider()
