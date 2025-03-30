import streamlit as st
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile
from models.knowledge_graph import KnowledgeGraphRAG
from styles.custom_css import CUSTOM_CSS

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Knowledge Graph RAG System")

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def process_pdf(pdf_file) -> List:
    """Process uploaded PDF file and return documents"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    documents = text_splitter.split_documents(pages)
    os.unlink(tmp_file_path)
    return documents

def main():
    st.title("Knowledge Graph RAG System")
    
    # Initialize KnowledgeGraphRAG
    kg_rag = KnowledgeGraphRAG()
    
    # File upload section
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            documents = process_pdf(uploaded_file)
            
            # Create vector store and knowledge graph
            kg_rag.create_vector_store(documents)
            kg_rag.create_knowledge_graph(documents)
            
            st.success("Document processed successfully!")
    
    # Query section
    st.header("Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Processing your question..."):
            result = kg_rag.query(question)
            
            # Display RAG answer
            st.subheader("Answer:")
            st.write(result["rag_answer"])
            
            # Display knowledge graph data
            if result["knowledge_graph"]:
                st.subheader("Related Knowledge Graph Data:")
                for item in result["knowledge_graph"]:
                    st.write(f"{item['source']} - {item['relationship']} -> {item['target']}")
    
    # Graph visualization section
    st.header("Knowledge Graph Visualization")
    
    # Graph customization options
    col1, col2, col3 = st.columns(3)
    with col1:
        node_size = st.slider("Node Size", 1, 20, 8)
    with col2:
        node_color = st.color_picker("Node Color", "#00ff00")
    with col3:
        edge_color = st.color_picker("Edge Color", "#888")
    
    # Display graph
    fig = kg_rag.create_3d_graph(
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color
    )
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No graph data available. Please upload a document first.")
    
    # Database management section
    st.header("Database Management")
    if st.button("Clear Database"):
        kg_rag.delete_database()
        st.success("Database cleared successfully!")

if __name__ == "__main__":
    main() 