import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# Embeddings Configuration
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Graph Visualization Settings
DEFAULT_NODE_SIZE = 8
DEFAULT_NODE_COLOR = '#00ff00'
DEFAULT_EDGE_WIDTH = 1
DEFAULT_EDGE_COLOR = '#888'
DEFAULT_SPRING_K = 0.1 