from neo4j import GraphDatabase
from typing import List, Dict
from langchain.vectorstores import Neo4jVector
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDINGS

def create_neo4j_driver():
    """Create and return a Neo4j driver instance"""
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

def create_vector_store(documents: List):
    """Create vector store in Neo4j"""
    return Neo4jVector.from_documents(
        documents,
        EMBEDDINGS,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="document_vectors",
        node_label="Document",
        embedding_node_property="embedding",
        text_node_property="text"
    )

def delete_database(driver):
    """Delete all nodes and relationships in the database"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        try:
            session.run("CALL db.index.vector.drop('document_vectors')")
        except Exception as e:
            print("Vector index might not exist or was already deleted.") 