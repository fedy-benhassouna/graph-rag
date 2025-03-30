from typing import List, Dict
import re
from config import LLM
from utils.neo4j_utils import create_neo4j_driver, create_vector_store, delete_database
from utils.visualization import create_3d_graph
from langchain.chains.question_answering import RetrievalQA

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = create_neo4j_driver()
    
    def delete_database(self):
        """Delete all nodes and relationships in the database"""
        delete_database(self.driver)
            
    def create_vector_store(self, documents: List):
        """Create vector store in Neo4j"""
        return create_vector_store(documents)

    def _parse_relationships(self, llm_response: str) -> List[Dict]:
        """Parse LLM relationship extraction response"""
        relationships = []
        pattern = r'\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)'
        
        for line in llm_response.split('\n'):
            line = line.strip()
            matches = re.findall(pattern, line)
            
            for match in matches:
                if len(match) == 3:
                    entity1, relationship, entity2 = match
                    entity1 = entity1.strip()
                    relationship = relationship.strip()
                    entity2 = entity2.strip()
                    
                    if entity1 and relationship and entity2:
                        relationships.append({
                            'entity1': entity1,
                            'relationship': relationship,
                            'entity2': entity2
                        })
        
        return relationships

    def create_knowledge_graph(self, documents: List):
        """Extract entities and relationships to create knowledge graph"""
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text. 
                Format each relationship exactly as: (entity1)-[relationship]->(entity2)
                Return one relationship per line.
                Only include clear, explicit relationships from the text.
                Text: {doc.page_content}
                """
                response = LLM.predict(prompt)
                
                relationships = self._parse_relationships(response)
                for rel in relationships:
                    if all(rel.values()):
                        session.run("""
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                        MERGE (e1)-[:RELATES {type: $relationship}]->(e2)
                        """, rel)

    def create_3d_graph(self, **kwargs):
        """Generate 3D graph visualization"""
        return create_3d_graph(self.driver, **kwargs)

    def query(self, question: str) -> Dict:
        """Query both vector store and knowledge graph"""
        vector_store = create_vector_store([])  # Empty list as we're just querying
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=retriever
        )
        
        rag_answer = qa_chain.run(question)
        
        kg_data = []
        with self.driver.session() as session:
            result = session.run("""
            MATCH path = (e1:Entity)-[r:RELATES]->(e2:Entity)
            WHERE e1.name CONTAINS $question OR e2.name CONTAINS $question
            RETURN e1.name as source, r.type as relationship, e2.name as target
            LIMIT 10
            """, question=question)
            kg_data = [dict(record) for record in result]
        
        return {
            "rag_answer": rag_answer,
            "knowledge_graph": kg_data
        } 