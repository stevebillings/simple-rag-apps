
import os
from typing import Any, Dict, List
from pinecone import Pinecone  # type: ignore
from openai_client import OpenAiClient
from frequently_asked_questions import FrequentlyAskedQuestions

class PineconeClient:
    def __init__(self, openai_client: OpenAiClient, faq: FrequentlyAskedQuestions) -> None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc_index_name = "faq-database"
        pc = Pinecone(api_key=pinecone_api_key)
        self._index = pc.Index(pc_index_name)
        self._openai_client = openai_client
        self._faq = faq

    def populate_vector_database_from_faq(self) -> None:
        vectors: List[Dict[str, Any]] = self._create_pinecone_upsertable_embedding_vectors()
        self._upsert(vectors=vectors, namespace='ns1')

    def retrieve_best_faq_answer(self, query_embedding, top_k=1) -> str:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="ns1",
        )
        return response['matches'][0]['metadata']['answer']
    
    def _create_pinecone_upsertable_embedding_vectors(self) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []
        for i, (q, a) in self._faq.enumerate():
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector(question=q),
                    "metadata": {"question": q, "answer": a},
                }
            )
        return upsertable_embedding_vectors
    
    def _upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        self._index.upsert(vectors=vectors, namespace=namespace)
