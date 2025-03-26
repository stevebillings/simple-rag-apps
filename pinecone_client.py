import abc
import os
from typing import Any, Dict, List
from pinecone import Pinecone  # type: ignore
from openai_client import OpenAiClient
from frequently_asked_questions import FrequentlyAskedQuestions

class PineconeClient(abc.ABC):
    def __init__(self, openai_client: OpenAiClient) -> None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc_index_name = "faq-database"
        pc = Pinecone(api_key=pinecone_api_key)
        self._index = pc.Index(pc_index_name)
        self._openai_client = openai_client

    def populate_vector_database(self) -> None:
        vectors: List[Dict[str, Any]] = self._create_pinecone_upsertable_embedding_vectors()
        self._upsert(vectors=vectors, namespace=self._get_namespace())

    def _upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        self._index.upsert(vectors=vectors, namespace=namespace)

    def retrieve_best_answer(self, query_embedding, top_k=1) -> str:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._get_namespace(),
        )
        return self._extract_answer_from_pinecone_response(response)
    
    @abc.abstractmethod
    def _extract_answer_from_pinecone_response(self, response: Dict[str, Any]) -> str:
        pass

    @abc.abstractmethod
    def _create_pinecone_upsertable_embedding_vectors(self) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def _get_namespace(self) -> str:
        pass
