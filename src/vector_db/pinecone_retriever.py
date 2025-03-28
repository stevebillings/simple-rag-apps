from typing import Any, Dict, List
from src.vector_db.vector_database import VectorDatabase


class PineconeRetriever:
    def __init__(
        self, pinecone_client: VectorDatabase, pinecone_namespace: str
    ) -> None:
        self._pinecone_client: VectorDatabase = pinecone_client
        self._pinecone_namespace = pinecone_namespace

    def retrieve_best_matches(self, query_embedding: List[float]) -> List[str]:
        return self._pinecone_client.query(
            query_embedding=query_embedding,
        )

    def _extract_answer_from_pinecone_response(self, response: Dict[str, Any]) -> str:
        return response["matches"][0]["metadata"]["chunk"]
