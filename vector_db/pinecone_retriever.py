from vector_db.pinecone_client import PineconeClient
from pinecone import Index
from typing import Any, Dict, List


class PineconeRetriever:
    def __init__(self, pinecone_index: Index, pinecone_namespace: str) -> None:
        self._pinecone_index: Index = pinecone_index
        self._pinecone_namespace = pinecone_namespace

    def retrieve_best_matches(self, query_embedding: List[float]) -> List[str]:
        response: Dict[str, Any] = self._pinecone_index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            namespace=self._pinecone_namespace,
        )
        # TODO: Handle multiple matches
        answer: str = self._extract_answer_from_pinecone_response(response)
        return [answer]

    def _extract_answer_from_pinecone_response(self, response: Dict[str, Any]) -> str:
        return response["matches"][0]["metadata"]["chunk"]
