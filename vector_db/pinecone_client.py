import os
from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore
from vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeClient:

    def __init__(
        self,
        pinecone_index_name: str,
        pinecone_namespace: str,
        query_response_parser: PineconeQueryResponseParser,
    ) -> None:
        self._pinecone_namespace = pinecone_namespace
        self._query_response_parser = query_response_parser
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        self._index: Index = pc.Index(pinecone_index_name)

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        self._index.upsert(
            vectors=vectors,
            namespace=self._pinecone_namespace,
        )

    def query(self, query_embedding: List[float], top_k: int = 1) -> List[str]:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self._pinecone_namespace,
        )
        answer: str = self._query_response_parser.parse_answer_from_query_response(
            response
        )
        return [answer]
