import os
from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class VectorDatabaseClient:

    def __init__(
        self,
        pinecone_index_name: str,
        pinecone_namespace: str,
        query_top_k: int,
    ) -> None:
        self._query_top_k = query_top_k
        self._pinecone_namespace = pinecone_namespace
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        self._index: Index = pc.Index(pinecone_index_name)

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        try:
            self._index.upsert(
                vectors=vectors,
                namespace=self._pinecone_namespace,
            )
        except Exception as e:
            print(e)
            raise e

    def query(self, vector: List[float]) -> Dict[str, Any]:
        response = self._index.query(
            vector=vector,
            top_k=self._query_top_k,
            include_metadata=True,
            namespace=self._pinecone_namespace,
        )
        return response
