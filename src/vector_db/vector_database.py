import os
from typing import Any, Dict, List
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.vector_database_client import VectorDatabaseClient
from src.vector_db.dto.scored_match import ScoredMatch


class VectorDatabase:

    def __init__(
        self,
        vector_database_client: VectorDatabaseClient,
        query_response_parser: PineconeQueryResponseParser,
        batch_size: int = 100,
    ) -> None:
        self._vector_database_client = vector_database_client
        self._query_response_parser = query_response_parser
        self._batch_size = batch_size

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        for i in range(0, len(vectors), self._batch_size):
            batch = vectors[i : i + self._batch_size]
            self._vector_database_client.upsert_vectors(
                vectors=batch,
            )

    def query(self, query_embedding: List[float]) -> List[ScoredMatch]:
        response = self._vector_database_client.query(
            vector=query_embedding,
        )
        relevant_content: List[ScoredMatch] = (
            self._query_response_parser.parse_relevant_content_from_query_response(
                response
            )
        )
        return relevant_content
