import os
from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore

from llm.openai_client import OpenAiClient
from vector_db.pinecone_client import PineconeClient


class PineconePopulator:
    def __init__(
        self, openai_client: OpenAiClient, pinecone_client: PineconeClient, namespace: str
    ) -> None:
        self._index: Index = pinecone_client.connect()
        # TODO there's an asymmetry here
        self._openai_client = openai_client
        self._namespace = namespace
    def populate_vector_database(self, chunks: List[str]) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors(chunks=chunks)
        )
        self._upsert(vectors=vectors, namespace=self._namespace)

    def _upsert(self, vectors: List[Dict[str, Any]], namespace: str) -> None:
        self._index.upsert(vectors=vectors, namespace=namespace)

    def _create_pinecone_upsertable_embedding_vectors(
        self, chunks: List[str]
    ) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []

        for i in range(len(chunks)):
            chunk: str = chunks[i]
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector(input=chunk),
                    "metadata": {"chunk": chunk},
                }
            )
        return upsertable_embedding_vectors

