import os
from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore

from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient


class PineconePopulatorChunks:
    def __init__(
        self,
        openai_client: OpenAiClient,
        pinecone_client: PineconeClient,
        namespace: str,
    ) -> None:
        self._pinecone_client: PineconeClient = pinecone_client
        self._openai_client: OpenAiClient = openai_client
        self._namespace = namespace

    def populate_vector_database(self, chunks: List[str]) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors(chunks=chunks)
        )
        self._pinecone_client.upsert(vectors=vectors)

    def _create_pinecone_upsertable_embedding_vectors(
        self, chunks: List[str]
    ) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []

        for i in range(len(chunks)):
            chunk: str = chunks[i]
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector_for_input(
                        input=chunk
                    ),
                    "metadata": {"chunk": chunk},
                }
            )
        return upsertable_embedding_vectors
