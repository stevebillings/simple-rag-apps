import os
from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore

from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient


class PineconePopulatorFaq:
    def __init__(
        self,
        openai_client: OpenAiClient,
        pinecone_client: PineconeClient,
        namespace: str,
        faq: Dict[str, str],
    ) -> None:
        self._pinecone_client: PineconeClient = pinecone_client
        self._openai_client: OpenAiClient = openai_client
        self._namespace = namespace
        self._faq = faq

    def populate_vector_database(self) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors()
        )
        self._pinecone_client.upsert(vectors=vectors)

    def _create_pinecone_upsertable_embedding_vectors(self) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []
        for i, (q, a) in enumerate(self._faq.items()):
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector_for_input(
                        input=q
                    ),
                    "metadata": {"question": q, "answer": a},
                }
            )
        return upsertable_embedding_vectors
