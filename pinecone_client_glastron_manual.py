import os
from typing import Any, Dict, List
from pinecone import Pinecone  # type: ignore
from openai_client import OpenAiClient
from frequently_asked_questions import FrequentlyAskedQuestions
from pinecone_client import PineconeClient


class PineconeClientGlastronManual(PineconeClient):
    def __init__(self, openai_client: OpenAiClient, chunks: List[str]) -> None:
        super().__init__(openai_client)
        self._chunks = chunks

    def _extract_answer_from_pinecone_response(self, response: Dict[str, Any]) -> str:
        return response["matches"][0]["metadata"]["answer"]

    def _get_namespace(self) -> str:
        return "glastron-manual"

    def _create_pinecone_upsertable_embedding_vectors(self) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []

        for i in range(len(self._chunks)):
            chunk: str = self._chunks[i]
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector(input=chunk),
                    "metadata": {"chunk": chunk},
                }
            )
        return upsertable_embedding_vectors
