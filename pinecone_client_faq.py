import os
from typing import Any, Dict, List
from pinecone import Pinecone  # type: ignore
from openai_client import OpenAiClient
from frequently_asked_questions import FrequentlyAskedQuestions
from pinecone_client import PineconeClient


class PineconeClientFaq(PineconeClient):
    def __init__(
        self, openai_client: OpenAiClient, faq: FrequentlyAskedQuestions
    ) -> None:
        super().__init__(openai_client)
        self._faq = faq

    def _extract_answer_from_pinecone_response(self, response: Dict[str, Any]) -> str:
        return response["matches"][0]["metadata"]["answer"]

    def _get_namespace(self) -> str:
        return "faq"

    def _create_pinecone_upsertable_embedding_vectors(self) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []
        for i, (q, a) in self._faq.enumerate():
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector(question=q),
                    "metadata": {"question": q, "answer": a},
                }
            )
        return upsertable_embedding_vectors


