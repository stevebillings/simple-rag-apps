from typing import Any, Dict, List
from pinecone import Pinecone, Index  # type: ignore

from src.llm.llm import Llm
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_populator import PineconePopulator


class PineconePopulatorFaq(PineconePopulator):
    def __init__(
        self,
        openai_client: Llm,
        pinecone_client: PineconeClient,
        namespace: str,
    ) -> None:
        super().__init__(openai_client, pinecone_client, namespace)

    def populate_vector_database(self, data: Dict[str, str]) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors(data)
        )
        self._pinecone_client.upsert(vectors=vectors)

    def _create_pinecone_upsertable_embedding_vectors(
        self, data: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []
        for i, (q, a) in enumerate(data.items()):
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
