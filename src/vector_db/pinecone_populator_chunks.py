from typing import Any, Dict, List

from src.llm.llm import Llm
from src.vector_db.vector_database import VectorDatabase
from src.vector_db.pinecone_populator import PineconePopulator


class PineconePopulatorChunks(PineconePopulator):
    def __init__(
        self,
        openai_client: Llm,
        pinecone_client: VectorDatabase,
        namespace: str,
    ) -> None:
        super().__init__(openai_client, pinecone_client, namespace)

    def populate_vector_database(self, data: List[str]) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors(data)
        )
        self._pinecone_client.upsert(vectors=vectors)

    def _create_pinecone_upsertable_embedding_vectors(
        self, data: List[str]
    ) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []

        for i in range(len(data)):
            chunk: str = data[i]
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
