from typing import Any, Dict, List
import json

from src.llm.llm import Llm
from src.vector_db.vector_database import VectorDatabase
from src.vector_db.pinecone_populator import PineconePopulator


class PineconePopulatorDictList(PineconePopulator):
    def __init__(
        self,
        openai_client: Llm,
        pinecone_client: VectorDatabase,
        namespace: str,
    ) -> None:
        super().__init__(openai_client, pinecone_client, namespace)

    def populate_vector_database(self, data: List[Dict[str, str]]) -> None:
        vectors: List[Dict[str, Any]] = (
            self._create_pinecone_upsertable_embedding_vectors(data)
        )
        self._pinecone_client.upsert(vectors=vectors)

    def _create_pinecone_upsertable_embedding_vectors(
        self, data: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        upsertable_embedding_vectors: List[Dict[str, Any]] = []
        for i, d in enumerate(data):
            api_parameters_value: str = json.dumps(d["api_parameters"])
            d["api_parameters"] = api_parameters_value
            upsertable_embedding_vectors.append(
                {
                    "id": str(i),
                    "values": self._openai_client.create_embedding_vector_for_input(
                        input=json.dumps(d)
                    ),
                    "metadata": d,
                }
            )
        return upsertable_embedding_vectors
