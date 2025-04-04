from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.llm.llm import Llm
from src.vector_db.vector_database import VectorDatabase
from src.config.config import CorpusType


class PineconePopulator(ABC):
    def __init__(
        self,
        openai_client: Llm,
        pinecone_client: VectorDatabase,
        namespace: str,
    ) -> None:
        self._pinecone_client: VectorDatabase = pinecone_client
        self._openai_client: Llm = openai_client
        self._namespace = namespace

    @abstractmethod
    def populate_vector_database(self, data: Any) -> None:
        pass

    @abstractmethod
    def _create_pinecone_upsertable_embedding_vectors(
        self, data: Any
    ) -> List[Dict[str, Any]]:
        pass

    @staticmethod
    def create_populator(
        corpus_type: CorpusType,
        openai_client: Llm,
        pinecone_client: VectorDatabase,
        namespace: str,
    ) -> "PineconePopulator":
        if corpus_type == CorpusType.PDFS:
            from src.vector_db.pinecone_populator_chunks import PineconePopulatorChunks

            return PineconePopulatorChunks(
                openai_client=openai_client,
                pinecone_client=pinecone_client,
                namespace=namespace,
            )
        elif corpus_type == CorpusType.JSON_LIST:
            from src.vector_db.pinecone_populator_dict_list import (
                PineconePopulatorDictList,
            )

            return PineconePopulatorDictList(
                openai_client=openai_client,
                pinecone_client=pinecone_client,
                namespace=namespace,
            )
        else:
            from src.vector_db.pinecone_populator_faq import PineconePopulatorFaq

            return PineconePopulatorFaq(
                openai_client=openai_client,
                pinecone_client=pinecone_client,
                namespace=namespace,
            )
