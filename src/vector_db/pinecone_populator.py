from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient
from src.config.config import CorpusType


class PineconePopulator(ABC):
    def __init__(
        self,
        openai_client: OpenAiClient,
        pinecone_client: PineconeClient,
        namespace: str,
    ) -> None:
        self._pinecone_client: PineconeClient = pinecone_client
        self._openai_client: OpenAiClient = openai_client
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
        openai_client: OpenAiClient,
        pinecone_client: PineconeClient,
        namespace: str,
    ) -> "PineconePopulator":
        if corpus_type == CorpusType.PDFS:
            from src.vector_db.pinecone_populator_chunks import PineconePopulatorChunks

            return PineconePopulatorChunks(
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
