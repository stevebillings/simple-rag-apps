import os
from pinecone import Pinecone, Index  # type: ignore


class PineconeClient:

    def __init__(self, pinecone_index_name: str) -> None:
        self._pinecone_index_name = pinecone_index_name
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self._pc = Pinecone(api_key=pinecone_api_key)

    def connect(self) -> Index:
        return self._pc.Index(self._pinecone_index_name)
