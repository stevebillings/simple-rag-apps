import os
from pinecone import Pinecone, Index  # type: ignore


class PineconeClient:
    def __init__(self) -> None:
        pass

    def connect(self) -> Index:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc_index_name = "faq-database"
        pc = Pinecone(api_key=pinecone_api_key)
        return pc.Index(pc_index_name)
