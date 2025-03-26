import os
import sys
from typing import List

from llm.openai_client import OpenAiClient
from corpus.pdf_document import PdfDocument
from vector_db.pinecone_populator import PineconePopulator
from vector_db.pinecone_client import PineconeClient

pinecone_namespace: str = "boat-manuals"

manual: PdfDocument = PdfDocument(pdf_path="data/Glastron-Owners-Manual-2022.pdf")
chunks: List[str] = manual.extract_chunks()

openai_client: OpenAiClient = OpenAiClient()
pinecone_client: PineconeClient = PineconeClient()
pinecone_populator = PineconePopulator(openai_client=openai_client, pinecone_client=pinecone_client, namespace=pinecone_namespace)
pinecone_populator.populate_vector_database(chunks=chunks)
