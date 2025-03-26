import os
import sys
from typing import List

from tools.config import Config
from tools.config_boat_manuals import ConfigBoatManuals
from llm.openai_client import OpenAiClient
from corpus.pdf_document import PdfDocument
from vector_db.pinecone_populator import PineconePopulator
from vector_db.pinecone_client import PineconeClient

config: Config = ConfigBoatManuals()

manual: PdfDocument = PdfDocument(
    pdf_path="resources/pdfs/Glastron-Owners-Manual-2022.pdf"
)
chunks: List[str] = manual.extract_chunks()

openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_client: PineconeClient = PineconeClient(
    pinecone_index_name=config.get_vector_db_index_name()
)
pinecone_populator = PineconePopulator(
    openai_client=openai_client,
    pinecone_client=pinecone_client,
    namespace=config.get_vector_db_namespace(),
)
pinecone_populator.populate_vector_database(chunks=chunks)
