import os
import sys
from typing import List

from src.config.config import Config
from src.config.config_boat_manuals import ConfigBoatManuals
from src.llm.openai_client import OpenAiClient
from src.corpus.pdf_document import PdfDocumentSet
from src.vector_db.pinecone_populator_chunks import PineconePopulatorChunks
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_chunks import (
    PineconeQueryResponseParserChunks,
)
from src.corpus.text_chunker import TextChunker
from src.corpus.text_cleaner import TextCleaner
from src.corpus.word_validator import WordValidator

config: Config = ConfigBoatManuals()

chunker: TextChunker = TextChunker(word_validator=WordValidator())
manual: PdfDocumentSet = PdfDocumentSet(
    text_cleaner=TextCleaner(),
    chunker=chunker, pdf_dir_path="resources/boat_manuals"
)
chunks: List[str] = manual.extract_chunks()

openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_query_response_parser: PineconeQueryResponseParser = (
    PineconeQueryResponseParserChunks()
)
pinecone_client: PineconeClient = PineconeClient(
    pinecone_index_name=config.get_vector_db_index_name(),
    pinecone_namespace=config.get_vector_db_namespace(),
    query_response_parser=pinecone_query_response_parser,
)
pinecone_populator = PineconePopulatorChunks(
    openai_client=openai_client,
    pinecone_client=pinecone_client,
    namespace=config.get_vector_db_namespace(),
)
pinecone_populator.populate_vector_database(chunks=chunks)
