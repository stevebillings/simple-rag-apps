from typing import List
import os

from src.config.config import Config
from src.llm.openai_client import OpenAiClient
from src.corpus.pdf_document import PdfDocumentSet
from src.vector_db.pinecone_populator_faq import PineconePopulatorFaq
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_faq import (
    PineconeQueryResponseParserFaq,
)

# Get the absolute path to the config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(
    current_dir, "..", "..", "resources", "config", "faq_config.json"
)

config: Config = Config(config_path)

openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_query_response_parser: PineconeQueryResponseParser = (
    PineconeQueryResponseParserFaq()
)
pinecone_client: PineconeClient = PineconeClient(
    pinecone_index_name=config.get_vector_db_index_name(),
    pinecone_namespace=config.get_vector_db_namespace(),
    query_response_parser=pinecone_query_response_parser,
)
pinecone_populator = PineconePopulatorFaq(
    openai_client=openai_client,
    pinecone_client=pinecone_client,
    namespace=config.get_vector_db_namespace(),
    faq=config.get_faq(),
)
pinecone_populator.populate_vector_database()
