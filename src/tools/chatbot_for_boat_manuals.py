from typing import List
import os

from src.config.config import Config
from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_chunks import (
    PineconeQueryResponseParserChunks,
)
from src.chatbot.chatbot import Chatbot

# Get the absolute path to the config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(
    current_dir, "..", "..", "resources", "config", "boat_manuals_config.json"
)

config: Config = Config(config_path)
openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_query_response_parser: PineconeQueryResponseParser = (
    PineconeQueryResponseParserChunks()
)
pinecone_client = PineconeClient(
    pinecone_index_name=config.get_vector_db_index_name(),
    pinecone_namespace=config.get_vector_db_namespace(),
    query_response_parser=pinecone_query_response_parser,
)

pinecone_retriever = PineconeRetriever(
    pinecone_client=pinecone_client, pinecone_namespace=config.get_vector_db_namespace()
)

chatbot = Chatbot(
    pinecone_retriever=pinecone_retriever,
    openai_client=openai_client,
    bot_prompt=config.get_bot_prompt(),
)

chatbot.chat_loop()
