from typing import List

from src.config.config import Config
from src.config.config_boat_manuals import ConfigBoatManuals
from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_chunks import (
    PineconeQueryResponseParserChunks,
)
from src.chatbot.chatbot import Chatbot


#################
# main()
#################
config: Config = ConfigBoatManuals()
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
