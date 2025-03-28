from typing import List
import os
import argparse

from src.config.config import Config
from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.chat.chat import Chat
from src.config.config import CorpusType


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the chatbot with a specific configuration"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="boat_manuals",
        help="Name of the configuration to use (default: boat_manuals)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Override the default config directory path",
    )
    args: argparse.Namespace = parser.parse_args()

    config_name: str = args.config_name
    config_filename: str = f"{config_name}_config.json"
    # Get the absolute path to the config file
    current_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Use provided config directory if specified, otherwise use default path
    if args.config_dir:
        config_path: str = os.path.join(args.config_dir, config_filename)
    else:
        config_path = os.path.join(
            current_dir, "..", "..", "resources", "config", config_filename
        )

    config: Config = Config(config_path)
    openai_client: OpenAiClient = OpenAiClient(
        system_prompt_content_template=config.get_system_prompt_content_template()
    )
    pinecone_query_response_parser: PineconeQueryResponseParser = (
        PineconeQueryResponseParser.create_parser(config.get_corpus_type())
    )
    pinecone_client: PineconeClient = PineconeClient(
        pinecone_index_name=config.get_vector_db_index_name(),
        pinecone_namespace=config.get_vector_db_namespace(),
        query_response_parser=pinecone_query_response_parser,
    )

    pinecone_retriever: PineconeRetriever = PineconeRetriever(
        pinecone_client=pinecone_client,
        pinecone_namespace=config.get_vector_db_namespace(),
    )

    chatbot: Chat = Chat(
        pinecone_retriever=pinecone_retriever,
        openai_client=openai_client,
        bot_prompt=config.get_bot_prompt(),
    )

    chatbot.chat_loop()


if __name__ == "__main__":
    main()
