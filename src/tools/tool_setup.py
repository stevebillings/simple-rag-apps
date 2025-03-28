import os
import argparse
from typing import Optional

from src.config.config import Config
from src.llm.llm import Llm
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.llm.llm_client import LlmClient


class ToolSetup:
    def __init__(self) -> None:
        self._current_dir: str = os.path.dirname(os.path.abspath(__file__))

    def get_config_path(self, config_name: str, config_dir: Optional[str]) -> str:
        config_filename = f"{config_name}_config.json"

        if config_dir:
            return os.path.join(config_dir, config_filename)

        return os.path.join(
            self._current_dir, "..", "..", "resources", "config", config_filename
        )

    def get_workspace_root(self) -> str:
        return os.path.dirname(os.path.dirname(self._current_dir))

    def setup_base_clients(self, config: Config) -> tuple[Llm, PineconeClient]:
        llm_client = LlmClient()
        llm = Llm(
            system_prompt_content_template=config.get_system_prompt_content_template(),
            llm_client=llm_client,
        )

        pinecone_query_response_parser = PineconeQueryResponseParser.create_parser(
            config.get_corpus_type()
        )

        pinecone_client = PineconeClient(
            pinecone_index_name=config.get_vector_db_index_name(),
            pinecone_namespace=config.get_vector_db_namespace(),
            query_response_parser=pinecone_query_response_parser,
        )

        return llm, pinecone_client

    def parse_common_args(self, description: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=description)
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
        return parser.parse_args()
