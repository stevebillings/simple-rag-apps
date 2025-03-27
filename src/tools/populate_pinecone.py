from typing import List, Union
import os
import argparse

from src.config.config import Config
from src.llm.openai_client import OpenAiClient
from src.corpus.pdf_document import PdfDocumentSet
from src.vector_db.pinecone_populator_faq import PineconePopulatorFaq
from src.vector_db.pinecone_populator_chunks import PineconePopulatorChunks
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_faq import (
    PineconeQueryResponseParserFaq,
)
from src.vector_db.pinecone_query_response_parser_chunks import (
    PineconeQueryResponseParserChunks,
)
from src.corpus.text_chunker import TextChunker
from src.corpus.text_cleaner import TextCleaner
from src.corpus.word_validator import WordValidator


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Populate Pinecone vector database with either FAQ or boat manual content"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="boat_manuals",
        help="Name of the configuration to use (default: boat_manuals)",
    )
    args: argparse.Namespace = parser.parse_args()

    config_name: str = args.config
    config_filename: str = f"{config_name}_config.json"
    # Get the absolute path to the config file
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    config_path: str = os.path.join(
        current_dir, "..", "..", "resources", "config", config_filename
    )

    config: Config = Config(config_path)
    openai_client: OpenAiClient = OpenAiClient(
        system_prompt_content_template=config.get_system_prompt_content_template()
    )

    pinecone_query_response_parser: PineconeQueryResponseParser
    pinecone_client: PineconeClient
    pinecone_populator: Union[PineconePopulatorChunks, PineconePopulatorFaq]

    if config.get_corpus_type() == "pdfs":
        pinecone_query_response_parser = PineconeQueryResponseParserChunks()
        pinecone_client = PineconeClient(
            pinecone_index_name=config.get_vector_db_index_name(),
            pinecone_namespace=config.get_vector_db_namespace(),
            query_response_parser=pinecone_query_response_parser,
        )

        chunker: TextChunker = TextChunker(word_validator=WordValidator())
        manual: PdfDocumentSet = PdfDocumentSet(
            text_cleaner=TextCleaner(),
            chunker=chunker,
            pdf_dir_path="resources/boat_manuals",
        )
        chunks: List[str] = manual.extract_chunks()

        pinecone_populator = PineconePopulatorChunks(
            openai_client=openai_client,
            pinecone_client=pinecone_client,
            namespace=config.get_vector_db_namespace(),
        )
        pinecone_populator.populate_vector_database(chunks=chunks)
    else:
        pinecone_query_response_parser = PineconeQueryResponseParserFaq()
        pinecone_client = PineconeClient(
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


if __name__ == "__main__":
    main()
