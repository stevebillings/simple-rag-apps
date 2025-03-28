
from src.config.config import Config
from src.corpus.pdf_document import PdfDocumentSet
from src.corpus.faq_reader import FaqReader
from src.vector_db.pinecone_populator import PineconePopulator
from src.corpus.text_chunker import TextChunker
from src.corpus.text_cleaner import TextCleaner
from src.corpus.word_validator import WordValidator
from src.config.config import CorpusType
from src.tools.tool_setup import ToolSetup


def setup_populator_clients(config: Config, tool_setup: ToolSetup) -> PineconePopulator:
    openai_client, pinecone_client = tool_setup.setup_base_clients(config)

    return PineconePopulator.create_populator(
        corpus_type=config.get_corpus_type(),
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        namespace=config.get_vector_db_namespace(),
    )


def populate_pdfs(config: Config, pinecone_populator: PineconePopulator) -> None:
    chunker = TextChunker(word_validator=WordValidator())
    manual = PdfDocumentSet(
        text_cleaner=TextCleaner(),
        chunker=chunker,
        pdf_dir_path=config.get_corpus_dir_path(),
    )
    chunks = manual.extract_chunks()
    pinecone_populator.populate_vector_database(chunks)


def populate_faqs(
    config: Config, pinecone_populator: PineconePopulator, workspace_root: str
) -> None:
    faq_reader = FaqReader(
        corpus_dir_path=config.get_corpus_dir_path(),
        workspace_root=workspace_root,
    )
    faq_data = faq_reader.read_faq()
    pinecone_populator.populate_vector_database(faq_data)


def main() -> None:
    tool_setup = ToolSetup()
    args = tool_setup.parse_common_args(
        "Populate Pinecone vector database with either FAQ or boat manual content"
    )
    config_path = tool_setup.get_config_path(args.config_name, args.config_dir)
    workspace_root = tool_setup.get_workspace_root()

    config = Config(config_path)
    pinecone_populator = setup_populator_clients(config, tool_setup)

    if config.get_corpus_type() == CorpusType.PDFS:
        populate_pdfs(config, pinecone_populator)
    else:
        populate_faqs(config, pinecone_populator, workspace_root)


if __name__ == "__main__":
    main()
