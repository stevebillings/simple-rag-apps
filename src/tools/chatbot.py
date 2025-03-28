from src.config.config import Config
from src.llm.llm import Llm
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.chat.chat import Chat
from src.tools.tool_setup import ToolSetup


def setup_chat_clients(
    config: Config, tool_setup: ToolSetup
) -> tuple[Llm, PineconeRetriever]:
    openai_client, pinecone_client = tool_setup.setup_base_clients(config)

    pinecone_retriever = PineconeRetriever(
        pinecone_client=pinecone_client,
        pinecone_namespace=config.get_vector_db_namespace(),
    )

    return openai_client, pinecone_retriever


def main() -> None:
    tool_setup = ToolSetup()
    args = tool_setup.parse_common_args("Run the chatbot with a specific configuration")
    config_path = tool_setup.get_config_path(args.config_name, args.config_dir)

    config = Config(config_path)
    openai_client, pinecone_retriever = setup_chat_clients(config, tool_setup)

    chatbot = Chat(
        pinecone_retriever=pinecone_retriever,
        openai_client=openai_client,
        bot_prompt=config.get_bot_prompt(),
    )

    chatbot.chat_loop()


if __name__ == "__main__":
    main()
