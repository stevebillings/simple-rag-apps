from typing import List

from src.config.config import Config
from src.config.config_faq import ConfigFaq
from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_client import PineconeClient
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.vector_db.pinecone_query_response_parser_faq import PineconeQueryResponseParserFaq


def rag_chatbot(
    pinecone_retriever: PineconeRetriever,
    user_question: str,
    openai_client: OpenAiClient,
) -> str:
    user_question_embedding: List[float] = (
        openai_client.create_embedding_vector_for_input(input=user_question)
    )
    best_matches: List[str] = pinecone_retriever.retrieve_best_matches(
        query_embedding=user_question_embedding,
    )
    best_match: str = best_matches[0]

    return openai_client.ask_llm_with_context(
        context=best_match, user_question=user_question
    )


#################
# main()
#################
config: Config = ConfigFaq()
openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_query_response_parser: PineconeQueryResponseParser = (
    PineconeQueryResponseParserFaq()
)
pinecone_client = PineconeClient(
    pinecone_index_name=config.get_vector_db_index_name(),
    pinecone_namespace=config.get_vector_db_namespace(),
    query_response_parser=pinecone_query_response_parser,
)

pinecone_retriever = PineconeRetriever(
    pinecone_client=pinecone_client, pinecone_namespace=config.get_vector_db_namespace()
)

print("Enter your question (or type 'exit' to quit):")
while True:
    try:
        user_question = input("> ").strip()
        if user_question.lower() == "exit":
            break
        if not user_question:
            continue

        resp_msg: str = rag_chatbot(
            user_question=user_question,
            pinecone_retriever=pinecone_retriever,
            openai_client=openai_client,
        )
        print(f">> {resp_msg}\n\n")
    except EOFError:  # Handle Control-D
        break
    except KeyboardInterrupt:  # Handle Control-C
        break

print("\nGoodbye!")
