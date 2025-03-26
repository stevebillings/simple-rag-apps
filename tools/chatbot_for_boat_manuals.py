import os
import sys
from typing import Any, Dict, List, Optional
from pinecone import Index  # type: ignore

from tools.config import Config
from tools.config_boat_manuals import ConfigBoatManuals
from llm.openai_client import OpenAiClient
from vector_db.pinecone_client import PineconeClient
from vector_db.pinecone_retriever import PineconeRetriever
from vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from vector_db.pinecone_query_response_parser_boat_manuals import PineconeQueryResponseParserBoatManuals

def rag_chatbot(
    pinecone_retriever: PineconeRetriever,
    user_question: str,
    openai_client: OpenAiClient,
) -> str:
    user_question_embedding: List[float] = openai_client.create_embedding_vector(
        input=user_question
    )
    best_matches: List[str] = pinecone_retriever.retrieve_best_matches(
        query_embedding=user_question_embedding,
    )
    best_match: str = best_matches[0]

    return openai_client.ask_llm(context=best_match, user_question=user_question)


#################
# main()
#################
config: Config = ConfigBoatManuals()
openai_client: OpenAiClient = OpenAiClient(
    system_prompt_content_template=config.get_system_prompt_content_template()
)
pinecone_query_response_parser: PineconeQueryResponseParser = PineconeQueryResponseParserBoatManuals()
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
