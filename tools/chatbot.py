import os
import sys
from typing import Any, Dict, List, Optional
from pinecone import Index

from llm.openai_client import OpenAiClient
from vector_db.pinecone_client import PineconeClient
from vector_db.pinecone_retriever import PineconeRetriever

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
openai_client: OpenAiClient = OpenAiClient()
pinecone_client = PineconeClient()
pinecone_index: Index = pinecone_client.connect()
pinecone_retriever = PineconeRetriever(pinecone_index=pinecone_index, pinecone_namespace="boat-manuals")

'''
answer: Optional[str] = rag_chatbot(
    user_question="What is the best way to clean the boat?",
    pinecone_retriever=pinecone_retriever,
    openai_client=openai_client,
)
print(answer)
os._exit(0)
'''

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
