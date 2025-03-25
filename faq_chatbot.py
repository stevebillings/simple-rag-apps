import os
import sys
from typing import Any, Dict, List, Optional

from frequently_asked_questions import FrequentlyAskedQuestions
from openai_client import OpenAiClient
from pinecone_client import PineconeClient


def rag_chatbot(
    pinecone_client: PineconeClient,
    user_question: str,
    openai_client: OpenAiClient,
) -> str:
    user_question_embedding: List[float] = openai_client.create_embedding_vector(
        question=user_question
    )
    best_answer: str = pinecone_client.retrieve_best_faq_answer(
        query_embedding=user_question_embedding,
        top_k=1,
    )

    return openai_client.ask_llm(context=best_answer, user_question=user_question)


#################
# main()
#################
faq: FrequentlyAskedQuestions = FrequentlyAskedQuestions()
openai_client: OpenAiClient = OpenAiClient()
pinecone_client = PineconeClient(faq=faq, openai_client=openai_client)

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
            pinecone_client=pinecone_client,
            openai_client=openai_client,
        )
        print(
            f">> {resp_msg}\n\n"
        )
    except EOFError:  # Handle Control-D
        break
    except KeyboardInterrupt:  # Handle Control-C
        break

print("\nGoodbye!")
