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
    user_question_embedding: List[float] = openai_client.create_embedding_vector(question=user_question)
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

faq_embedding_vector_database = {}
for faq_question in faq.get_questions():
    faq_embedding_vector_database[faq_question] = openai_client.create_embedding_vector(
        question=faq_question
    )

user_question_answerable: str = "Can I track an order?"
user_question_unanswerable: str = (
    "If I take a course and don't like it, can I get a refund?"
)

resp_msg: str = rag_chatbot(
    user_question=user_question_answerable,
    pinecone_client=pinecone_client,
    openai_client=openai_client,
)
print(f"\n========================\nresp_msg: {resp_msg}\n========================\n")
