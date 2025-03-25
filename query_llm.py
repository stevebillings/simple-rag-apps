import os
import sys
from typing import Any, Dict, List, Optional

from frequently_asked_questions import FrequentlyAskedQuestions
from openai_client import OpenAiClient
from pinecone_client import PineconeClient


def rag_chatbot(
    faq: FrequentlyAskedQuestions,
    user_question: str,
    embedding_vector_database: Dict[str, Any],
    openai_client: OpenAiClient,
) -> str:
    best_match: Optional[str] = openai_client.find_most_similar_question(
        user_question, embedding_vector_database
    )
    assert best_match
    best_answer: str = faq.lookup_answer(best_match)
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
    faq=faq,
    user_question=user_question_answerable,
    embedding_vector_database=faq_embedding_vector_database,
    openai_client=openai_client,
)
print(f"resp_msg: {resp_msg}")
