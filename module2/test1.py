import os
import sys
from typing import Any, Dict, List, Optional
from pinecone import Pinecone  # type: ignore
import numpy as np
from frequently_asked_questions import FrequentlyAskedQuestions
from openai_client import OpenAiClient


def prompt_builder(template: str, context: str) -> str:
    return template.format(context)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_similar_question(
    user_question, faq_vector_db, openai_client
) -> Optional[str]:
    query_embedding = openai_client.create_embedding_vector(user_question)
    best_match: Optional[str] = None
    highest_similarity = -1

    for faq_question, faq_vector in faq_vector_db.items():
        similarity = cosine_similarity(query_embedding, faq_vector)
        if similarity > highest_similarity:
            best_match = faq_question
            highest_similarity = similarity

    return best_match


def rag_chatbot(
    faq: FrequentlyAskedQuestions,
    user_question: str,
    embedding_vector_database: Dict[str, Any],
    faq_database: Dict[str, str],
    openai_client: Any,
) -> str:
    best_match: Optional[str] = find_most_similar_question(
        user_question, embedding_vector_database, openai_client
    )
    assert best_match
    best_answer: str = faq.lookup_answer(best_match)
    system_prompt_content_template: str = """
        You are a helpful e-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
        Our store sells e-books and courses for IT professionals. 

        Base your answers on the information in the following context. If the context does not contain the information, say that you don't know:

        Context: {}
    """
    augmented_prompt_content: str = prompt_builder(
        template=system_prompt_content_template, context=best_answer
    )
    return openai_client.ask_llm(system_prompt_content=augmented_prompt_content, user_question=user_question)

def create_pinecone_client():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc_index_name = "faq-database"
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pc_index_name)
    return index


def create_faq() -> Dict[str, str]:
    faq: Dict[str, str] = {
        "How do I track my order?": "The best way to track your order is to log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Track'.",
        "Cancelling an order": "Within 30 minutes after placing an order, you can cancel the order. To do this, log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Cancel'.",
    }
    return faq


def create_pinecone_upsertable_embedding_vectors(
    create_embedding_vector, openai_client, faq_database
) -> List[Dict[str, Any]]:
    upsertable_embedding_vectors: List[Dict[str, Any]] = []
    for i, (q, a) in enumerate(faq_database.items()):
        upsertable_embedding_vectors.append(
            {
                "id": str(i),
                "values": create_embedding_vector(q, openai_client),
                "metadata": {"question": q, "answer": a},
            }
        )
    return upsertable_embedding_vectors


#################
# main()
#################
faq: FrequentlyAskedQuestions = FrequentlyAskedQuestions()
openai_client: OpenAiClient = OpenAiClient()
pinecone_client = create_pinecone_client()
faq_database: Dict[str, str] = create_faq()
# upsertable_embedding_vectors: List[Dict[str, Any]] = create_pinecone_upsertable_embedding_vectors(create_embedding_vector, openai_client, faq_database)
# pinecone_client.upsert(vectors=upsertable_embedding_vectors, namespace='ns1')

# sys.exit(0)

faq_embedding_vector_database = {}
for faq_question in faq_database:
    faq_embedding_vector_database[faq_question] = openai_client.create_embedding_vector(
        question=faq_question
    )

user_question_answerable: str = "Can I track an order?"
user_question_unanswerable: str = "If I take a course and don't like it, can I get a refund?"

resp_msg: str = rag_chatbot(
    faq=faq,
    user_question=user_question_answerable,
    embedding_vector_database=faq_embedding_vector_database,
    faq_database=faq_database,
    openai_client=openai_client,
)
print(f"resp_msg: {resp_msg}")
