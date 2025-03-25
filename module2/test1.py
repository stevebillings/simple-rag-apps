import os
from typing import Any, Dict, List, Optional
import openai
import numpy as np

def get_answer(question: str, database: Dict[str, str]) -> str:
    return database[question]

def prompt_builder(template: str, context: str) -> str:
    return template.format(context)

def create_embedding_vector(query: str, openai_client, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(
        model=model,
        input=query,
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar_question(user_question, faq_vector_db, openai_client) -> Optional[str]:
    query_embedding = create_embedding_vector(user_question, openai_client)
    best_match: Optional[str] = None
    highest_similarity = -1

    for faq_question, faq_vector in faq_vector_db.items():
        similarity = cosine_similarity(query_embedding, faq_vector)
        if similarity > highest_similarity:
            best_match = faq_question
            highest_similarity = similarity

    return best_match

def rag_chatbot(user_question: str, embedding_vector_database: Dict[str, Any], faq_database: Dict[str, str], openai_client: Any) -> str:
    best_match: Optional[str] = find_most_similar_question(user_question, embedding_vector_database, openai_client)
    assert best_match
    best_answer: str = get_answer(best_match, faq_database)
    system_prompt_content_template: str = """
        You are a helpful e-Commerce assistant helping customers with their general questions regarding policies and procedures when buying in our store.
        Our store sells e-books and courses for IT professionals. 

        Base your answers on the information in the following context. If the context does not contain the information, say that you don't know:

        Context: {}
    """
    augmented_prompt_content: str = prompt_builder(template=system_prompt_content_template, context=best_answer)
    system_prompt: Dict[str, str] = {
        "role": "system",
        "content": augmented_prompt_content,
    }
    user_query: Dict[str, str] = {
        "role": "user",
        "content": user_question,
    }
    messages: List[Dict[str, str]] = [system_prompt, user_query]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
    )
    return response.choices[0].message.content


# main()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(
        api_key=openai_api_key
    )
faq_database: Dict[str, str] = {
    "How do I track my order?": "The best way to track your order is to log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Track'.",
    "Cancelling an order": "Within 30 minutes after placing an order, you can cancel the order. To do this, log in, click on 'Account' in the top right corner of any page, select 'Orders' from the menu, select the order from the list, and click 'Cancel'.",
}
faq_embedding_vector_database = {}
for faq_question in faq_database:
    faq_embedding_vector_database[faq_question] = create_embedding_vector(faq_question, openai_client)

user_question: str = "If I take a course and don't like it, can I get a refund?"

resp_msg: str = rag_chatbot(user_question=user_question, embedding_vector_database=faq_embedding_vector_database, faq_database=faq_database, openai_client=openai_client)
print(f"resp_msg: {resp_msg}")




