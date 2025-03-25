import os
from typing import Dict, List, Optional
import openai
import numpy as np

class OpenAiClient:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_client = openai.OpenAI(api_key=openai_api_key)
        self._model = "text-embedding-3-small"

    def create_embedding_vector(self, question: str):
        response = self._openai_client.embeddings.create(
            model=self._model,
            input=question,
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_most_similar_question(
            self,
        user_question, faq_vector_db
    ) -> Optional[str]:
        query_embedding = self.create_embedding_vector(user_question)
        best_match: Optional[str] = None
        highest_similarity = -1

        for faq_question, faq_vector in faq_vector_db.items():
            similarity = self._cosine_similarity(query_embedding, faq_vector)
            if similarity > highest_similarity:
                best_match = faq_question
                highest_similarity = similarity

        return best_match

    def ask_llm(self, system_prompt_content: str, user_question: str):
        system_prompt: Dict[str, str] = {
            "role": "system",
            "content": system_prompt_content,
        }
        user_query: Dict[str, str] = {
            "role": "user",
            "content": user_question,
        }
        messages: List[Dict[str, str]] = [system_prompt, user_query]
        response = self._openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )
        return response.choices[0].message.content
