import os
from typing import Dict, List, Optional
import openai
import numpy as np
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class OpenAiClient:
    def __init__(self, system_prompt_content_template: str):
        self._system_prompt_content_template = system_prompt_content_template
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_client = openai.OpenAI(api_key=openai_api_key)
        self._model = "text-embedding-3-small"

    def create_embedding_vector(self, input: str):
        response = self._openai_client.embeddings.create(
            model=self._model,
            input=input,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_most_similar_question(self, user_question, faq_vector_db) -> Optional[str]:
        query_embedding = self.create_embedding_vector(user_question)
        best_match: Optional[str] = None
        highest_similarity = -1

        for faq_question, faq_vector in faq_vector_db.items():
            similarity = self._cosine_similarity(query_embedding, faq_vector)
            if similarity > highest_similarity:
                best_match = faq_question
                highest_similarity = similarity

        return best_match

    def ask_llm(self, context: str, user_question: str):
        system_prompt_content: str = self._prompt_builder(
            template=self._system_prompt_content_template,
            context=context,
        )
        system_prompt: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt_content,
        }
        user_query: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_question,
        }
        messages: List[
            ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
        ] = [system_prompt, user_query]
        response = self._openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )
        return response.choices[0].message.content

    def _prompt_builder(self, template: str, context: str) -> str:
        return template.format(context)
