import os
from typing import Dict, List
import openai


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
