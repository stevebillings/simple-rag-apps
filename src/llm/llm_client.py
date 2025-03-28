import os
from typing import List
import openai
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class LlmClient:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o",
        max_tokens: int = 500,
        openai_api_key_env_var_name: str = "OPENAI_API_KEY",
    ):
        openai_api_key = os.getenv(openai_api_key_env_var_name)
        self._openai_client = openai.OpenAI(api_key=openai_api_key)
        self._embedding_model = embedding_model
        self._chat_model = chat_model
        self._max_tokens = max_tokens

    def create_embedding_vector_for_input(self, input: str):
        response = self._openai_client.embeddings.create(
            model=self._embedding_model,
            input=input,
        )
        return response.data[0].embedding

    def ask_llm_with_context(self, messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]):
        response = self._openai_client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content
