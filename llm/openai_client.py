import os
from typing import Dict, List, Optional
import openai
import numpy as np
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class OpenAiClient:
    def __init__(
        self,
        system_prompt_content_template: str,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o",
        max_tokens: int = 500,
        openai_api_key_env_var_name: str = "OPENAI_API_KEY",
    ):
        self._system_prompt_content_template = system_prompt_content_template
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

    def ask_llm_with_context(self, context: str, user_question: str):
        system_prompt = self.construct_system_prompt(context)
        user_query = self.construct_user_query(user_question)
        messages = self.assemble_system_prompt_and_user_query(system_prompt, user_query)

        response = self._openai_client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def assemble_system_prompt_and_user_query(self, system_prompt, user_query) -> List[
            ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
        ]:
        return [system_prompt, user_query]

    def construct_user_query(self, user_question) -> ChatCompletionUserMessageParam:
        return {
            "role": "user",
            "content": user_question,
        }

    def construct_system_prompt(self, context) -> ChatCompletionSystemMessageParam:
        system_prompt_content: str = (
            self._insert_context_into_prompt_template_at_curly_braces(
                prompt_template=self._system_prompt_content_template,
                context=context,
            )
        )
        return {
            "role": "system",
            "content": system_prompt_content,
        }

    def _insert_context_into_prompt_template_at_curly_braces(
        self, prompt_template: str, context: str
    ) -> str:
        return prompt_template.format(context)
