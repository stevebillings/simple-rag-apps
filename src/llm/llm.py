import os
from typing import List
import openai
import numpy as np
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from src.llm.llm_client import LlmClient


class Llm:
    def __init__(
        self,
        system_prompt_content_template: str,
        llm_client: LlmClient,
    ):
        self._system_prompt_content_template = system_prompt_content_template
        self._llm_client = llm_client


    def create_embedding_vector_for_input(self, input: str):
        return self._llm_client.create_embedding_vector_for_input(input)

    def ask_llm_with_context(self, context: str, user_question: str):
        system_prompt = self._construct_system_prompt(context)
        user_query = self.construct_user_query(user_question)
        messages = self._assemble_system_prompt_and_user_query(
            system_prompt, user_query
        )

        return self._llm_client.ask_llm_with_context(messages)

    def _assemble_system_prompt_and_user_query(
        self, system_prompt, user_query
    ) -> List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
        return [system_prompt, user_query]

    def construct_user_query(self, user_question) -> ChatCompletionUserMessageParam:
        return {
            "role": "user",
            "content": user_question,
        }

    def _construct_system_prompt(self, context) -> ChatCompletionSystemMessageParam:
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
