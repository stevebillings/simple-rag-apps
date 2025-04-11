from typing import List
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class LlmPrompt:
    def __init__(self, system_prompt_content_template: str):
        self._system_prompt_content_template = system_prompt_content_template

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

    def assemble_system_prompt_and_user_query(
        self, system_prompt, user_query
    ) -> List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
        return [system_prompt, user_query]

    def _insert_context_into_prompt_template_at_curly_braces(
        self, prompt_template: str, context: str
    ) -> str:
        return prompt_template.format(context)
