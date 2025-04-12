from typing import List, Dict, Any
from src.llm.llm_client import LlmClient
from src.llm.llm_prompt import LlmPrompt
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from src.config.config import Config


class AltQuestionGenerator:
    def __init__(
        self,
        config: Config,
        llm_client: LlmClient,
        llm_prompt: LlmPrompt,
    ):
        self._config = config
        self._llm_client = llm_client
        self._llm_prompt = llm_prompt
        self._system_prompt_template = config.get_alt_question_generator_system_prompt_content_template()

    def generate_alt_questions(self, user_question: str) -> List[str]:
        system_prompt_content = self._system_prompt_template.format(
            user_question=user_question
        )
        messages: List[
            ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
        ] = [{"role": "system", "content": system_prompt_content}]
        return self._llm_client.ask_llm_for_json(messages=messages)
