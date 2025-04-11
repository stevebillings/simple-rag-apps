from typing import List, Dict, Any
from openai.types.chat import ChatCompletionUserMessageParam
from src.llm.llm_client import LlmClient
from src.llm.llm_prompt import LlmPrompt
from src.llm.alt_question_generator import AltQuestionGenerator


class Llm:
    def __init__(
        self,
        system_prompt_content_template: str,
        llm_client: LlmClient,
        alt_question_generator: AltQuestionGenerator,
    ):
        self._llm_prompt = LlmPrompt(system_prompt_content_template)
        self._llm_client = llm_client
        self._alt_question_generator = alt_question_generator

    def generate_alt_questions(self, user_question: str) -> List[str]:
        return self._alt_question_generator.generate_alt_questions(user_question)

    def create_embedding_vector_for_input(self, input: str):
        return self._llm_client.create_embedding_vector_for_input(input)

    def ask_llm_with_context(self, context: str, user_question: str):
        system_prompt = self._llm_prompt.construct_system_prompt(context)
        user_query = self.construct_user_query(user_question)
        messages = self._llm_prompt.assemble_system_prompt_and_user_query(
            system_prompt, user_query
        )

        return self._llm_client.ask_llm_with_context(messages)

    def construct_user_query(self, user_question) -> ChatCompletionUserMessageParam:
        return {
            "role": "user",
            "content": user_question,
        }
