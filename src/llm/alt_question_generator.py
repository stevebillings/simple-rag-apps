from typing import List, Dict, Any
from src.llm.llm_client import LlmClient
from src.llm.llm_prompt import LlmPrompt
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class AltQuestionGenerator:
    def __init__(
        self,
        llm_client: LlmClient,
        llm_prompt: LlmPrompt,
        num_alt_questions: int = 5,
    ):
        self._llm_client = llm_client
        self._llm_prompt = llm_prompt
        self._num_alt_questions = num_alt_questions
        self._system_prompt_template = """
        You are an expert at generating alternative questions for a given question.

        You will be given a user question, and asked to generate alternatives for it.
        The user question is about using a REST API to retrieve information or make changes to it.
        Specifically, they want to know which endpoint or endpoints might help.
        Your task is to generate {num_alt_questions} alternative questions for the given question.

        The alternative questions you generate will be used to retrieve relevant documents from a vector database.
        The questions should be short and concise.
        The output should be a list of strings in JSON format; each string is an alternative question.

        Here is the user's question; please generate alternative questions for it:
        
        {user_question}
        """

    def generate_alt_questions(self, user_question: str) -> List[str]:
        system_prompt_content = self._system_prompt_template.format(
            num_alt_questions=self._num_alt_questions,
            user_question=user_question
        )
        messages: List[
            ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
        ] = [{"role": "system", "content": system_prompt_content}]
        return self._llm_client.ask_llm_for_json(messages=messages)
