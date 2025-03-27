from typing import List

from src.llm.openai_client import OpenAiClient
from src.vector_db.pinecone_retriever import PineconeRetriever


class Chat:

    def __init__(
        self,
        pinecone_retriever: PineconeRetriever,
        openai_client: OpenAiClient,
        bot_prompt: str,
    ):
        self.pinecone_retriever = pinecone_retriever
        self.openai_client = openai_client
        self.bot_prompt = bot_prompt

    def chat_loop(self) -> None:

        print(f"{self.bot_prompt} (or type 'exit' to quit):")
        while True:
            try:
                should_continue: bool = self._ask_and_answer_question()
                if not should_continue:
                    break
            except EOFError:  # Handle Control-D
                break
            except KeyboardInterrupt:  # Handle Control-C
                break

        print("\nGoodbye!")

    def _ask_and_answer_question(self) -> bool:
        user_question = input("> ").strip()
        if user_question.lower() == "exit":
            return False
        if not user_question:
            return True

        resp_msg: str = self._ask_llm(
            user_question=user_question,
        )
        print(f">> {resp_msg}\n\n")
        return True

    def _ask_llm(
        self,
        user_question: str,
    ) -> str:
        user_question_embedding: List[float] = (
            self.openai_client.create_embedding_vector_for_input(input=user_question)
        )
        best_matches: List[str] = self.pinecone_retriever.retrieve_best_matches(
            query_embedding=user_question_embedding,
        )
        best_matches_str: str = "\n\n".join(best_matches)

        return self.openai_client.ask_llm_with_context(
            context=best_matches_str, user_question=user_question
        )
