from typing import List, Dict, Any

from src.llm.llm import Llm
from src.vector_db.pinecone_retriever import PineconeRetriever
from src.vector_db.dto.scored_match import ScoredMatch

class Chat:

    def __init__(
        self,
        pinecone_retriever: PineconeRetriever,
        openai_client: Llm,
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
        alt_questions: List[str] = self.openai_client.generate_alt_questions(user_question)
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
        best_matches: List[ScoredMatch] = self.pinecone_retriever.retrieve_best_matches(
            query_embedding=user_question_embedding,
        )
        best_matches_str: str = "\n\n".join([match.get_match() for match in best_matches])

        return self.openai_client.ask_llm_with_context(
            context=best_matches_str, user_question=user_question
        )
