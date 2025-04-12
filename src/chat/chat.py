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
        best_matches_str: str = self._collect_best_matches(user_question, alt_questions)
        resp_msg: str = self._ask_llm(
            user_question=user_question,
            best_matches_str=best_matches_str,
        )
        print(f">> {resp_msg}\n\n")
        return True

    def _ask_llm(
        self,
        user_question: str,
        best_matches_str: str,
    ) -> str:

        return self.openai_client.ask_llm_with_context(
            context=best_matches_str, user_question=user_question
        )

    def _collect_best_matches(self, user_question: str, alt_questions: List[str]) -> str:
        scored_matches = self._collect_all_matches(user_question, alt_questions)
        docs: List[str] = self._filter_w_reciprocal_rank_fusion(scored_matches)        
        return "\n\n".join(docs)

    def _collect_all_matches(self, user_question: str, alt_questions: List[str]) -> List[ScoredMatch]:
        scored_matches: List[ScoredMatch] = []
        user_question_embedding: List[float] = (
            self.openai_client.create_embedding_vector_for_input(input=user_question)
        )
        best_matches_for_user_question: List[ScoredMatch] = self.pinecone_retriever.retrieve_best_matches(
            query_embedding=user_question_embedding,
        )
        scored_matches.extend(best_matches_for_user_question)
        # Then, get scored matches for the alt questions
        for alt_question in alt_questions:
            alt_question_embedding: List[float] = (
                self.openai_client.create_embedding_vector_for_input(input=alt_question)
            )
            alt_question_best_matches: List[ScoredMatch] = self.pinecone_retriever.retrieve_best_matches(
                query_embedding=alt_question_embedding,
            )
            scored_matches.extend(alt_question_best_matches)
        return scored_matches
        
    def _filter_w_reciprocal_rank_fusion(self, scored_matches: List[ScoredMatch], k: int = 60, top_n: int = 5) -> List[str]:
        ranked_docs: Dict[str, float] = {}
        for i, scored_match in enumerate(scored_matches):
            doc: str = scored_match.get_match()
            if doc not in ranked_docs:
                ranked_docs[doc] = 0
            ranked_docs[doc] += 1 / (i + k)
        top_n_docs = [doc for doc, score in sorted(ranked_docs.items(), key=lambda item: item[1], reverse=True)[:top_n]]

        print(f"Reciprocal rank fusion selected {len(top_n_docs)} of the original {len(scored_matches)} matches")
        return top_n_docs
        

