import abc
from typing import Any, Dict, List
from src.config.config import CorpusType


class PineconeQueryResponseParser(abc.ABC):

    @abc.abstractmethod
    def parse_relevant_content_from_query_response(
        self, query_response: Dict[str, Any]
    ) -> List[str]:
        pass

    @staticmethod
    def create_parser(corpus_type: CorpusType) -> "PineconeQueryResponseParser":
        if corpus_type == CorpusType.PDFS:
            from src.vector_db.pinecone_query_response_parser_chunks import (
                PineconeQueryResponseParserChunks,
            )

            return PineconeQueryResponseParserChunks()
        elif corpus_type == CorpusType.JSON_LIST:
            from src.vector_db.pinecone_query_response_parser_dict_list import (
                PineconeQueryResponseParserDictList,
            )

            return PineconeQueryResponseParserDictList()
        else:
            from src.vector_db.pinecone_query_response_parser_faq import (
                PineconeQueryResponseParserFaq,
            )

            return PineconeQueryResponseParserFaq()
