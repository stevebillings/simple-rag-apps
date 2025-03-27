from typing import Any, Dict, List
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserChunks(PineconeQueryResponseParser):

    def parse_relevant_content_from_query_response(self, query_response: Dict[str, Any]) -> List[str]:
        results: List[Dict[str, Any]] = query_response["matches"]
        relevant_content_list: List[str] = []
        for result in results:
            score: float = result["score"]
            if score > 0.5:
                relevant_content_list.append(result["metadata"]["chunk"])
        return relevant_content_list
