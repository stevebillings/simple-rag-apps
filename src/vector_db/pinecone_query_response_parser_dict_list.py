import json
from typing import Any, Dict, List
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserDictList(PineconeQueryResponseParser):

    def parse_relevant_content_from_query_response(self, query_response: Dict[str, Any]) -> List[str]:
        results: List[Dict[str, Any]] = query_response["matches"]
        relevant_content_list: List[str] = []
        for result in results:
            score: float = result["score"]
            if score > 0.5:
                metadata_str: str = json.dumps(result["metadata"])
                relevant_content_list.append(metadata_str)
        return relevant_content_list
