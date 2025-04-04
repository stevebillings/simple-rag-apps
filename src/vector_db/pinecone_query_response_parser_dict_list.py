import json
from typing import Any, Dict, List
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserDictList(PineconeQueryResponseParser):

    def parse_relevant_content_from_query_response(self, query_response: Dict[str, Any]) -> List[str]:
        results: List[Dict[str, Any]] = query_response["matches"]

        result = results[0]
        metadata_str: str = json.dumps(result["metadata"])
        return [metadata_str]
