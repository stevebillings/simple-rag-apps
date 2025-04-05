import json
from typing import Any, Dict, List
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserDictList(PineconeQueryResponseParser):

    def parse_relevant_content_from_query_response(self, query_response: Dict[str, Any]) -> List[str]:
        results: List[Dict[str, Any]] = query_response["matches"]
        #print(f"Found {len(results)} relevant content items")
        endpoint_descriptions: List[str] = []
        for result in results:
            if result['score'] < 0.30:
                break
            metadata_str: str = json.dumps(result["metadata"])
            #print(f"Score: {result['score']}: {result["metadata"]["name"]}")
            endpoint_descriptions.append(metadata_str + "\n")
        return endpoint_descriptions
