from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserFaq(PineconeQueryResponseParser):

    def parse_relevant_content_from_query_response(self, response: dict) -> str:
        return response["matches"][0]["metadata"]["answer"]
