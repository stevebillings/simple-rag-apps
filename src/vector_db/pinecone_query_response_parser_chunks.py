from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser


class PineconeQueryResponseParserChunks(PineconeQueryResponseParser):

    def parse_answer_from_query_response(self, query_response: dict) -> str:
        return query_response["matches"][0]["metadata"]["chunk"]
