from vector_db.pinecone_query_response_parser import PineconeQueryResponseParser

class PineconeQueryResponseParserBoatManuals(PineconeQueryResponseParser):
    
    def get_answer(self, response: dict) -> str:
        return response["matches"][0]["metadata"]["chunk"]