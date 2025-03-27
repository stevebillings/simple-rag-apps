import abc


class PineconeQueryResponseParser(abc.ABC):

    @abc.abstractmethod
    def parse_relevant_content_from_query_response(self, response: dict) -> str:
        pass
