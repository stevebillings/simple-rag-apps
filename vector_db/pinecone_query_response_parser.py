import abc

class PineconeQueryResponseParser(abc.ABC):

    @abc.abstractmethod
    def get_answer(self, response: dict) -> str:
        pass