import abc
from typing import Any, Dict, List


class PineconeQueryResponseParser(abc.ABC):

    @abc.abstractmethod
    def parse_relevant_content_from_query_response(self, query_response: Dict[str, Any]) -> List[str]:
        pass
