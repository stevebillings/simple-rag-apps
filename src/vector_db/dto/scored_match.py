
class ScoredMatch:
    def __init__(
        self,
        score: float,
        match: str,
    ):
        self._score = score
        self._match = match

    def get_score(self) -> float:
        return self._score

    def get_match(self) -> str:
        return self._match