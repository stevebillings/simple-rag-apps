
class WordValidator:

    def __init__(self, max_word_length: int = 20):
        self._max_word_length: int = max_word_length

    def is_valid(self, word: str) -> bool:
        return len(word) <= self._max_word_length