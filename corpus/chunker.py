
from typing import List


class Chunker:

    def __init__(self, chunk_size: int = 200, overlap: int = 50, max_word_length: int = 20) -> None:
        self._chunk_size: int = chunk_size
        self._overlap: int = overlap
        self._max_word_length: int = max_word_length

    def chunk_text(
        self, text: str
    ) -> List[str]:
        words: List[str] = text.split()
        chunks: List[str] = []
        start: int = 0

        valid_words: List[str] = self._extract_valid_words(words)

        while start < len(valid_words):
            end: int = start + self._chunk_size
            self._add_chunk(chunks, start, valid_words, end)
            start += self._chunk_size - self._overlap

        return chunks

    def _extract_valid_words(self, words: List[str]) -> List[str]:
        valid_words: List[str] = []
        for word in words:
            if len(word) > self._max_word_length:
                continue
            valid_words.append(word)
        return valid_words

    def _add_chunk(self, chunks: List[str], start: int, valid_words: List[str], end: int) -> None:
        chunk_words: List[str] = valid_words[start:end]
        chunk: str = " ".join(chunk_words)
        chunks.append(chunk)