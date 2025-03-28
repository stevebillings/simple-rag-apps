from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 50,
    ) -> None:
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        return self._text_splitter.split_text(text)
