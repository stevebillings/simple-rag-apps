from io import StringIO
import re
import os
from typing import List
from PyPDF2 import PdfReader


class PdfDocument:

    def __init__(self, pdf_path: str) -> None:
        if not os.path.isdir(pdf_path):
            raise ValueError(f"Expected a directory path, but got: {pdf_path}")
        self._pdf_dir = pdf_path
        self._pdf_files = [
            f for f in os.listdir(pdf_path) if f.lower().endswith(".pdf")
        ]
        if not self._pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_path}")

    def extract_chunks(self) -> List[str]:
        text_buffer = StringIO()

        for pdf_file in self._pdf_files:
            pdf_path = os.path.join(self._pdf_dir, pdf_file)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                page_text = self._remove_non_alphanumeric(page_text)
                if page_text:
                    text_buffer.write(page_text)

        chunks: List[str] = self._chunk_text(text_buffer.getvalue())
        return chunks

    def _remove_non_alphanumeric(self, page_text):
        page_text = re.sub(r"[^\w\s]", "", page_text)
        return page_text

    def _chunk_text(
        self, text: str, chunk_size: int = 200, overlap: int = 50
    ) -> List[str]:
        words: List[str] = text.split()
        chunks: List[str] = []
        start: int = 0

        valid_words: List[str] = self._extract_valid_words(words)

        while start < len(valid_words):
            end: int = start + chunk_size
            chunk_words: List[str] = valid_words[start:end]
            chunk: str = " ".join(chunk_words)
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def _extract_valid_words(self, words: List[str]) -> List[str]:
        valid_words: List[str] = []
        for word in words:
            if len(word) > 20:
                continue
            valid_words.append(word)
        return valid_words
