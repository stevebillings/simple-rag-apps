from io import StringIO
import re
import os
from typing import List
from PyPDF2 import PdfReader
from PyPDF2 import PageObject
from corpus.text_chunker import TextChunker
from corpus.text_cleaner import TextCleaner


class PdfDocumentSet:

    def __init__(
        self, chunker: TextChunker, text_cleaner: TextCleaner, pdf_dir_path: str
    ) -> None:
        self._text_cleaner: TextCleaner = text_cleaner
        self._chunker: TextChunker = chunker
        if not os.path.isdir(pdf_dir_path):
            raise ValueError(f"Expected a directory path, but got: {pdf_dir_path}")
        self._pdf_dir = pdf_dir_path
        self._pdf_files = [
            f for f in os.listdir(pdf_dir_path) if f.lower().endswith(".pdf")
        ]
        if not self._pdf_files:
            raise ValueError(f"No PDF files found in directory: {pdf_dir_path}")

    def extract_chunks(self) -> List[str]:
        text: str = self._extract_text()
        chunks: List[str] = self._chunker.chunk_text(text)
        return chunks

    def _extract_text(self) -> str:
        text_buffer: StringIO = StringIO()
        for pdf_file in self._pdf_files:
            self._add_file_text_to_buffer(text_buffer, pdf_file)
        return text_buffer.getvalue()

    def _add_file_text_to_buffer(self, text_buffer: StringIO, pdf_file: str) -> None:
        pdf_path: str = os.path.join(self._pdf_dir, pdf_file)
        reader: PdfReader = PdfReader(pdf_path)
        for page in reader.pages:
            self._add_page_text_to_buffer(text_buffer, page)

    def _add_page_text_to_buffer(self, text_buffer: StringIO, page: PageObject) -> None:
        page_text: str = page.extract_text()
        page_text_cleaned: str = self._text_cleaner.clean(page_text)
        if page_text_cleaned:
            text_buffer.write(page_text_cleaned)
