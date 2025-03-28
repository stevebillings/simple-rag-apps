import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from src.corpus.text_chunker import TextChunker
from src.corpus.text_cleaner import TextCleaner


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
        text_parts: List[str] = []
        for pdf_file in self._pdf_files:
            pdf_path: str = os.path.join(self._pdf_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for page in pages:
                page_text = page.page_content
                cleaned_text = self._text_cleaner.clean(page_text)
                if cleaned_text:
                    text_parts.append(cleaned_text)

        return "\n".join(text_parts)
