from typing import List
from PyPDF2 import PdfReader

class PdfDocument:

    def __init__(self, pdf_path: str) -> None:
        self._reader: PdfReader = PdfReader(pdf_path)

    def extract_text_from_pdf(self) -> List[str]:
        pages: List[str] = []
        for page in self._reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return pages