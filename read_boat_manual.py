import os
import sys
from typing import Any, Dict, List, Optional

from pdf_document import PdfDocument



#################
# main()
#################
pdf_document: PdfDocument = PdfDocument(pdf_path="data/Glastron-Owners-Manual-2022.pdf")
text: str = pdf_document.extract_text_from_pdf()
