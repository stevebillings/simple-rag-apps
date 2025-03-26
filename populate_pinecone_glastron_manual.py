import os
import sys
from typing import Any, Dict, List, Optional

from frequently_asked_questions import FrequentlyAskedQuestions
from openai_client import OpenAiClient
from pdf_document import PdfDocument
from pinecone_client_glastron_manual import PineconeClientGlastronManual


manual: PdfDocument = PdfDocument(pdf_path="data/Glastron-Owners-Manual-2022.pdf")
chunks: List[str] = manual.extract_text_from_pdf()

openai_client: OpenAiClient = OpenAiClient()
pinecone_client = PineconeClientGlastronManual(chunks=chunks, openai_client=openai_client)
pinecone_client.populate_vector_database()
