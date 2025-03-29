import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from src.corpus.text_cleaner import TextCleaner


def test_pdf_text_extraction_quality():
    # Get the test PDF path
    test_pdf_path = os.path.join("tests", "resources", "test.pdf")

    # Extract and clean text using PyPDF2 + TextCleaner
    reader = PdfReader(test_pdf_path)
    pypdf_text = ""
    for page in reader.pages:
        pypdf_text += page.extract_text() + "\n"
    cleaner = TextCleaner()
    pypdf_cleaned = cleaner.clean(pypdf_text)

    # Extract text using LangChain (no cleaning)
    loader = PyPDFLoader(test_pdf_path)
    langchain_docs = loader.load()
    langchain_text = "\n".join(doc.page_content for doc in langchain_docs)

    # Compare the results
    print("\nPyPDF2 + TextCleaner extracted text (first 500 chars):")
    print("-" * 80)
    print(pypdf_cleaned[:500])
    print("\nLangChain raw extracted text (first 500 chars):")
    print("-" * 80)
    print(langchain_text[:500])

    # Basic metrics
    print("\nMetrics:")
    print(f"PyPDF2 + TextCleaner text length: {len(pypdf_cleaned)}")
    print(f"LangChain raw text length: {len(langchain_text)}")
    print(f"PyPDF2 + TextCleaner unique characters: {len(set(pypdf_cleaned))}")
    print(f"LangChain raw unique characters: {len(set(langchain_text))}")

    # Additional metrics for raw text quality
    print("\nRaw Text Quality Metrics:")
    print(f"PyPDF2 raw text length: {len(pypdf_text)}")
    print(f"PyPDF2 raw unique characters: {len(set(pypdf_text))}")
    print(f"Characters removed by TextCleaner: {len(pypdf_text) - len(pypdf_cleaned)}")
    print(
        f"Unique characters removed by TextCleaner: {len(set(pypdf_text)) - len(set(pypdf_cleaned))}"
    )

    print("\n\n")
    print(f"PyPDF2 raw unique characters: {set(pypdf_text)}")
    print(f"PyPDF2 cleaned unique characters: {set(pypdf_cleaned)}")
    print(f"LangChain raw unique characters: {set(langchain_text)}")
