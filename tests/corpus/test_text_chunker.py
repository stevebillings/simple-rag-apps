import pytest
from src.corpus.text_chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=100, overlap=20)


def test_chunk_text_standard(chunker) -> None:
    text = "This is a test of the chunking system"
    chunks = chunker.chunk_text(text)

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify each chunk is not too long
    for chunk in chunks:
        assert len(chunk) <= 100

    # Verify overlap between chunks
    for i in range(len(chunks) - 1):
        # Check that consecutive chunks share some content
        overlap = set(chunks[i].split()) & set(chunks[i + 1].split())
        assert len(overlap) > 0


def test_empty_text(chunker) -> None:
    chunks = chunker.chunk_text("")
    assert chunks == []


def test_text_chunker():
    # Create a sample text with multiple paragraphs
    text = """This is the first paragraph.
    
    This is the second paragraph with some more content that should be split into chunks.
    
    This is the third paragraph that should also be properly chunked."""

    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text(text)

    # Verify we got multiple chunks
    assert len(chunks) > 1

    # Verify each chunk is not too long
    for chunk in chunks:
        assert len(chunk) <= 100

    # Verify overlap between chunks
    for i in range(len(chunks) - 1):
        # Check that consecutive chunks share some content
        overlap = set(chunks[i].split()) & set(chunks[i + 1].split())
        assert len(overlap) > 0
