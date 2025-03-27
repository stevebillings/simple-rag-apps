import pytest
from unittest.mock import Mock
from src.corpus.text_chunker import TextChunker


@pytest.fixture
def word_validator():
    validator = Mock()
    validator.is_valid.return_value = True
    return validator


@pytest.fixture
def chunker(word_validator):
    return TextChunker(
        word_validator=word_validator,
        chunk_size=5,
        overlap=2
    )


def test_chunk_text_standard(chunker, word_validator):
    text = "This is a test of the chunking system"
    chunks = chunker.chunk_text(text)
    
    # With chunk_size=5 and overlap=2, we get 3 chunks from 8 words
    assert len(chunks) == 3
    assert chunks[0] == "This is a test of"
    assert chunks[1] == "test of the chunking system"
    assert chunks[2] == "chunking system"
    
    # Check that word_validator was called for each word
    assert word_validator.is_valid.call_count == 8


def test_chunk_text_with_invalid_words(chunker, word_validator):
    # Configure validator to reject specific words
    word_validator.is_valid.side_effect = lambda word: word != "chunking"
    
    text = "This is a test of the chunking system"
    chunks = chunker.chunk_text(text)
    
    # With "chunking" filtered out, we should have 3 chunks still
    assert len(chunks) == 3
    assert chunks[0] == "This is a test of"
    assert chunks[1] == "test of the system"
    assert chunks[2] == "system"


def test_empty_text(chunker):
    chunks = chunker.chunk_text("")
    assert chunks == []