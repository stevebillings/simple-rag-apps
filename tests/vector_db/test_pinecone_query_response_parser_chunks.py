import pytest
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.config.config import CorpusType


def test_parse_relevant_content_from_query_response() -> None:
    # Arrange
    parser = PineconeQueryResponseParser.create_parser(CorpusType.PDFS)
    test_response = {
        "matches": [
            {
                "id": "chunk_1",
                "score": 0.95,
                "metadata": {"chunk": "This is a test chunk content."},
            }
        ]
    }

    # Act
    result = parser.parse_relevant_content_from_query_response(test_response)

    # Assert
    assert result == ["This is a test chunk content."]


def test_parse_relevant_content_from_query_response_multiple_matches() -> None:
    # Arrange
    parser = PineconeQueryResponseParser.create_parser(CorpusType.PDFS)
    test_response = {
        "matches": [
            {
                "id": "chunk_1",
                "score": 0.95,
                "metadata": {"chunk": "This is the first chunk."},
            },
            {
                "id": "chunk_2",
                "score": 0.85,
                "metadata": {"chunk": "This is the second chunk."},
            },
        ]
    }

    # Act
    result = parser.parse_relevant_content_from_query_response(test_response)

    # Assert
    assert result == ["This is the first chunk.", "This is the second chunk."]
