import pytest
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.config.config import CorpusType
from src.vector_db.dto.scored_match import ScoredMatch


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
    assert len(result) == 1
    assert isinstance(result[0], ScoredMatch)
    assert result[0].get_match() == "This is a test chunk content."
    assert result[0].get_score() == 0.95


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
    assert len(result) == 2
    assert isinstance(result[0], ScoredMatch)
    assert isinstance(result[1], ScoredMatch)
    assert result[0].get_match() == "This is the first chunk."
    assert result[1].get_match() == "This is the second chunk."
    assert result[0].get_score() == 0.95
    assert result[1].get_score() == 0.85
