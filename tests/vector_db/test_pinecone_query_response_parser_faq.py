import pytest
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.config.config import CorpusType
from src.vector_db.dto.scored_match import ScoredMatch


def test_parse_relevant_content_from_query_response() -> None:
    # Arrange
    parser = PineconeQueryResponseParser.create_parser(CorpusType.FAQ)
    test_response = {
        "matches": [
            {
                "id": "faq_1",
                "score": 0.95,
                "metadata": {
                    "question": "How do I track my order?",
                    "answer": "Log in to your account and check the order status.",
                },
            }
        ]
    }

    # Act
    result = parser.parse_relevant_content_from_query_response(test_response)

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], ScoredMatch)
    assert result[0].get_match() == "Log in to your account and check the order status."
    assert result[0].get_score() == 0.95


def test_parse_relevant_content_from_query_response_multiple_matches() -> None:
    # Arrange
    parser = PineconeQueryResponseParser.create_parser(CorpusType.FAQ)
    test_response = {
        "matches": [
            {
                "id": "faq_1",
                "score": 0.95,
                "metadata": {
                    "question": "How do I track my order?",
                    "answer": "First answer",
                },
            },
            {
                "id": "faq_2",
                "score": 0.85,
                "metadata": {
                    "question": "How do I cancel my order?",
                    "answer": "Second answer",
                },
            },
        ]
    }

    # Act
    result = parser.parse_relevant_content_from_query_response(test_response)

    # Assert
    assert len(result) == 2
    assert isinstance(result[0], ScoredMatch)
    assert isinstance(result[1], ScoredMatch)
    assert result[0].get_match() == "First answer"
    assert result[1].get_match() == "Second answer"
    assert result[0].get_score() == 0.95
    assert result[1].get_score() == 0.85
