import pytest
from src.vector_db.pinecone_query_response_parser import PineconeQueryResponseParser
from src.config.config import CorpusType


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
    assert result == ["Log in to your account and check the order status."]


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
    assert result == ["First answer", "Second answer"]
