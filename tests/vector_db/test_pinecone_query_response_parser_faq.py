import pytest
from src.vector_db.pinecone_query_response_parser_faq import PineconeQueryResponseParserFaq


def test_parse_answer_from_query_response():
    # Arrange
    parser = PineconeQueryResponseParserFaq()
    test_response = {
        "matches": [
            {
                "id": "faq_1",
                "score": 0.95,
                "metadata": {
                    "question": "How do I track my order?",
                    "answer": "Log in to your account and check the order status."
                }
            }
        ]
    }
    
    # Act
    result = parser.parse_answer_from_query_response(test_response)
    
    # Assert
    assert result == "Log in to your account and check the order status."


def test_parse_answer_from_query_response_multiple_matches():
    # Arrange
    parser = PineconeQueryResponseParserFaq()
    test_response = {
        "matches": [
            {
                "id": "faq_1",
                "score": 0.95,
                "metadata": {
                    "question": "How do I track my order?",
                    "answer": "First answer"
                }
            },
            {
                "id": "faq_2",
                "score": 0.85,
                "metadata": {
                    "question": "How do I cancel my order?",
                    "answer": "Second answer"
                }
            }
        ]
    }
    
    # Act
    result = parser.parse_answer_from_query_response(test_response)
    
    # Assert
    assert result == "First answer"