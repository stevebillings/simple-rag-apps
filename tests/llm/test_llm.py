import pytest
from unittest.mock import patch, Mock
from src.llm.llm import Llm
from src.llm.llm_client import LlmClient
from src.llm.alt_question_generator import AltQuestionGenerator
from src.llm.llm_prompt import LlmPrompt


@pytest.fixture
def mock_llm_client():
    mock = Mock(spec=LlmClient)
    mock.create_embedding_vector_for_input.return_value = [0.1, 0.2, 0.3]
    mock.ask_llm_with_context.return_value = "Mock response"
    return mock

@pytest.fixture
def mock_alt_question_generator():
    mock = Mock(spec=AltQuestionGenerator)
    mock.generate_alt_questions.return_value = ["alt1", "alt2", "alt3"]
    return mock


def test_generate_alt_questions(mock_llm_client, mock_alt_question_generator) -> None:
    # Arrange
    template = "You are an assistant. Use this context: {}"
    client = Llm(system_prompt_content_template=template, llm_client=mock_llm_client, alt_question_generator=mock_alt_question_generator)
    question = "How does this work?"

    # Act
    result = client.generate_alt_questions(question)

    # Assert
    assert result == ["alt1", "alt2", "alt3"]
    mock_alt_question_generator.generate_alt_questions.assert_called_once_with(question)


def test_construct_user_query(mock_llm_client, mock_alt_question_generator) -> None:
    # Arrange
    client = Llm(system_prompt_content_template="template", llm_client=mock_llm_client, alt_question_generator=mock_alt_question_generator)
    question = "How does this work?"

    # Act
    result = client.construct_user_query(question)

    # Assert
    assert result["role"] == "user"
    assert result["content"] == "How does this work?"


def test_ask_llm_with_context(mock_llm_client, mock_alt_question_generator) -> None:
    # Arrange
    client = Llm(system_prompt_content_template="template", llm_client=mock_llm_client, alt_question_generator=mock_alt_question_generator)
    context = "Important context"
    question = "How does this work?"

    # Act
    result = client.ask_llm_with_context(context, question)

    # Assert
    assert result == "Mock response"
