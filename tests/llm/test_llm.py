import pytest
from unittest.mock import patch, Mock
from src.llm.llm import Llm
from src.llm.llm_client import LlmClient


@pytest.fixture
def mock_llm_client():
    mock = Mock(spec=LlmClient)
    mock.create_embedding_vector_for_input.return_value = [0.1, 0.2, 0.3]
    mock.ask_llm_with_context.return_value = "Mock response"
    return mock


def test_construct_system_prompt(mock_llm_client) -> None:
    # Arrange
    template = "You are an assistant. Use this context: {}"
    client = Llm(system_prompt_content_template=template, llm_client=mock_llm_client)
    context = "Important information"

    # Act
    result = client._construct_system_prompt(context)

    # Assert
    assert result["role"] == "system"
    assert (
        result["content"]
        == "You are an assistant. Use this context: Important information"
    )


def test_construct_user_query(mock_llm_client) -> None:
    # Arrange
    client = Llm(system_prompt_content_template="template", llm_client=mock_llm_client)
    question = "How does this work?"

    # Act
    result = client.construct_user_query(question)

    # Assert
    assert result["role"] == "user"
    assert result["content"] == "How does this work?"


def test_assemble_system_prompt_and_user_query(mock_llm_client) -> None:
    # Arrange
    client = Llm(system_prompt_content_template="template", llm_client=mock_llm_client)
    system_prompt = {"role": "system", "content": "System prompt"}
    user_query = {"role": "user", "content": "User query"}

    # Act
    result = client._assemble_system_prompt_and_user_query(system_prompt, user_query)

    # Assert
    assert len(result) == 2
    assert result[0] == system_prompt
    assert result[1] == user_query


def test_insert_context_into_prompt_template(mock_llm_client) -> None:
    # Arrange
    template = "Context: {}"
    client = Llm(
        system_prompt_content_template="not used here", llm_client=mock_llm_client
    )
    context = "Test context"

    # Act
    result = client._insert_context_into_prompt_template_at_curly_braces(
        template, context
    )

    # Assert
    assert result == "Context: Test context"
