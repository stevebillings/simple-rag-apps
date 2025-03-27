import pytest
from unittest.mock import patch, Mock
from src.llm.openai_client import OpenAiClient


def test_construct_system_prompt() -> None:
    # Arrange
    template = "You are an assistant. Use this context: {}"
    client = OpenAiClient(system_prompt_content_template=template)
    context = "Important information"
    
    # Act
    result = client.construct_system_prompt(context)
    
    # Assert
    assert result["role"] == "system"
    assert result["content"] == "You are an assistant. Use this context: Important information"


def test_construct_user_query() -> None:
    # Arrange
    client = OpenAiClient(system_prompt_content_template="template")
    question = "How does this work?"
    
    # Act
    result = client.construct_user_query(question)
    
    # Assert
    assert result["role"] == "user"
    assert result["content"] == "How does this work?"


def test_assemble_system_prompt_and_user_query() -> None:
    # Arrange
    client = OpenAiClient(system_prompt_content_template="template")
    system_prompt = {"role": "system", "content": "System prompt"}
    user_query = {"role": "user", "content": "User query"}
    
    # Act
    result = client.assemble_system_prompt_and_user_query(system_prompt, user_query)
    
    # Assert
    assert len(result) == 2
    assert result[0] == system_prompt
    assert result[1] == user_query


def test_insert_context_into_prompt_template() -> None:
    # Arrange
    template = "Context: {}"
    client = OpenAiClient(system_prompt_content_template="not used here")
    context = "Test context"
    
    # Act
    result = client._insert_context_into_prompt_template_at_curly_braces(template, context)
    
    # Assert
    assert result == "Context: Test context"