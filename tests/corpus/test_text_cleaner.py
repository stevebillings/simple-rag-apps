import pytest
from src.corpus.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_keep_punctuation(cleaner) -> None:
    text = "Hello, world! This is a test. Or is it?"
    expected = "Hello, world! This is a test. Or is it?"
    assert cleaner.clean(text) == expected


def test_keep_alphanumeric_and_whitespace(cleaner) -> None:
    text = "Test123 with @#$% special & chars"
    expected = "Test123 with @#$% special & chars"
    assert cleaner.clean(text) == expected


def test_empty_string(cleaner) -> None:
    assert cleaner.clean("") == ""