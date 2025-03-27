import pytest
from src.corpus.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_remove_punctuation(cleaner):
    text = "Hello, world! This is a test."
    expected = "Hello world This is a test"
    assert cleaner.clean(text) == expected


def test_keep_alphanumeric_and_whitespace(cleaner):
    text = "Test123 with @#$% special & chars"
    expected = "Test123 with  special  chars"
    assert cleaner.clean(text) == expected


def test_empty_string(cleaner):
    assert cleaner.clean("") == ""