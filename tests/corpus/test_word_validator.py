import pytest
from src.corpus.word_validator import WordValidator


def test_valid_word():
    validator = WordValidator(max_word_length=20)
    assert validator.is_valid("short")
    assert validator.is_valid("exactly20characters")


def test_invalid_word():
    validator = WordValidator(max_word_length=10)
    assert not validator.is_valid("thisisalongwordthatexceedsmax")


def test_default_max_length():
    validator = WordValidator()  # Default max_word_length=20
    assert validator.is_valid("a" * 20)
    assert not validator.is_valid("a" * 21)