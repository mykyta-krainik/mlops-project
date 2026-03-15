"""Unit tests for TextPreprocessor feature transforms."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


def test_lowercase(preprocessor: TextPreprocessor) -> None:
    result = preprocessor.preprocess_text("Hello World THIS IS A TEST")
    assert result == result.lower()


def test_url_removal(preprocessor: TextPreprocessor) -> None:
    result = preprocessor.preprocess_text("Check this out https://example.com for more info")
    assert "https" not in result
    assert "example.com" not in result


def test_html_tag_removal(preprocessor: TextPreprocessor) -> None:
    result = preprocessor.preprocess_text("<b>bold text</b> and <i>italic</i>")
    assert "<b>" not in result
    assert "</b>" not in result
    assert "bold text" in result


def test_empty_string(preprocessor: TextPreprocessor) -> None:
    result = preprocessor.preprocess_text("")
    assert isinstance(result, str)


def test_whitespace_normalisation(preprocessor: TextPreprocessor) -> None:
    result = preprocessor.preprocess_text("too   many    spaces")
    assert "  " not in result


def test_preprocess_dataframe_adds_no_rows(preprocessor: TextPreprocessor) -> None:
    df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "comment_text": ["Hello", "World", "Test"],
        "toxic": [0, 1, 0],
        "severe_toxic": [0, 0, 0],
        "obscene": [0, 0, 0],
        "threat": [0, 0, 0],
        "insult": [0, 1, 0],
        "identity_hate": [0, 0, 0],
    })
    result = preprocessor.preprocess_dataframe(df)
    assert len(result) <= len(df)
    assert "comment_text" in result.columns


def test_preprocess_dataframe_drops_empty_comments(preprocessor: TextPreprocessor) -> None:
    df = pd.DataFrame({
        "id": ["1", "2"],
        "comment_text": ["valid comment", ""],
        "toxic": [0, 0],
        "severe_toxic": [0, 0],
        "obscene": [0, 0],
        "threat": [0, 0],
        "insult": [0, 0],
        "identity_hate": [0, 0],
    })
    result = preprocessor.preprocess_dataframe(df)
    # Empty comments should be dropped or have non-empty text
    assert all(len(str(c).strip()) > 0 for c in result["comment_text"])
