import pandas as pd
import pytest

from src.data.preprocessing import TextPreprocessor, validate_dataset_schema


class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_lowercase(self, preprocessor):
        result = preprocessor.preprocess_text("HELLO WORLD")
        assert result == "hello world"

    def test_remove_urls(self, preprocessor):
        text = "Check this out: https://example.com and www.test.com"
        result = preprocessor.preprocess_text(text)
        assert "https" not in result
        assert "www" not in result

    def test_remove_html(self, preprocessor):
        text = "<p>Hello</p> <b>World</b>"
        result = preprocessor.preprocess_text(text)
        assert "<" not in result
        assert ">" not in result

    def test_remove_extra_whitespace(self, preprocessor):
        text = "Hello    World   Test"
        result = preprocessor.preprocess_text(text)
        assert "  " not in result

    def test_empty_input(self, preprocessor):
        result = preprocessor.preprocess_text("")
        assert result == ""

    def test_non_string_input(self, preprocessor):
        result = preprocessor.preprocess_text(None)
        assert result == ""

        result = preprocessor.preprocess_text(123)
        assert result == ""

    def test_batch_processing(self, preprocessor):
        texts = ["HELLO", "WORLD", "TEST"]
        results = preprocessor.preprocess_batch(texts)

        assert len(results) == 3
        assert results[0] == "hello"
        assert results[1] == "world"
        assert results[2] == "test"

    def test_dataframe_processing(self, preprocessor):
        df = pd.DataFrame({
            "comment_text": ["HELLO WORLD", "TEST TEXT"],
            "other_col": [1, 2],
        })

        result = preprocessor.preprocess_dataframe(df)

        assert result["comment_text"].iloc[0] == "hello world"
        assert result["comment_text"].iloc[1] == "test text"
        assert "other_col" in result.columns


class TestSchemaValidation:
    def test_valid_schema(self):
        df = pd.DataFrame({
            "id": ["1", "2"],
            "comment_text": ["text1", "text2"],
            "toxic": [0, 1],
            "severe_toxic": [0, 0],
            "obscene": [0, 1],
            "threat": [0, 0],
            "insult": [0, 1],
            "identity_hate": [0, 0],
        })

        assert validate_dataset_schema(df) is True

    def test_missing_columns(self):
        df = pd.DataFrame({
            "id": ["1", "2"],
            "comment_text": ["text1", "text2"],
        })

        with pytest.raises(ValueError) as exc_info:
            validate_dataset_schema(df)

        assert "Missing required columns" in str(exc_info.value)
