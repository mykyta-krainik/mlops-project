import re
import unicodedata
from typing import List, Optional, Union

import pandas as pd


class TextPreprocessor:
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        strip_accents: bool = True,
        min_length: int = 1,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.strip_accents = strip_accents
        self.min_length = min_length

        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_pattern = re.compile(r"<[^>]+>")
        self._newline_pattern = re.compile(r"\n+")
        self._whitespace_pattern = re.compile(r"\s+")
        self._special_char_pattern = re.compile(r"[^\w\s]")
        self._number_pattern = re.compile(r"\d+")

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        result = text

        if self.remove_html:
            result = self._html_pattern.sub(" ", result)

        if self.remove_urls:
            result = self._url_pattern.sub(" ", result)

        if self.lowercase:
            result = result.lower()

        if self.strip_accents:
            result = "".join(
                c for c in unicodedata.normalize("NFD", result)
                if unicodedata.category(c) != "Mn"
            )

        result = self._newline_pattern.sub(" ", result)

        if self.remove_numbers:
            result = self._number_pattern.sub(" ", result)

        if self.remove_special_chars:
            result = self._special_char_pattern.sub(" ", result)

        if self.remove_extra_whitespace:
            result = self._whitespace_pattern.sub(" ", result).strip()

        if len(result) < self.min_length:
            return ""

        return result

    def preprocess_batch(self, texts: Union[List[str], pd.Series]) -> List[str]:
        return [self.preprocess_text(text) for text in texts]

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "comment_text",
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        output_col = output_column or text_column
        df[output_col] = df[text_column].apply(self.preprocess_text)
        df = df[df[output_col].str.len() > 0].reset_index(drop=True)
        return df


def validate_dataset_schema(df: pd.DataFrame) -> bool:
    required_columns = {
        "id": str,
        "comment_text": str,
        "toxic": (int, float, bool),
        "severe_toxic": (int, float, bool),
        "obscene": (int, float, bool),
        "threat": (int, float, bool),
        "insult": (int, float, bool),
        "identity_hate": (int, float, bool),
    }

    missing_columns = set(required_columns.keys()) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    validate_dataset_schema(df)
    return df

