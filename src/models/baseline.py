import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from src.config import config


class ToxicCommentClassifier:
    TARGET_COLUMNS = config.model.target_columns

    def __init__(
        self,
        max_features: int = config.model.tfidf_max_features,
        ngram_range: Tuple[int, int] = config.model.tfidf_ngram_range,
        min_df: int = config.model.tfidf_min_df,
        max_df: float = config.model.tfidf_max_df,
        C: float = config.model.lr_C,
        max_iter: int = config.model.lr_max_iter,
        solver: str = config.model.lr_solver,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.C = C
        self.max_iter = max_iter
        self.solver = solver

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[OneVsRestClassifier] = None
        self._pipeline: Optional[Pipeline] = None

        self._onnx_session: Optional[ort.InferenceSession] = None

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None or self._onnx_session is not None

    def _build_pipeline(self) -> Pipeline:
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer="word",
            token_pattern=r"\w{1,}",
            sublinear_tf=True,
        )

        self._classifier = OneVsRestClassifier(
            LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver=self.solver,
                class_weight="balanced",
                n_jobs=-1,
            )
        )

        self._pipeline = Pipeline([
            ("tfidf", self._vectorizer),
            ("classifier", self._classifier),
        ])

        return self._pipeline

    def fit(self, X: List[str], y: np.ndarray) -> "ToxicCommentClassifier":
        pipeline = self._build_pipeline()
        pipeline.fit(X, y)
        return self

    def predict(self, X: Union[str, List[str]]) -> np.ndarray:
        if isinstance(X, str):
            X = [X]

        if self._onnx_session is not None:
            return self._predict_onnx(X)
        elif self._pipeline is not None:
            return self._pipeline.predict(X)
        else:
            raise RuntimeError("Model not trained or loaded")

    def predict_proba(self, X: Union[str, List[str]]) -> np.ndarray:
        if isinstance(X, str):
            X = [X]

        if self._onnx_session is not None:
            return self._predict_proba_onnx(X)
        elif self._pipeline is not None:
            return self._pipeline.predict_proba(X)
        else:
            raise RuntimeError("Model not trained or loaded")

    def predict_single(self, text: str) -> Dict[str, float]:
        probs = self.predict_proba(text)[0]
        return {label: float(prob) for label, prob in zip(self.TARGET_COLUMNS, probs)}

    def _predict_onnx(self, X: List[str]) -> np.ndarray:
        inputs = {self._onnx_session.get_inputs()[0].name: np.array(X)}
        outputs = self._onnx_session.run(None, inputs)
        return outputs[0]

    def _predict_proba_onnx(self, X: List[str]) -> np.ndarray:
        inputs = {self._onnx_session.get_inputs()[0].name: np.array(X)}
        outputs = self._onnx_session.run(None, inputs)
        return outputs[1] # contains probabilities for ONNX classifier

    def save_sklearn(self, path: Union[str, Path]) -> None:
        if self._pipeline is None:
            raise RuntimeError("No sklearn pipeline to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pipeline": self._pipeline,
                    "params": {
                        "max_features": self.max_features,
                        "ngram_range": self.ngram_range,
                        "min_df": self.min_df,
                        "max_df": self.max_df,
                        "C": self.C,
                        "max_iter": self.max_iter,
                        "solver": self.solver,
                    },
                },
                f,
            )

    def load_sklearn(self, path: Union[str, Path]) -> "ToxicCommentClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._pipeline = data["pipeline"]
        self._vectorizer = self._pipeline.named_steps["tfidf"]
        self._classifier = self._pipeline.named_steps["classifier"]

        params = data.get("params", {})
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def save_onnx(self, path: Union[str, Path]) -> None:
        if self._pipeline is None:
            raise RuntimeError("No sklearn pipeline to export")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        initial_type = [("input", StringTensorType([None]))]

        onnx_model = to_onnx(
            self._pipeline,
            initial_types=initial_type,
            target_opset={"": 15, "ai.onnx.ml": 2},
            options={id(self._classifier): {"zipmap": False}},
        )

        onnx.save_model(onnx_model, str(path))

    def load_onnx(self, path: Union[str, Path]) -> "ToxicCommentClassifier":
        path = Path(path)

        self._onnx_session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )

        return self

    def get_params(self) -> Dict:
        return {
            "tfidf_max_features": self.max_features,
            "tfidf_ngram_range": str(self.ngram_range),
            "tfidf_min_df": self.min_df,
            "tfidf_max_df": self.max_df,
            "lr_C": self.C,
            "lr_max_iter": self.max_iter,
            "lr_solver": self.solver,
        }

    def get_feature_names(self) -> Optional[List[str]]:
        if self._vectorizer is not None:
            return self._vectorizer.get_feature_names_out().tolist()
        return None
