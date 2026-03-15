"""Unit tests for training step logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import ToxicCommentClassifier
from src.train import compute_metrics


# ── compute_metrics ────────────────────────────────────────────────────────────

def _make_binary_labels(n: int, n_labels: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(n, n_labels))


def test_compute_metrics_keys() -> None:
    y = _make_binary_labels(50)
    metrics = compute_metrics(y, y, y.astype(float))
    assert "f1_macro" in metrics
    assert "roc_auc_macro" in metrics
    assert "accuracy" in metrics


def test_compute_metrics_perfect_predictions() -> None:
    y = _make_binary_labels(50)
    metrics = compute_metrics(y, y, y.astype(float))
    assert metrics["f1_macro"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_shape_mismatch_raises() -> None:
    y_true = _make_binary_labels(50)
    y_wrong = _make_binary_labels(40)
    with pytest.raises(Exception):
        compute_metrics(y_true, y_wrong, y_wrong.astype(float))


# ── ToxicCommentClassifier ─────────────────────────────────────────────────────

SYNTHETIC_COMMENTS = [
    "This is a nice comment",
    "You are terrible and I hate you",
    "Great article thank you",
    "This is stupid garbage",
    "I agree with your points",
    "You should be ashamed of yourself",
    "Well written analysis",
    "Disgusting content",
    "Very informative piece",
    "Stop spreading misinformation",
]


def _make_synthetic_dataset(n: int = 100):
    rng = np.random.default_rng(0)
    comments = [SYNTHETIC_COMMENTS[i % len(SYNTHETIC_COMMENTS)] for i in range(n)]
    labels = rng.integers(0, 2, size=(n, 6))
    return comments, labels


def test_classifier_fit_predict() -> None:
    X, y = _make_synthetic_dataset(100)
    model = ToxicCommentClassifier(max_features=500, max_iter=100)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5, 6)
    assert set(preds.flatten().tolist()).issubset({0, 1})


def test_classifier_predict_proba_range() -> None:
    X, y = _make_synthetic_dataset(100)
    model = ToxicCommentClassifier(max_features=500, max_iter=100)
    model.fit(X, y)
    probas = model.predict_proba(X[:5])
    assert probas.shape == (5, 6)
    assert (probas >= 0).all() and (probas <= 1).all()


def test_classifier_predict_single_string() -> None:
    X, y = _make_synthetic_dataset(100)
    model = ToxicCommentClassifier(max_features=500, max_iter=100)
    model.fit(X, y)
    result = model.predict_single("some comment text")
    assert isinstance(result, dict)
    assert len(result) == 6


def test_classifier_default_hyperparams() -> None:
    model = ToxicCommentClassifier()
    assert model.max_features == 10000
    assert model.ngram_range == (1, 2)
    assert model.C == 1.0


def test_improved_hyperparams_differ() -> None:
    baseline = ToxicCommentClassifier(max_features=10000, ngram_range=(1, 2), C=1.0)
    improved = ToxicCommentClassifier(max_features=20000, ngram_range=(1, 3), C=5.0)
    assert improved.max_features > baseline.max_features
    assert improved.C > baseline.C
    assert improved.ngram_range[1] > baseline.ngram_range[1]


def test_onnx_roundtrip(tmp_path: Path) -> None:
    X, y = _make_synthetic_dataset(100)
    model = ToxicCommentClassifier(max_features=500, max_iter=100)
    model.fit(X, y)

    onnx_path = tmp_path / "model.onnx"
    model.save_onnx(onnx_path)

    loaded = ToxicCommentClassifier()
    loaded.load_onnx(onnx_path)

    original_proba = model.predict_proba(X[:3])
    loaded_proba = loaded.predict_proba(X[:3])
    np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-5)
