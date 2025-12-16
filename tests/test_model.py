import numpy as np
import pytest

from src.models.baseline import ToxicCommentClassifier


class TestToxicCommentClassifier:
    @pytest.fixture
    def sample_data(self):
        X = [
            "this is a normal comment",
            "you are stupid and dumb",
            "great article thanks for sharing",
            "i hate you so much",
            "interesting perspective on the topic",
            "you should die",
        ]
        y = np.array([
            [0, 0, 0, 0, 0, 0],  # normal
            [1, 0, 1, 0, 1, 0],  # toxic, obscene, insult
            [0, 0, 0, 0, 0, 0],  # normal
            [1, 0, 0, 0, 1, 0],  # toxic, insult
            [0, 0, 0, 0, 0, 0],  # normal
            [1, 1, 0, 1, 0, 0],  # toxic, severe, threat
        ])
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        X, y = sample_data
        model = ToxicCommentClassifier(
            max_features=100,
            ngram_range=(1, 1),
        )
        model.fit(X, y)
        return model

    def test_model_init(self):
        model = ToxicCommentClassifier()
        assert not model.is_trained
        assert model.max_features == 10000

    def test_model_training(self, sample_data):
        X, y = sample_data
        model = ToxicCommentClassifier(max_features=100)

        model.fit(X, y)

        assert model.is_trained
        assert model._pipeline is not None

    def test_predict_binary(self, trained_model):
        predictions = trained_model.predict("you are an idiot")

        assert predictions.shape == (1, 6)
        assert predictions.dtype in [np.int64, np.int32, np.float64]

    def test_predict_proba(self, trained_model):
        probs = trained_model.predict_proba("you are an idiot")

        assert probs.shape == (1, 6)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_single(self, trained_model):
        result = trained_model.predict_single("test comment")

        assert isinstance(result, dict)
        assert len(result) == 6
        assert "toxic" in result
        assert "severe_toxic" in result
        assert all(0 <= v <= 1 for v in result.values())

    def test_batch_prediction(self, trained_model):
        texts = ["hello world", "you are stupid", "great job"]
        predictions = trained_model.predict(texts)

        assert predictions.shape == (3, 6)

    def test_get_params(self, trained_model):
        params = trained_model.get_params()

        assert "tfidf_max_features" in params
        assert "lr_C" in params
        assert params["tfidf_max_features"] == 100

    def test_model_not_trained_error(self):
        model = ToxicCommentClassifier()

        with pytest.raises(RuntimeError):
            model.predict("test")

    def test_save_load_sklearn(self, trained_model, tmp_path):
        model_path = tmp_path / "model.pkl"

        trained_model.save_sklearn(model_path)
        assert model_path.exists()

        new_model = ToxicCommentClassifier()
        new_model.load_sklearn(model_path)

        test_text = "test comment"
        original_pred = trained_model.predict_single(test_text)
        loaded_pred = new_model.predict_single(test_text)

        for key in original_pred:
            assert abs(original_pred[key] - loaded_pred[key]) < 1e-6
