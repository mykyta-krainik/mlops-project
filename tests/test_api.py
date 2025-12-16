import pytest

from src.api.app import create_app, model_manager
from src.api.moderation import ModerationDecider
from src.api.schemas import ModerationAction


class TestModerationDecider:
    @pytest.fixture
    def decider(self):
        return ModerationDecider()

    def test_ban_severe_toxic(self, decider):
        predictions = {
            "toxic": 0.5,
            "severe_toxic": 0.8,
            "obscene": 0.3,
            "threat": 0.1,
            "insult": 0.4,
            "identity_hate": 0.1,
        }

        action = decider.decide(predictions)
        assert action == ModerationAction.BAN

    def test_ban_threat(self, decider):
        predictions = {
            "toxic": 0.5,
            "severe_toxic": 0.1,
            "obscene": 0.3,
            "threat": 0.7,
            "insult": 0.4,
            "identity_hate": 0.1,
        }

        action = decider.decide(predictions)
        assert action == ModerationAction.BAN

    def test_ban_high_toxic(self, decider):
        predictions = {
            "toxic": 0.9,
            "severe_toxic": 0.1,
            "obscene": 0.3,
            "threat": 0.1,
            "insult": 0.4,
            "identity_hate": 0.1,
        }

        action = decider.decide(predictions)
        assert action == ModerationAction.BAN

    def test_review_medium_scores(self, decider):
        predictions = {
            "toxic": 0.6,
            "severe_toxic": 0.1,
            "obscene": 0.3,
            "threat": 0.1,
            "insult": 0.4,
            "identity_hate": 0.1,
        }

        action = decider.decide(predictions)
        assert action == ModerationAction.REVIEW

    def test_allow_low_scores(self, decider):
        predictions = {
            "toxic": 0.1,
            "severe_toxic": 0.05,
            "obscene": 0.1,
            "threat": 0.02,
            "insult": 0.1,
            "identity_hate": 0.03,
        }

        action = decider.decide(predictions)
        assert action == ModerationAction.ALLOW

    def test_is_toxic(self, decider):
        toxic_preds = {"toxic": 0.6, "severe_toxic": 0.1}
        assert decider.is_toxic(toxic_preds) is True

        safe_preds = {"toxic": 0.1, "severe_toxic": 0.1}
        assert decider.is_toxic(safe_preds) is False


class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.get_json()
        assert "version" in data
        assert "model_type" in data
        assert "target_labels" in data

    def test_predict_without_model(self, client):
        model_manager._model = None

        response = client.post(
            "/predict",
            json={"comment": "test comment"},
        )

        assert response.status_code in [200, 503]

    def test_predict_invalid_request(self, client):
        response = client.post(
            "/predict",
            json={},
        )

        assert response.status_code == 400

    def test_batch_predict_invalid_request(self, client):
        response = client.post(
            "/predict/batch",
            json={"comments": []},
        )

        assert response.status_code == 400
