from typing import Dict

from src.api.schemas import ModerationAction
from src.config import config


class ModerationDecider:
    def __init__(
        self,
        ban_severe_toxic: float = None,
        ban_threat: float = None,
        ban_toxic: float = None,
        ban_obscene: float = None,
        ban_insult: float = None,
        ban_identity_hate: float = None,
        review_min: float = None,
    ):
        thresholds = config.moderation
        self.ban_severe_toxic = ban_severe_toxic or thresholds.ban_severe_toxic
        self.ban_threat = ban_threat or thresholds.ban_threat
        self.ban_toxic = ban_toxic or thresholds.ban_toxic
        self.ban_obscene = ban_obscene or thresholds.ban_obscene
        self.ban_insult = ban_insult or thresholds.ban_insult
        self.ban_identity_hate = ban_identity_hate or thresholds.ban_identity_hate

        self.review_min = review_min or thresholds.review_min

    def decide(self, predictions: Dict[str, float]) -> ModerationAction:
        for label, score in predictions.items():
            if score > self._get_ban_threshold(label):
                return ModerationAction.BAN

            if score >= self.review_min:
                return ModerationAction.REVIEW

        return ModerationAction.ALLOW

    def _get_ban_threshold(self, label: str) -> float:
        thresholds = {
            "toxic": self.ban_toxic,
            "severe_toxic": self.ban_severe_toxic,
            "threat": self.ban_threat,
            "obscene": self.ban_obscene,
            "insult": self.ban_insult,
            "identity_hate": self.ban_identity_hate,
        }
        return thresholds.get(label, 0.8)

    def is_toxic(self, predictions: Dict[str, float], threshold: float = 0.5) -> bool:
        return any(score > threshold for score in predictions.values())


moderation_decider = ModerationDecider()
