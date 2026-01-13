import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from contextlib import contextmanager


class TestDataPreparationLambda:
    """Tests for the data preparation Lambda function."""

    @pytest.fixture
    def mock_aws_clients(self):
        """Mock AWS clients."""
        with patch.dict(os.environ, {
            "PROJECT_NAME": "mlops-toxic",
            "ENVIRONMENT": "test",
            "RAW_DATA_BUCKET": "test-raw-bucket",
            "PROCESSED_DATA_BUCKET": "test-processed-bucket",
            "MODELS_BUCKET": "test-models-bucket",
            "DB_SECRET_ARN": "arn:aws:secretsmanager:test",
            "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789:role/test-role",
            "TRAINING_IMAGE_URI": "123456789.dkr.ecr.us-east-1.amazonaws.com/test:latest",
        }):
            with patch("boto3.client") as mock_client:
                yield mock_client

    def test_list_raw_data_files(self, mock_aws_clients):
        """Test listing raw data files from S3."""
        from lambda_pkg.data_preparation.handler import list_raw_data_files

        mock_s3 = MagicMock()
        mock_s3.get_paginator.return_value.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/train.csv"},
                    {"Key": "data/batch_001.csv"},
                    {"Key": "data/readme.txt"},  # Should be filtered out
                ]
            }
        ]
        mock_aws_clients.return_value = mock_s3

        # Re-import to get patched version
        import importlib
        import lambda_pkg.data_preparation.handler as handler_module
        importlib.reload(handler_module)

        files = handler_module.list_raw_data_files()
        assert "data/train.csv" in files
        assert "data/batch_001.csv" in files
        assert "data/readme.txt" not in files

    def test_merge_datasets(self, mock_aws_clients):
        """Test merging raw data with reviewed comments."""
        import pandas as pd
        from lambda_pkg.data_preparation.handler import merge_datasets

        raw_df = pd.DataFrame({
            "id": ["1", "2"],
            "comment_text": ["Hello", "World"],
            "toxic": [0, 1],
            "severe_toxic": [0, 0],
            "obscene": [0, 1],
            "threat": [0, 0],
            "insult": [0, 1],
            "identity_hate": [0, 0],
        })

        reviewed_df = pd.DataFrame({
            "id": ["3_reviewed"],
            "comment_text": ["New reviewed comment"],
            "toxic": [1],
            "severe_toxic": [0],
            "obscene": [0],
            "threat": [0],
            "insult": [1],
            "identity_hate": [0],
        })

        merged = merge_datasets(raw_df, reviewed_df)

        assert len(merged) == 3
        assert "New reviewed comment" in merged["comment_text"].values


class TestModelPromotionLambda:
    """Tests for the model promotion Lambda function."""

    @pytest.fixture
    def mock_aws_clients(self):
        """Mock AWS clients."""
        with patch.dict(os.environ, {
            "PROJECT_NAME": "mlops-toxic",
            "ENVIRONMENT": "test",
            "MODELS_BUCKET": "test-models-bucket",
            "BASELINE_MODEL_GROUP": "test-baseline",
            "IMPROVED_MODEL_GROUP": "test-improved",
            "PROMOTION_THRESHOLD": "0.02",
        }):
            with patch("boto3.client") as mock_client:
                yield mock_client

    def test_should_promote_better_model(self, mock_aws_clients):
        """Test that a better model gets promoted."""
        from lambda_pkg.model_promotion.handler import should_promote_model

        new_metrics = {"roc_auc_macro": 0.95, "f1_macro": 0.90}
        current_metrics = {"roc_auc_macro": 0.92, "f1_macro": 0.88}
        threshold = 0.02

        should_promote, reason = should_promote_model(
            new_metrics, current_metrics, threshold
        )

        assert should_promote is True
        assert "0.03" in reason or "exceeding" in reason.lower()

    def test_should_not_promote_worse_model(self, mock_aws_clients):
        """Test that a worse model does not get promoted."""
        from lambda_pkg.model_promotion.handler import should_promote_model

        new_metrics = {"roc_auc_macro": 0.91, "f1_macro": 0.88}
        current_metrics = {"roc_auc_macro": 0.92, "f1_macro": 0.90}
        threshold = 0.02

        should_promote, reason = should_promote_model(
            new_metrics, current_metrics, threshold
        )

        assert should_promote is False
        assert "not exceed" in reason.lower() or "does not" in reason.lower()

    def test_promote_first_model(self, mock_aws_clients):
        """Test that the first model gets promoted when no current model exists."""
        from lambda_pkg.model_promotion.handler import should_promote_model

        new_metrics = {"roc_auc_macro": 0.85, "f1_macro": 0.80}
        current_metrics = {}  # No current model
        threshold = 0.02

        should_promote, reason = should_promote_model(
            new_metrics, current_metrics, threshold
        )

        assert should_promote is True
        assert "no current" in reason.lower()


class TestReviewDatabase:
    @pytest.fixture
    def mock_session(self):
        mock_session = MagicMock()
        
        @contextmanager
        def mock_get_session():
            yield mock_session
        
        with patch("src.review.database.DatabaseManager") as mock_db_manager:
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_session = mock_get_session
            mock_db_manager.return_value = mock_manager_instance
            yield mock_session

    def test_add_pending_review(self, mock_session):
        from src.review.database import ReviewDatabase

        db = ReviewDatabase(
            host="localhost",
            database="test",
            user="test",
            password="test",
        )

        review_id = db.add_pending_review(
            comment_text="Test comment",
            predictions={"toxic": 0.8},
            model_version="v1.0.0",
            source="test",
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        
        added_comment = mock_session.add.call_args[0][0]
        assert added_comment.comment_text == "Test comment"
        assert added_comment.original_predictions == {"toxic": 0.8}
        assert added_comment.model_version == "v1.0.0"
        assert added_comment.source == "test"
        assert added_comment.status == "pending"

    def test_submit_review(self, mock_session):
        from src.review.database import ReviewDatabase
        from src.data.models import ReviewedComment

        mock_comment = MagicMock(spec=ReviewedComment)
        mock_comment.id = 1
        mock_comment.status = "pending"
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_comment
        mock_session.execute.return_value = mock_result

        db = ReviewDatabase(
            host="localhost",
            database="test",
            user="test",
            password="test",
        )

        success = db.submit_review(
            review_id=1,
            reviewed_labels={"toxic": 1, "severe_toxic": 0},
            moderator_id="test_moderator",
        )

        assert success is True
        assert mock_comment.reviewed_labels == {"toxic": 1, "severe_toxic": 0}
        assert mock_comment.moderator_id == "test_moderator"
        assert mock_comment.status == "reviewed"
        assert mock_comment.reviewed_at is not None

    def test_get_pending_reviews(self, mock_session):
        from src.review.database import ReviewDatabase
        from src.data.models import ReviewedComment

        # Create mock comment objects
        mock_comment1 = MagicMock(spec=ReviewedComment)
        mock_comment1.id = 1
        mock_comment1.comment_text = "Test 1"
        mock_comment1.original_predictions = {"toxic": 0.8}
        mock_comment1.model_version = "v1.0.0"
        mock_comment1.source = "api"
        mock_comment1.created_at = datetime.now()

        mock_comment2 = MagicMock(spec=ReviewedComment)
        mock_comment2.id = 2
        mock_comment2.comment_text = "Test 2"
        mock_comment2.original_predictions = {"toxic": 0.6}
        mock_comment2.model_version = "v1.0.0"
        mock_comment2.source = "api"
        mock_comment2.created_at = datetime.now()

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_comment1, mock_comment2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        db = ReviewDatabase(
            host="localhost",
            database="test",
            user="test",
            password="test",
        )

        reviews = db.get_pending_reviews(limit=10, offset=0)

        assert len(reviews) == 2
        assert reviews[0]["id"] == 1
        assert reviews[0]["comment_text"] == "Test 1"
        assert reviews[1]["id"] == 2
        assert reviews[1]["comment_text"] == "Test 2"

    def test_get_pending_count(self, mock_session):
        from src.review.database import ReviewDatabase

        mock_session.execute.return_value.scalar.return_value = 5

        db = ReviewDatabase(
            host="localhost",
            database="test",
            user="test",
            password="test",
        )

        count = db.get_pending_count()

        assert count == 5
        mock_session.execute.assert_called_once()
