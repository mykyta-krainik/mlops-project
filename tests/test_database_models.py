import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data.models import ReviewedComment
from src.data.database import DatabaseManager


class TestReviewedCommentModel:
    def test_model_attributes(self):
        comment = ReviewedComment(
            comment_text="Test comment",
            original_predictions={"toxic": 0.8, "severe_toxic": 0.2},
            reviewed_labels={"toxic": 1, "severe_toxic": 0},
            moderator_id="test_moderator",
            reviewed_at=datetime.now(),
            source="api",
            model_version="v1.0.0",
            status="reviewed",
        )

        assert comment.comment_text == "Test comment"
        assert comment.original_predictions == {"toxic": 0.8, "severe_toxic": 0.2}
        assert comment.reviewed_labels == {"toxic": 1, "severe_toxic": 0}
        assert comment.moderator_id == "test_moderator"
        assert comment.source == "api"
        assert comment.model_version == "v1.0.0"
        assert comment.status == "reviewed"

    def test_to_dict_method(self):
        now = datetime.now()
        comment = ReviewedComment(
            id=1,
            comment_text="Test comment",
            original_predictions={"toxic": 0.8},
            reviewed_labels={"toxic": 1},
            moderator_id="test_moderator",
            reviewed_at=now,
            created_at=now,
            source="api",
            model_version="v1.0.0",
            status="reviewed",
        )

        result = comment.to_dict()

        assert result["id"] == 1
        assert result["comment_text"] == "Test comment"
        assert result["original_predictions"] == {"toxic": 0.8}
        assert result["reviewed_labels"] == {"toxic": 1}
        assert result["moderator_id"] == "test_moderator"
        assert result["reviewed_at"] == now
        assert result["created_at"] == now
        assert result["source"] == "api"
        assert result["model_version"] == "v1.0.0"
        assert result["status"] == "reviewed"

    def test_optional_fields(self):
        comment = ReviewedComment(
            comment_text="Test comment",
            status="pending",
        )

        assert comment.comment_text == "Test comment"
        assert comment.status == "pending"
        assert comment.original_predictions is None
        assert comment.reviewed_labels is None
        assert comment.moderator_id is None
        assert comment.reviewed_at is None
        assert comment.model_version is None


class TestDatabaseManager:
    @pytest.fixture
    def mock_credentials(self):
        return {
            "host": "localhost",
            "port": 5432,
            "dbname": "test_db",
            "username": "test_user",
            "password": "test_password",
        }

    @pytest.fixture
    def mock_secrets_manager(self, mock_credentials):
        with patch("boto3.client") as mock_client:
            mock_sm = MagicMock()
            mock_sm.get_secret_value.return_value = {
                "SecretString": str(mock_credentials).replace("'", '"')
            }
            mock_client.return_value = mock_sm
            yield mock_sm

    def test_initialization(self):
        manager = DatabaseManager(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_password",
        )

        assert manager._host == "localhost"
        assert manager._port == 5432
        assert manager._database == "test_db"
        assert manager._user == "test_user"
        assert manager._password == "test_password"

    def test_initialization_with_secret_arn(self):
        manager = DatabaseManager(
            secret_arn="arn:aws:secretsmanager:us-east-1:123456789:secret:test"
        )

        assert manager._secret_arn == "arn:aws:secretsmanager:us-east-1:123456789:secret:test"

    @patch("src.data.database.create_engine")
    def test_create_engine_with_lambda_pool(self, mock_create_engine, mock_credentials):
        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
            use_lambda_pool=True,
        )

        _ = manager.engine

        mock_create_engine.assert_called_once()
        call_kwargs = mock_create_engine.call_args[1]
        assert "poolclass" in call_kwargs

    @patch("src.data.database.create_engine")
    def test_create_engine_with_connection_pool(self, mock_create_engine, mock_credentials):
        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
            use_lambda_pool=False,
        )

        _ = manager.engine

        mock_create_engine.assert_called_once()
        call_kwargs = mock_create_engine.call_args[1]
        assert "pool_size" in call_kwargs
        assert "max_overflow" in call_kwargs
        assert "pool_pre_ping" in call_kwargs

    @patch("src.data.database.create_engine")
    def test_get_session_context_manager(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
        )

        mock_session = MagicMock()
        with patch("src.data.database.sessionmaker") as mock_sessionmaker:
            mock_sessionmaker.return_value = lambda: mock_session

            with manager.get_session() as session:
                assert session == mock_session

            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    @patch("src.data.database.create_engine")
    def test_get_session_rollback_on_exception(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
        )

        mock_session = MagicMock()
        with patch("src.data.database.sessionmaker") as mock_sessionmaker:
            mock_sessionmaker.return_value = lambda: mock_session

            with pytest.raises(ValueError):
                with manager.get_session() as session:
                    raise ValueError("Test error")

            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
            mock_session.commit.assert_not_called()

    @patch("src.data.database.create_engine")
    def test_create_tables(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
        )

        with patch("src.data.models.Base.metadata.create_all") as mock_create_all:
            manager.create_tables()
            mock_create_all.assert_called_once_with(mock_engine)

    @patch("src.data.database.create_engine")
    def test_close_engine(self, mock_create_engine):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_password",
        )

        _ = manager.engine

        manager.close()

        mock_engine.dispose.assert_called_once()
        assert manager._engine is None
        assert manager._session_factory is None
