import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import boto3
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from src.data.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5432,
        database: str = "reviews",
        user: Optional[str] = None,
        password: Optional[str] = None,
        secret_arn: Optional[str] = None,
        use_lambda_pool: bool = False,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._secret_arn = secret_arn or os.environ.get("DB_SECRET_ARN")
        self._use_lambda_pool = use_lambda_pool
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    def _get_credentials(self) -> Dict[str, Any]:
        if self._secret_arn:
            try:
                client = boto3.client("secretsmanager")
                response = client.get_secret_value(SecretId=self._secret_arn)
                creds = json.loads(response["SecretString"])
                logger.info("Retrieved credentials from Secrets Manager")
                return creds
            except Exception as e:
                logger.warning(f"Failed to get credentials from Secrets Manager: {e}")

        return {
            "host": self._host or os.environ.get("RDS_HOST", "localhost"),
            "port": self._port or int(os.environ.get("RDS_PORT", 5432)),
            "dbname": self._database or os.environ.get("RDS_DATABASE", "reviews"),
            "username": self._user or os.environ.get("RDS_USER", "admin"),
            "password": self._password or os.environ.get("RDS_PASSWORD", ""),
        }

    def _create_engine(self) -> Engine:
        creds = self._get_credentials()

        connection_url = (
            f"postgresql://{creds['username']}:{creds['password']}"
            f"@{creds['host']}:{creds['port']}/{creds['dbname']}"
        )

        if self._use_lambda_pool:
            engine = create_engine(
                connection_url,
                poolclass=NullPool,
                echo=False,
            )
            logger.info("Created engine with NullPool for Lambda")
        else:
            engine = create_engine(
                connection_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("Created engine with connection pooling")

        return engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")
