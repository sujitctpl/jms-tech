import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment
os.environ["APP_ENV"] = "testing"
os.environ["LOG_LEVEL"] = "WARNING"

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.workers.video_worker import worker_manager

# In-memory SQLite engine for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def cleanup_workers():
    yield
    worker_manager.stop_all()


@pytest.fixture
def sample_entry_line_config():
    return {
        "line": [[100.0, 200.0], [500.0, 200.0]],
        "direction": "top_to_bottom",
    }


@pytest.fixture
def sample_camera_payload(sample_entry_line_config):
    return {
        "name": "Main Entrance Camera",
        "stream_url": "rtsp://user:secret123@192.168.1.100:554/stream1",
        "location": "Front Gate Showroom",
        "entry_line_config": sample_entry_line_config,
    }
