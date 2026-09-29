"""Shared test configuration.

Keeps the test suite away from the working directory and from the real model API:

- The database and the vector store folder are pointed at a temporary directory
  before any application module reads its settings.
- The API key is replaced with a dummy value, so a key in the environment or in
  ``.env`` is never used by a test.
- Any call that reaches the real Anthropic client fails the test that made it.
- Background learning extraction is replaced with a recorder, because the chat
  endpoints start it in a thread that outlives the request.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest

_TEST_DIR = tempfile.mkdtemp(prefix="ai-cofounder-tests-")

# Environment variables take priority over ``.env`` in pydantic-settings. These
# must be set before ``src.config.settings`` is first used.
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_DIR, "test.db").replace("\\", "/")
os.environ["VECTOR_STORE_DIR"] = os.path.join(_TEST_DIR, "vector_stores")
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"


def pytest_sessionfinish(session, exitstatus):
    """Release the test database and remove the temporary directory."""
    try:
        from src.memory.database import engine

        engine.dispose()
    except Exception:
        pass
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def no_real_model_calls(monkeypatch):
    """Fail any test that reaches the real Anthropic Messages API.

    Tests are expected to patch ``get_haiku_client`` / ``get_sonnet_client`` (or
    the function that calls them). The application catches exceptions around
    model calls, so attempts are also recorded and checked after the test.
    """
    attempts = []

    def _blocked(*args, **kwargs):
        attempts.append(kwargs.get("model", "unknown model"))
        raise AssertionError("A test reached the real Anthropic client; mock the model call.")

    monkeypatch.setattr("anthropic.resources.messages.Messages.create", _blocked)
    monkeypatch.setattr("anthropic.resources.messages.Messages.stream", _blocked)
    yield attempts
    assert attempts == [], f"Unmocked model calls were attempted: {attempts}"


@pytest.fixture(autouse=True)
def background_extraction(monkeypatch):
    """Replace background learning extraction with a recorder.

    Returns the recorder so a test can check what the endpoint handed to it.
    """
    import src.api.routes.chat as chat

    recorder = MagicMock(name="_run_learning_extraction_background")
    monkeypatch.setattr(chat, "_run_learning_extraction_background", recorder)
    return recorder
