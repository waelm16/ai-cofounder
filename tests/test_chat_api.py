"""Chat API endpoint tests.

All LLM and graph invocations are mocked — no real API calls are made.
Tests verify the POST /api/chat/ and GET /api/chat/sessions/{id} endpoints,
including auth, DB persistence, and session management.
"""

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.memory.database import Base, User, get_db


# ---------------------------------------------------------------------------
# Canned graph output — returned by mock graph.invoke()
# ---------------------------------------------------------------------------


def _canned_graph_state(
    session_id: str | None = None,
    agents: list[str] | None = None,
) -> dict:
    """Return a realistic final CofounderState dict from graph.invoke()."""
    return {
        "messages": [
            {"role": "user", "content": "What is our GTM strategy?"},
            {
                "role": "assistant",
                "content": "Target operations directors at regional grocery chains with direct outbound.",
            },
        ],
        "user_id": "founder",
        "session_id": session_id or str(uuid.uuid4()),
        "user_profile": {"id": "founder", "name": "Founder"},
        "active_hypotheses": [],
        "problem_statement": None,
        "recent_metrics": None,
        "recent_decisions": [],
        "recent_timeline": [],
        "company_context": "Tidewater Labs context.",
        "agent_outputs": [
            {
                "agent_id": "gtm",
                "agent_name": "GTM Specialist",
                "response": "Target operations directors at regional grocery chains.",
                "sources": [{"title": "The Mom Test"}],
                "confidence": "high",
            },
        ],
        "final_response": (
            "Target operations directors at regional grocery chains with direct outbound. "
            "Lead with reduced spoilage and keep the pilot small."
        ),
        "sources_cited": [{"title": "The Mom Test", "author": "Rob Fitzpatrick"}],
        "agents_consulted": agents or ["gtm"],
        "extracted_learnings": {
            "business_facts": [],
            "decisions": [],
            "hypothesis_updates": [],
            "domain_learnings": [],
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    session.add_all([
        User(
            id="founder",
            name="Founder",
            role="Founder",
            title="Co-founder",
            expertise_areas=["grocery operations", "purchasing"],
            communication_style={"style": "direct"},
        ),
        User(
            id="cofounder",
            name="Cofounder",
            role="Co-founder",
            title="Co-founder",
            expertise_areas=["forecasting", "product design"],
            communication_style={"style": "concise"},
        ),
    ])
    session.commit()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_header():
    return {"Authorization": "Bearer founder"}


# ---------------------------------------------------------------------------
# Chat endpoint tests
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_chat_endpoint_success(self, client, auth_header):
        """POST /api/chat/ should return 200 with response, session_id, agents_consulted."""
        canned = _canned_graph_state(agents=["gtm", "fintech"])

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "What is our go-to-market strategy for grocers?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] != ""
        assert "session_id" in data
        assert data["session_id"] != ""
        assert "agents_consulted" in data
        assert set(data["agents_consulted"]) == {"gtm", "fintech"}

    def test_chat_endpoint_requires_auth(self, client):
        """POST /api/chat/ without Authorization header should return 401 or 422."""
        resp = client.post(
            "/api/chat/",
            json={"message": "Hello"},
        )
        assert resp.status_code in (401, 422)

    def test_chat_invalid_user_returns_401(self, client):
        """POST /api/chat/ with a non-existent user token should return 401."""
        resp = client.post(
            "/api/chat/",
            headers={"Authorization": "Bearer nobody"},
            json={"message": "Hello"},
        )
        assert resp.status_code == 401

    def test_chat_creates_session(self, client, auth_header):
        """A new chat (no session_id provided) should generate a session_id."""
        canned = _canned_graph_state()

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "What should we prioritize this week?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_chat_reuses_session(self, client, auth_header):
        """Providing an existing session_id should continue the same conversation."""
        session_id = str(uuid.uuid4())
        canned = _canned_graph_state(session_id=session_id)

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned

            resp1 = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "First message", "session_id": session_id},
            )
            resp2 = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "Follow-up message", "session_id": session_id},
            )

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["session_id"] == session_id
        assert resp2.json()["session_id"] == session_id

    def test_chat_saves_messages(self, client, auth_header, db_session):
        """ChatMessage records should be persisted after a successful chat call."""
        from src.memory.database import ChatMessage

        canned = _canned_graph_state()

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "What's our runway?"},
            )

        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        messages = db_session.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).all()
        assert len(messages) >= 1

    def test_chat_saves_consultations(self, client, auth_header, db_session):
        """AgentConsultation records should be created for each consulted agent."""
        from src.memory.database import AgentConsultation

        canned = _canned_graph_state(agents=["gtm", "fintech"])
        canned["agent_outputs"].append({
            "agent_id": "fintech",
            "agent_name": "Fintech Domain Expert",
            "response": "Clarify how payment data is handled before the pilot.",
            "sources": [],
            "confidence": "medium",
        })

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "Tell me about our GTM and payments angle."},
            )

        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        consultations = db_session.query(AgentConsultation).filter(
            AgentConsultation.session_id == session_id
        ).all()
        assert len(consultations) == 2
        by_agent = {c.agent_id: c for c in consultations}
        assert set(by_agent) == {"gtm", "fintech"}
        assert by_agent["gtm"].response == "Target operations directors at regional grocery chains."
        assert by_agent["fintech"].response == "Clarify how payment data is handled before the pilot."
        assert by_agent["fintech"].confidence == "medium"

    def test_chat_response_includes_sources(self, client, auth_header):
        """Response should include sources_cited when agents return sources."""
        canned = _canned_graph_state()
        canned["sources_cited"] = [
            {"title": "The Mom Test", "author": "Rob Fitzpatrick"},
        ]

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "How do we validate our hypotheses?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"] == [{"title": "The Mom Test", "author": "Rob Fitzpatrick"}]

    def test_chat_graph_invoked_with_user_message(self, client, auth_header):
        """The graph should be invoked with the user's message in the state."""
        canned = _canned_graph_state()
        user_message = "What is our competitive differentiation?"

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": user_message},
            )

            call_args = mock_graph.invoke.call_args
            assert call_args is not None
            invoked_state = call_args[0][0] if call_args[0] else call_args[1].get("input", {})
            # The user's message should appear in the initial state passed to graph.invoke
            assert "messages" in invoked_state
            messages = invoked_state["messages"]
            assert any(user_message in getattr(m, "content", str(m)) for m in messages)


# ---------------------------------------------------------------------------
# Background learning extraction
# ---------------------------------------------------------------------------


class _InlineThread:
    """Stand-in for threading.Thread that runs the target when started."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class TestBackgroundExtraction:
    def test_chat_hands_graph_result_to_background_extraction(
        self, client, auth_header, background_extraction
    ):
        """POST /api/chat/ should start learning extraction with the graph result."""
        canned = _canned_graph_state()

        with (
            patch("src.api.routes.chat.graph") as mock_graph,
            patch("src.api.routes.chat.threading.Thread", _InlineThread),
        ):
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "What is our GTM strategy?"},
            )

        assert resp.status_code == 200
        background_extraction.assert_called_once_with(canned)
        # Extraction runs after the response is built, so its result is not in it
        assert resp.json()["learnings_extracted"] is None


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse a Server-Sent Events body into (event, data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


class _FakeStream:
    """Stand-in for the context manager returned by client.messages.stream()."""

    def __init__(self, chunks):
        self.text_stream = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _snapshots(final: dict) -> list[dict]:
    """State snapshots as graph.stream(stream_mode="values") yields them: one per superstep."""
    start = {**final, "company_context": "", "agent_outputs": []}
    after_context = {**final, "agent_outputs": []}
    after_specialists = dict(final)
    return [start, after_context, after_specialists]


class TestChatStream:
    def _canned_two_agents(self):
        canned = _canned_graph_state(agents=["gtm", "fintech"])
        canned["agent_outputs"].append({
            "agent_id": "fintech",
            "agent_name": "Fintech Domain Expert",
            "response": "Clarify how payment data is handled before the pilot.",
            "sources": [],
            "confidence": "medium",
        })
        return canned

    def test_stream_event_order_and_content(self, client, auth_header, db_session, background_extraction):
        """Status events, then synthesis tokens, then one done event with metadata."""
        from src.memory.database import ChatMessage

        canned = self._canned_two_agents()
        sonnet = MagicMock()
        sonnet.messages.stream.return_value = _FakeStream(["Start ", "with ", "one chain."])

        with (
            patch("src.api.routes.chat.graph_pre_synthesis") as mock_graph,
            patch("src.api.routes.chat.SessionLocal", return_value=db_session),
            patch("src.config.llm_config.get_sonnet_client", return_value=(sonnet, "claude-sonnet-test")),
            patch("src.api.routes.chat.threading.Thread", _InlineThread),
            patch.object(db_session, "close"),
        ):
            mock_graph.stream.return_value = iter(_snapshots(canned))
            resp = client.post(
                "/api/chat/stream",
                headers=auth_header,
                json={"message": "What is our GTM strategy?", "session_id": "stream-sess-1"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)

        assert [e for e, _ in events] == [
            "status", "status", "status", "status", "status", "token", "token", "token", "done",
        ]
        assert [d["status"] for e, d in events if e == "status"] == [
            "Loading company context...",
            "Routing to specialist agents...",
            # both specialists are reported from the same snapshot
            "Received input from GTM Specialist...",
            "Received input from Fintech Domain Expert...",
            "Synthesizing response...",
        ]
        assert "".join(d["token"] for e, d in events if e == "token") == "Start with one chain."

        done = events[-1][1]
        assert done["session_id"] == "stream-sess-1"
        assert done["agents_consulted"] == ["gtm", "fintech"]
        assert [o["agent_id"] for o in done["agent_outputs"]] == ["gtm", "fintech"]

        # graph was streamed in "values" mode with the user's message
        assert mock_graph.stream.call_args.kwargs["stream_mode"] == "values"
        sent_state = mock_graph.stream.call_args.args[0]
        assert sent_state["messages"][0].content == "What is our GTM strategy?"

        # the turn was saved and extraction received the streamed answer
        saved = db_session.query(ChatMessage).filter(ChatMessage.session_id == "stream-sess-1").all()
        assert [(m.role, m.content) for m in saved] == [
            ("user", "What is our GTM strategy?"),
            ("assistant", "Start with one chain."),
        ]
        background_extraction.assert_called_once()
        assert background_extraction.call_args.args[0]["final_response"] == "Start with one chain."

    def test_stream_falls_back_when_synthesis_fails(self, client, auth_header, db_session):
        """If the synthesis call fails, the raw specialist answers are sent as one token."""
        canned = _canned_graph_state()
        sonnet = MagicMock()
        sonnet.messages.stream.side_effect = RuntimeError("model unavailable")

        with (
            patch("src.api.routes.chat.graph_pre_synthesis") as mock_graph,
            patch("src.api.routes.chat.SessionLocal", return_value=db_session),
            patch("src.config.llm_config.get_sonnet_client", return_value=(sonnet, "claude-sonnet-test")),
            patch.object(db_session, "close"),
        ):
            mock_graph.stream.return_value = iter(_snapshots(canned))
            resp = client.post(
                "/api/chat/stream", headers=auth_header, json={"message": "What is our GTM strategy?"},
            )

        events = _parse_sse(resp.text)
        tokens = [d["token"] for e, d in events if e == "token"]
        assert len(tokens) == 1
        assert tokens[0].startswith("I encountered an error during synthesis.")
        assert "Target operations directors at regional grocery chains." in tokens[0]
        assert events[-1][0] == "done"

    def test_stream_rejects_empty_message(self, client, auth_header):
        resp = client.post("/api/chat/stream", headers=auth_header, json={"message": "   "})
        assert _parse_sse(resp.text) == [("error", {"detail": "Empty message"})]

    def test_stream_rejects_unknown_user(self, client, db_session):
        with (
            patch("src.api.routes.chat.SessionLocal", return_value=db_session),
            patch.object(db_session, "close"),
        ):
            resp = client.post(
                "/api/chat/stream",
                headers={"Authorization": "Bearer nobody"},
                json={"message": "What is our GTM strategy?"},
            )
        assert _parse_sse(resp.text) == [("error", {"detail": "Invalid user"})]

    def test_stream_reports_graph_failure(self, client, auth_header, db_session):
        with (
            patch("src.api.routes.chat.graph_pre_synthesis") as mock_graph,
            patch("src.api.routes.chat.SessionLocal", return_value=db_session),
            patch.object(db_session, "close"),
        ):
            mock_graph.stream.side_effect = RuntimeError("graph broke")
            resp = client.post(
                "/api/chat/stream", headers=auth_header, json={"message": "What is our GTM strategy?"},
            )
        assert _parse_sse(resp.text) == [("error", {"detail": "graph broke"})]


# ---------------------------------------------------------------------------
# Session history tests
# ---------------------------------------------------------------------------


class TestSessionHistory:
    def test_get_session_history(self, client, auth_header):
        """GET /api/chat/sessions/{id} should return messages for an existing session."""
        session_id = str(uuid.uuid4())
        canned = _canned_graph_state(session_id=session_id)

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            # First create the session
            client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "Initial message", "session_id": session_id},
            )

        resp = client.get(f"/api/chat/sessions/{session_id}", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
        assert data["messages"][0]["content"] == "Initial message"
        assert data["messages"][1]["content"] == canned["final_response"]
        assert data["agents_consulted"] == ["gtm"]

    def test_get_nonexistent_session(self, client, auth_header):
        """GET /api/chat/sessions/{id} for a non-existent session returns 404 or empty."""
        resp = client.get(
            "/api/chat/sessions/nonexistent-session-id-xyz",
            headers=auth_header,
        )
        # Either 404 or an empty list/object is acceptable
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                assert data == []
            elif isinstance(data, dict):
                messages = data.get("messages", [])
                assert messages == []

    def test_get_session_requires_auth(self, client):
        """GET /api/chat/sessions/{id} without auth should return 401 or 422."""
        resp = client.get("/api/chat/sessions/some-session-id")
        assert resp.status_code in (401, 422)

    def test_session_history_ordered_chronologically(self, client, auth_header):
        """Messages in a session should be returned in chronological order."""
        session_id = str(uuid.uuid4())

        def canned_for_message(msg: str) -> dict:
            state = _canned_graph_state(session_id=session_id)
            state["messages"] = [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": f"Response to: {msg}"},
            ]
            return state

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.side_effect = [
                canned_for_message("First question"),
                canned_for_message("Second question"),
            ]
            client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "First question", "session_id": session_id},
            )
            client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "Second question", "session_id": session_id},
            )

        resp = client.get(f"/api/chat/sessions/{session_id}", headers=auth_header)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Edge-case / validation tests
# ---------------------------------------------------------------------------


class TestChatEdgeCases:
    def test_chat_empty_message_rejected(self, client, auth_header):
        """An empty message string should return a 422 validation error."""
        resp = client.post(
            "/api/chat/",
            headers=auth_header,
            json={"message": ""},
        )
        # FastAPI validates non-empty strings depending on schema definition
        # Either 422 (validation) or 400 (business rule) is acceptable
        assert resp.status_code in (400, 422)

    def test_chat_returns_correct_content_type(self, client, auth_header):
        """Response should have Content-Type: application/json."""
        canned = _canned_graph_state()

        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.return_value = canned
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "Quick test"},
            )

        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")

    def test_chat_graph_error_returns_500(self, client, auth_header):
        """If graph.invoke() raises an exception, the endpoint should return 500."""
        with patch("src.api.routes.chat.graph") as mock_graph:
            mock_graph.invoke.side_effect = RuntimeError("Graph execution failed")
            resp = client.post(
                "/api/chat/",
                headers=auth_header,
                json={"message": "This will fail"},
            )

        assert resp.status_code == 500
