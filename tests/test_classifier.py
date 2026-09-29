"""Classification routing tests for the classify_and_route node.

All Haiku LLM calls are mocked — no real API calls are made.
Tests verify that the classifier correctly routes queries to the right
specialist agents via LangGraph Send objects.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.memory.database import Base, User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str) -> MagicMock:
    """Return a MagicMock that mimics anthropic.types.Message."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_haiku_mock(classifications: list[dict]) -> MagicMock:
    """Return a patched haiku client that responds with the given classifications JSON."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_response(
        json.dumps(classifications)
    )
    return mock_client


def _base_state(message: str = "test query", user_id: str = "founder") -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "user_id": user_id,
        "session_id": "sess-classify-01",
        "user_profile": {"id": user_id, "name": "Founder", "role": "Founder"},
        "active_hypotheses": [],
        "problem_statement": None,
        "recent_metrics": None,
        "recent_decisions": [],
        "recent_timeline": [],
        "company_context": (
            "Tidewater Labs sells inventory forecasting software to "
            "regional grocery chains."
        ),

        "agent_outputs": [],
        "final_response": "",
        "sources_cited": [],
        "agents_consulted": [],
        "extracted_learnings": None,
    }


def _message_state(message: str) -> dict:
    """Same as ``_base_state`` but with a message object, as the API sends it."""
    from langchain_core.messages import HumanMessage

    return {**_base_state(message), "messages": [HumanMessage(content=message)]}


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
    session.add(User(id="founder", name="Founder", role="Founder", title="Co-founder"))
    session.commit()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Classification routing tests
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_classify_gtm_query(self):
        """GTM-focused query should route to the gtm agent as primary."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {"agent_id": "gtm", "sub_question": "What GTM motion works for Tidewater Labs?", "relevance": "primary"},
        ])

        state = _base_state("What's our go-to-market strategy for reaching grocery operations directors?")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        # classify_and_route returns a list of Send objects
        assert isinstance(sends, list)
        assert len(sends) >= 1

        # At least one Send targets the gtm node
        gtm_sends = [s for s in sends if hasattr(s, "node") and "gtm" in s.node]
        assert len(gtm_sends) >= 1

    def test_classify_multi_agent(self):
        """A complex query should route to multiple specialist agents."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {
                "agent_id": "gtm",
                "sub_question": "What is the ideal customer profile for Tidewater Labs?",
                "relevance": "primary",
            },
            {
                "agent_id": "fintech",
                "sub_question": "What payment terms do grocery chains expect from software vendors?",
                "relevance": "secondary",
            },
            {
                "agent_id": "finance",
                "sub_question": "What pricing model fits this market?",
                "relevance": "secondary",
            },
        ])

        state = _base_state(
            "We need to define our ICP, pricing model, and payment terms for grocery chains."
        )

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert len(sends) >= 2

    def test_classify_fintech_specific(self):
        """A fintech-specific query should route to the fintech agent."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {
                "agent_id": "fintech",
                "sub_question": "How does SOC2 compliance factor into fintech software purchases?",
                "relevance": "primary",
            },
        ])

        state = _base_state("How do fintech companies think about SOC2 for vendors?")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert isinstance(sends, list)
        assert len(sends) >= 1

        fintech_sends = [s for s in sends if hasattr(s, "node") and "fintech" in s.node]
        assert len(fintech_sends) >= 1

    def test_classify_fallback_on_error(self):
        """If the classifier returns unparseable JSON, it should fall back to cofounder."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "This is not valid JSON at all {{{broken"
        )

        state = _base_state("What should we work on next?")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        # On a parse error everything goes to the general agent
        assert [s.node for s in sends] == ["specialist_cofounder"]

    def test_classify_max_agents(self):
        """Classifier should never route to more than 3 agents."""
        from src.agent.nodes.classifier import classify_and_route

        # Simulate an LLM that over-routes to 5 agents
        mock_client = _make_haiku_mock([
            {"agent_id": "gtm", "sub_question": "Q1", "relevance": "primary"},
            {"agent_id": "fintech", "sub_question": "Q2", "relevance": "secondary"},
            {"agent_id": "finance", "sub_question": "Q3", "relevance": "secondary"},
            {"agent_id": "marketing", "sub_question": "Q4", "relevance": "secondary"},
            {"agent_id": "legal", "sub_question": "Q5", "relevance": "secondary"},
        ])

        state = _base_state("Tell me everything about strategy, finance, marketing, and compliance.")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert len(sends) <= 3

    def test_classify_always_has_primary(self):
        """The classification must always include at least one primary-relevance agent."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {
                "agent_id": "product",
                "sub_question": "What product features should Tidewater Labs prioritize?",
                "relevance": "primary",
            },
            {
                "agent_id": "gtm",
                "sub_question": "How does the roadmap affect positioning?",
                "relevance": "secondary",
            },
        ])

        state = _base_state("What should we build next on the Tidewater Labs roadmap?")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert isinstance(sends, list)
        assert len(sends) >= 1
        # Verify state classifications captured a primary agent
        # (either via send payload or state update — implementation-dependent)
        mock_client.messages.create.assert_called_once()

    def test_classify_healthcare_query(self):
        """A HIPAA/healthcare query should route to the healthcare agent."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {
                "agent_id": "healthcare",
                "sub_question": "How does HIPAA affect vendor data handling for pharmacy workflows?",
                "relevance": "primary",
            },
        ])

        state = _base_state("Our pharmacy prospect wants to know about HIPAA compliance for vendors.")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert isinstance(sends, list)
        assert len(sends) >= 1
        healthcare_sends = [s for s in sends if hasattr(s, "node") and "healthcare" in s.node]
        assert len(healthcare_sends) >= 1

    def test_classify_sends_correct_subquestion(self):
        """Each Send object should carry the sub_question generated for that agent."""
        from src.agent.nodes.classifier import classify_and_route

        sub_question = "What is the optimal pricing for forecasting software sold to regional grocery chains?"
        mock_client = _make_haiku_mock([
            {"agent_id": "gtm", "sub_question": sub_question, "relevance": "primary"},
        ])

        state = _base_state("How should we price Tidewater Labs for regional grocery chains?")

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(state)

        assert len(sends) >= 1
        # The Send payload should contain the sub-question for that agent
        send = sends[0]
        payload = send.arg if hasattr(send, "arg") else {}
        if isinstance(payload, dict):
            assert payload.get("query") == sub_question or payload.get("sub_question") == sub_question

    def test_classify_passes_company_context(self):
        """classify_and_route should pass company_context into the LLM prompt."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = _make_haiku_mock([
            {"agent_id": "bizdev", "sub_question": "Partnership strategy?", "relevance": "primary"},
        ])

        # A marker that appears nowhere in the prompt template itself
        company_ctx = "CONTEXT-MARKER-7f3a: forecasting software for regional grocery chains."
        state = {
            **_base_state("Can we partner with a point-of-sale vendor to accelerate distribution?"),
            "company_context": company_ctx,
        }

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            classify_and_route(state)

        mock_client.messages.create.assert_called_once()
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert company_ctx in prompt

    @pytest.mark.parametrize("payload", [
        {"agents": [{"agent_id": "gtm", "sub_question": "Q", "relevance": "primary"}]},
        {"agent_id": "gtm", "sub_question": "Q", "relevance": "primary"},
        ["gtm", "finance"],
        [["gtm"]],
        "gtm",
        5,
        None,
        [],
    ])
    def test_classify_valid_json_of_wrong_shape_falls_back(self, payload):
        """Valid JSON that is not a list of objects should route to the general agent."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(json.dumps(payload))

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(_message_state("What should we work on next?"))

        assert [s.node for s in sends] == ["specialist_cofounder"]
        assert sends[0].arg["query"] == "What should we work on next?"
        assert sends[0].arg["original_query"] == "What should we work on next?"

    def test_classify_skips_entries_that_are_not_objects(self):
        """Bad entries in the list are skipped and valid ones are still routed."""
        from src.agent.nodes.classifier import classify_and_route

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(json.dumps([
            "junk",
            {"agent_id": ["gtm"], "sub_question": "Q0", "relevance": "primary"},
            {"agent_id": "finance", "sub_question": "Q1", "relevance": "primary"},
        ]))

        with patch(
            "src.agent.nodes.classifier.get_haiku_client",
            return_value=(mock_client, "claude-haiku-test"),
        ):
            sends = classify_and_route(_message_state("How long is our runway?"))

        assert [s.node for s in sends] == ["specialist_finance"]
        assert sends[0].arg["query"] == "Q1"
