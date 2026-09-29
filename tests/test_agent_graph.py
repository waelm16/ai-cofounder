"""End-to-end graph tests for the multi-agent cofounder pipeline.

All LLM calls are mocked — no real API calls are made in these tests.
The tests cover individual nodes (context_loader, specialist, synthesizer,
learning_extractor) as well as a full graph execution path.
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.memory.database import Base, User
from src.memory.hypotheses_tracker import HypothesesTracker


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic message response
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str) -> MagicMock:
    """Return a MagicMock that mimics anthropic.types.Message."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


CLASSIFIER_JSON = json.dumps([
    {
        "agent_id": "gtm",
        "sub_question": "What is the best GTM strategy for Tidewater Labs?",
        "relevance": "primary",
    },
    {
        "agent_id": "fintech",
        "sub_question": "How should Tidewater Labs handle payment data from grocery chains?",
        "relevance": "secondary",
    },
])

SPECIALIST_TEXT = "Focus on direct outbound to operations directors at regional grocery chains."

RETRIEVED_CONTENT = "Focus on direct sales to regional grocery chains."

SYNTHESIZER_TEXT = (
    "Based on GTM and fintech expertise: target operations directors at regional grocery chains "
    "with a direct outbound motion. Lead with reduced spoilage, not with features."
)

LEARNING_JSON = json.dumps({
    "business_facts": [
        {
            "fact": "Tidewater Labs should target operations directors at regional grocery chains",
            "category": "market",
            "importance": "high",
        },
    ],
    "decisions": [],
    "hypothesis_updates": [],
    "problem_evolution": {"changed": False, "new_problem": None, "new_solution": None},
    "user_learnings": {"expertise_update": [], "preference_update": []},
    "domain_learnings": [
        {
            "agent_id": "gtm",
            "learning": "Direct outbound works for operations directors",
            "context": "Grocery GTM",
        },
    ],
})


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
def mock_vector_store():
    """Patch VectorStore and embeddings where the nodes look them up.

    The nodes import these names into their own modules, so the patch has to
    target ``src.agent.nodes.*`` and not ``src.rag.*``. With this fixture no
    FAISS file is read or written and the embedding model is not loaded.
    """
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {
            "content": RETRIEVED_CONTENT,
            "source_type": "book",
            "source_name": "100m_offers",
            "title": "$100M Offers",
            "author": "Alex Hormozi",
            "score": 0.92,
            "chunk_index": 0,
        }
    ]
    mock_store.size = 5
    mock_store.add_documents.return_value = 1

    fake_vector = [0.0] * 384
    with (
        patch("src.agent.nodes.specialist.VectorStore", return_value=mock_store) as specialist_cls,
        patch("src.agent.nodes.learning_extractor.VectorStore", return_value=mock_store) as extractor_cls,
        patch("src.agent.nodes.specialist.embed_query", return_value=fake_vector),
        patch("src.agent.nodes.learning_extractor.embed_query", return_value=fake_vector),
        patch(
            "src.agent.nodes.learning_extractor.embed_texts",
            side_effect=lambda texts: [fake_vector for _ in texts],
        ),
    ):
        mock_store.specialist_cls = specialist_cls
        mock_store.extractor_cls = extractor_cls
        yield mock_store


def _base_state(user_id: str = "founder", session_id: str = "sess-001") -> dict:
    """Return a minimal valid CofounderState dict."""
    return {
        "messages": [],
        "user_id": user_id,
        "session_id": session_id,
        "user_profile": {},
        "active_hypotheses": [],
        "problem_statement": None,
        "recent_metrics": None,
        "recent_decisions": [],
        "recent_timeline": [],
        "company_context": "",

        "agent_outputs": [],
        "final_response": "",
        "sources_cited": [],
        "agents_consulted": [],
        "extracted_learnings": None,
    }


# ---------------------------------------------------------------------------
# context_loader tests
# ---------------------------------------------------------------------------


class TestContextLoader:
    def test_context_loader_loads_profile(self, db_session):
        """context_loader should populate user_profile from the DB."""
        from src.agent.nodes.context_loader import context_loader

        state = _base_state()
        result = context_loader(state, db=db_session)

        assert "user_profile" in result
        profile = result["user_profile"]
        assert profile["id"] == "founder"
        assert profile["name"] == "Founder"

    def test_context_loader_loads_hypotheses(self, db_session):
        """context_loader should populate active_hypotheses from the DB."""
        from src.agent.nodes.context_loader import context_loader

        tracker = HypothesesTracker(db_session)
        tracker.add_hypothesis(
            hypothesis="Operations directors care about spoilage cost reduction",
            owner_id="founder",
            category="market",
        )
        tracker.add_hypothesis(
            hypothesis="Store onboarding takes less than 1 day",
            owner_id="founder",
            category="product",
        )

        state = _base_state(session_id="sess-002")
        result = context_loader(state, db=db_session)

        assert "active_hypotheses" in result
        assert len(result["active_hypotheses"]) >= 2

    def test_context_loader_empty_db(self, db_session):
        """context_loader should handle a DB with only users and no extra data."""
        from src.agent.nodes.context_loader import context_loader

        state = _base_state(session_id="sess-003")
        result = context_loader(state, db=db_session)

        # Should not raise; should return safe defaults
        assert result.get("active_hypotheses", []) == []
        assert result.get("recent_decisions", []) == []
        assert result.get("recent_timeline", []) == []
        assert result.get("problem_statement") is None


# ---------------------------------------------------------------------------
# Specialist node tests
# ---------------------------------------------------------------------------


class TestSpecialistNode:
    def test_specialist_node_returns_output(self, mock_vector_store):
        """make_agent_node should create a callable that returns a valid AgentOutput."""
        from src.agent.nodes.specialist import make_agent_node

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.return_value = _make_anthropic_response(SPECIALIST_TEXT)

        with patch(
            "src.agent.nodes.specialist.get_sonnet_client",
            return_value=(mock_sonnet_client, "claude-sonnet-test"),
        ):
            node = make_agent_node("gtm")
            result = node({
                "query": "What GTM strategy works best for forecasting software?",
                "original_query": "How should we go to market?",
                "company_context": "Tidewater Labs sells inventory forecasting software to regional grocery chains.",
                "user_id": "founder",
            })

        # Result wraps output in agent_outputs list (reducer pattern)
        assert "agent_outputs" in result
        outputs = result["agent_outputs"]
        assert len(outputs) == 1
        output = outputs[0]
        assert output["agent_id"] == "gtm"
        assert output["response"] == SPECIALIST_TEXT
        assert output["confidence"] in ("high", "medium", "low")
        assert [s["source_name"] for s in output["sources"]] == ["100m_offers"]

    def test_specialist_node_searches_knowledge(self, mock_vector_store):
        """Specialist node should search its own index and put the result in the prompt."""
        from src.agent.nodes.specialist import make_agent_node

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.return_value = _make_anthropic_response(SPECIALIST_TEXT)

        with patch(
            "src.agent.nodes.specialist.get_sonnet_client",
            return_value=(mock_sonnet_client, "claude-sonnet-test"),
        ):
            node = make_agent_node("gtm")
            node({
                "query": "GTM strategy for grocery chains",
                "original_query": "How should we go to market?",
                "company_context": "Tidewater Labs targets regional grocery chains.",
                "user_id": "founder",
            })

        # The gtm agent opened its own index and searched it
        mock_vector_store.specialist_cls.assert_called_once()
        assert mock_vector_store.specialist_cls.call_args.kwargs["name"] == "gtm_knowledge"
        mock_vector_store.search.assert_called_once()

        # The retrieved passage reached the model inside the prompt
        mock_sonnet_client.messages.create.assert_called_once()
        prompt = mock_sonnet_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert RETRIEVED_CONTENT in prompt

    def test_specialist_node_different_agents(self, mock_vector_store):
        """make_agent_node should produce nodes with distinct agent_id values."""
        from src.agent.nodes.specialist import make_agent_node

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.return_value = _make_anthropic_response(SPECIALIST_TEXT)

        base_input = {
            "query": "test query",
            "original_query": "original",
            "company_context": "context",
            "user_id": "founder",
        }

        with patch(
            "src.agent.nodes.specialist.get_sonnet_client",
            return_value=(mock_sonnet_client, "claude-sonnet-test"),
        ):
            gtm_result = make_agent_node("gtm")(base_input)
            fintech_result = make_agent_node("fintech")(base_input)

        assert gtm_result["agent_outputs"][0]["agent_id"] == "gtm"
        assert fintech_result["agent_outputs"][0]["agent_id"] == "fintech"


# ---------------------------------------------------------------------------
# Synthesizer tests
# ---------------------------------------------------------------------------


class TestSynthesizer:
    def test_synthesizer_combines_outputs(self, mock_vector_store):
        """synthesize should merge multiple agent outputs into a single final_response."""
        from src.agent.nodes.synthesizer import synthesize

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.return_value = _make_anthropic_response(SYNTHESIZER_TEXT)

        state = {
            **_base_state(session_id="sess-synth-01"),
            "messages": [{"role": "user", "content": "What's our GTM strategy?"}],
            "agent_outputs": [
                {
                    "agent_id": "gtm",
                    "agent_name": "GTM Specialist",
                    "response": "Direct outbound to operations directors.",
                    "sources": [{"title": "The Mom Test"}],
                    "confidence": "high",
                },
                {
                    "agent_id": "fintech",
                    "agent_name": "Fintech Expert",
                    "response": "Clarify how payment data is handled before the pilot.",
                    "sources": [],
                    "confidence": "medium",
                },
            ],
            "company_context": "Tidewater Labs: inventory forecasting for regional grocery chains.",
        }

        with patch(
            "src.agent.nodes.synthesizer.get_sonnet_client",
            return_value=(mock_sonnet_client, "claude-sonnet-test"),
        ):
            result = synthesize(state)

        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "agents_consulted" in result
        assert set(result["agents_consulted"]) == {"gtm", "fintech"}

    def test_synthesizer_single_agent(self, mock_vector_store):
        """synthesize should work with a single agent output."""
        from src.agent.nodes.synthesizer import synthesize

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.return_value = _make_anthropic_response(
            "Single agent synthesis result."
        )

        state = {
            **_base_state(session_id="sess-synth-02"),
            "messages": [{"role": "user", "content": "Finance question?"}],
            "agent_outputs": [
                {
                    "agent_id": "finance",
                    "agent_name": "Finance & Sales",
                    "response": "Aim for 18 months runway before Series A.",
                    "sources": [],
                    "confidence": "high",
                },
            ],
            "company_context": "Tidewater Labs context.",
        }

        with patch(
            "src.agent.nodes.synthesizer.get_sonnet_client",
            return_value=(mock_sonnet_client, "claude-sonnet-test"),
        ):
            result = synthesize(state)

        assert result["final_response"] != ""
        assert "finance" in result["agents_consulted"]


# ---------------------------------------------------------------------------
# Learning extractor tests
# ---------------------------------------------------------------------------


class TestLearningExtractor:
    def test_learning_extractor_persists(self, db_session, mock_vector_store):
        """extract_learnings should call Haiku and persist what it returns."""
        from src.agent.nodes.learning_extractor import extract_learnings

        mock_haiku_client = MagicMock()
        mock_haiku_client.messages.create.return_value = _make_anthropic_response(LEARNING_JSON)

        state = {
            **_base_state(session_id="sess-learn-01"),
            "messages": [
                {"role": "user", "content": "How should we target grocery chains?"},
                {"role": "assistant", "content": SYNTHESIZER_TEXT},
            ],
            "agent_outputs": [
                {
                    "agent_id": "gtm",
                    "agent_name": "GTM Specialist",
                    "response": "Target operations directors directly.",
                    "sources": [],
                    "confidence": "high",
                },
            ],
            "final_response": SYNTHESIZER_TEXT,
            "agents_consulted": ["gtm"],
        }

        with patch(
            "src.agent.nodes.learning_extractor.get_haiku_client",
            return_value=(mock_haiku_client, "claude-haiku-test"),
        ):
            result = extract_learnings(state, db=db_session)

        assert result["extracted_learnings"] == json.loads(LEARNING_JSON)

        # The domain learning went to the gtm agent's index
        assert mock_vector_store.extractor_cls.call_args.kwargs["name"] == "gtm_knowledge"
        mock_vector_store.add_documents.assert_called_once()
        written = mock_vector_store.add_documents.call_args.kwargs
        assert written["texts"] == ["Direct outbound works for operations directors Context: Grocery GTM"]
        assert written["metadata_list"][0]["source_type"] == "business_learning"
        assert written["metadata_list"][0]["agent_id"] == "gtm"

        # The business fact went to the timeline
        from src.memory.database import TimelineEvent

        events = db_session.query(TimelineEvent).all()
        assert [e.description for e in events] == [
            "Tidewater Labs should target operations directors at regional grocery chains"
        ]

    @pytest.mark.parametrize("agent_id, expected_index", [
        ("gtm", "gtm_knowledge"),
        ("marketing", "marketing_knowledge"),
        ("devils_advocate", "unified_knowledge"),
        ("cofounder", "unified_knowledge"),
    ])
    def test_domain_learning_goes_to_the_index_the_agent_reads(
        self, db_session, mock_vector_store, agent_id, expected_index
    ):
        from src.agent.agents.registry import AGENT_REGISTRY
        from src.agent.nodes.learning_extractor import _persist_learnings

        learnings = {"domain_learnings": [{"agent_id": agent_id, "learning": "L", "context": "C"}]}
        _persist_learnings(_base_state(), learnings, "conversation text", db=db_session)

        assert mock_vector_store.extractor_cls.call_args.kwargs["name"] == expected_index
        assert expected_index == AGENT_REGISTRY[agent_id].knowledge_store_name
        mock_vector_store.add_documents.assert_called_once()

    @pytest.mark.parametrize("agent_id", ["not_an_agent", "../../outside", "", None, ["gtm"], 7])
    def test_domain_learning_for_unknown_agent_is_skipped(self, db_session, mock_vector_store, agent_id):
        from src.agent.nodes.learning_extractor import _persist_learnings

        learnings = {"domain_learnings": [{"agent_id": agent_id, "learning": "L", "context": "C"}]}
        _persist_learnings(_base_state(), learnings, "conversation text", db=db_session)

        mock_vector_store.extractor_cls.assert_not_called()
        mock_vector_store.add_documents.assert_not_called()

    def test_learning_extractor_returns_structured_output(self, db_session, mock_vector_store):
        """extract_learnings should parse Haiku JSON into a structured dict."""
        from src.agent.nodes.learning_extractor import extract_learnings

        mock_haiku_client = MagicMock()
        mock_haiku_client.messages.create.return_value = _make_anthropic_response(LEARNING_JSON)

        state = {
            **_base_state(session_id="sess-learn-02"),
            "messages": [{"role": "user", "content": "Any market insights?"}],
            "agent_outputs": [],
            "final_response": "Grocery chains are reviewing suppliers.",
            "agents_consulted": [],
        }

        with patch(
            "src.agent.nodes.learning_extractor.get_haiku_client",
            return_value=(mock_haiku_client, "claude-haiku-test"),
        ):
            result = extract_learnings(state, db=db_session)

        learnings = result["extracted_learnings"]
        assert isinstance(learnings, dict)
        assert any(
            k in learnings
            for k in ("business_facts", "decisions", "hypothesis_updates", "domain_learnings")
        )


# ---------------------------------------------------------------------------
# Full graph execution tests
# ---------------------------------------------------------------------------


class TestFullGraph:
    def test_full_graph_execution(self, db_session, mock_vector_store):
        """Invoke the full compiled LangGraph with mocked LLM calls.

        Verifies that after graph.invoke() the state contains a non-empty
        final_response and a populated agents_consulted list.
        """
        from src.agent.graph import graph

        mock_haiku_client = MagicMock()
        mock_haiku_client.messages.create.side_effect = [
            _make_anthropic_response(CLASSIFIER_JSON),  # classify_and_route
        ]

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.side_effect = [
            _make_anthropic_response(SPECIALIST_TEXT),   # gtm specialist
            _make_anthropic_response(SPECIALIST_TEXT),   # fintech specialist
            _make_anthropic_response(SYNTHESIZER_TEXT),  # synthesizer
        ]

        initial_state = {
            **_base_state(session_id=str(uuid.uuid4())),
            "messages": [{"role": "user", "content": "What is our go-to-market strategy for grocers?"}],
        }

        with (
            patch(
                "src.agent.nodes.classifier.get_haiku_client",
                return_value=(mock_haiku_client, "claude-haiku-test"),
            ),
            patch(
                "src.agent.nodes.specialist.get_sonnet_client",
                return_value=(mock_sonnet_client, "claude-sonnet-test"),
            ),
            patch(
                "src.agent.nodes.synthesizer.get_sonnet_client",
                return_value=(mock_sonnet_client, "claude-sonnet-test"),
            ),
            patch("src.agent.nodes.context_loader.get_db", return_value=iter([db_session])),
        ):
            final_state = graph.invoke(initial_state)

        assert final_state["final_response"] == SYNTHESIZER_TEXT
        assert set(final_state["agents_consulted"]) == {"gtm", "fintech"}
        assert {o["agent_id"] for o in final_state["agent_outputs"]} == {"gtm", "fintech"}

        # One small-model call (classification) and three large-model calls
        # (two specialists, one synthesis). Learning extraction is not a graph node.
        assert mock_haiku_client.messages.create.call_count == 1
        assert mock_sonnet_client.messages.create.call_count == 3
        assert final_state.get("extracted_learnings") is None
        mock_vector_store.add_documents.assert_not_called()

    def test_full_graph_sets_session_id(self, db_session, mock_vector_store):
        """Graph should preserve the session_id provided in the initial state."""
        from src.agent.graph import graph

        session_id = "test-session-abc-123"

        mock_haiku_client = MagicMock()
        mock_haiku_client.messages.create.side_effect = [
            _make_anthropic_response(
                json.dumps([{"agent_id": "gtm", "sub_question": "GTM?", "relevance": "primary"}])
            ),
        ]

        mock_sonnet_client = MagicMock()
        mock_sonnet_client.messages.create.side_effect = [
            _make_anthropic_response(SPECIALIST_TEXT),
            _make_anthropic_response(SYNTHESIZER_TEXT),
        ]

        initial_state = {
            **_base_state(session_id=session_id),
            "messages": [{"role": "user", "content": "Simple question"}],
        }

        with (
            patch(
                "src.agent.nodes.classifier.get_haiku_client",
                return_value=(mock_haiku_client, "claude-haiku-test"),
            ),
            patch(
                "src.agent.nodes.specialist.get_sonnet_client",
                return_value=(mock_sonnet_client, "claude-sonnet-test"),
            ),
            patch(
                "src.agent.nodes.synthesizer.get_sonnet_client",
                return_value=(mock_sonnet_client, "claude-sonnet-test"),
            ),
            patch("src.agent.nodes.context_loader.get_db", return_value=iter([db_session])),
        ):
            final_state = graph.invoke(initial_state)

        assert final_state.get("session_id") == session_id
