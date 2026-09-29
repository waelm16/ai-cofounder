"""Integration tests for classifier -> specialist routing pipeline.

Verifies that AgentInput carries source_hints, that devils_advocate is
registered, that SPECIALIST_PROMPT includes the methodology placeholder,
and that the knowledge router returns correct MethodologyMatch objects.
"""

from src.agent.agents.registry import AGENT_REGISTRY
from src.agent.knowledge_router import MethodologyMatch, route_from_hints, route_query
from src.agent.prompts.specialist import SPECIALIST_PROMPT
from src.agent.state import AgentInput


class TestAgentInputSourceHints:
    """AgentInput TypedDict must include source_hints."""

    def test_source_hints_is_valid_key(self):
        inp = AgentInput(
            query="test",
            original_query="test",
            company_context="",
            user_id="u1",
            source_hints=["mom_test"],
        )
        assert "source_hints" in inp
        assert inp["source_hints"] == ["mom_test"]

    def test_source_hints_empty_list(self):
        inp = AgentInput(
            query="q",
            original_query="q",
            company_context="",
            user_id="u1",
            source_hints=[],
        )
        assert inp["source_hints"] == []


class TestDevilsAdvocateRegistry:
    """devils_advocate must be in the registry with correct config."""

    def test_devils_advocate_present(self):
        assert "devils_advocate" in AGENT_REGISTRY

    def test_devils_advocate_knowledge_store(self):
        cfg = AGENT_REGISTRY["devils_advocate"]
        assert cfg.knowledge_store_name == "unified_knowledge"

    def test_devils_advocate_has_name(self):
        cfg = AGENT_REGISTRY["devils_advocate"]
        assert cfg.name == "Devil's Advocate"

    def test_devils_advocate_llm_tier(self):
        cfg = AGENT_REGISTRY["devils_advocate"]
        assert cfg.llm_tier == "sonnet"


class TestSpecialistPromptTemplate:
    """SPECIALIST_PROMPT must have the methodology_context placeholder."""

    def test_methodology_context_placeholder(self):
        assert "{methodology_context}" in SPECIALIST_PROMPT

    def test_all_required_placeholders(self):
        for placeholder in [
            "{agent_name}",
            "{agent_expertise}",
            "{company_context}",
            "{knowledge_context}",
            "{methodology_context}",
            "{sub_question}",
            "{original_query}",
        ]:
            assert placeholder in SPECIALIST_PROMPT, f"Missing {placeholder}"

    def test_prompt_renders_without_error(self):
        rendered = SPECIALIST_PROMPT.format(
            agent_name="Test Agent",
            agent_expertise="Testing",
            company_context="Tidewater Labs context",
            knowledge_context="Some knowledge",
            methodology_context="Apply Mom Test",
            sub_question="How to validate?",
            original_query="Help me validate",
        )
        assert "Test Agent" in rendered
        assert "Apply Mom Test" in rendered


class TestRegistryNoDeadTemplate:
    """_DEFAULT_TEMPLATE was removed — specialist.py uses SPECIALIST_PROMPT directly."""

    def test_no_default_template_in_registry(self):
        import src.agent.agents.registry as reg

        assert not hasattr(reg, "_DEFAULT_TEMPLATE")


class TestKnowledgeRouterIntegration:
    """route_query and route_from_hints return correct MethodologyMatch objects."""

    def test_route_query_mom_test(self):
        match = route_query("mom test questions for discovery")
        assert match is not None
        assert isinstance(match, MethodologyMatch)
        assert match.methodology_id == "mom_test"

    def test_route_query_original_query_fallback(self):
        # Keyword in original_query but not in query
        match = route_query("prepare questions", "help with mom test")
        assert match is not None
        assert match.methodology_id == "mom_test"

    def test_route_query_garbage_returns_none(self):
        match = route_query("xyzzy flurble nonsense")
        assert match is None

    def test_route_query_crossing_the_chasm(self):
        match = route_query("crossing the chasm strategy for grocery chains")
        assert match is not None
        assert match.methodology_id == "crossing_the_chasm"

    def test_route_from_hints_single(self):
        matches = route_from_hints(["mom_test"])
        assert len(matches) == 1
        assert matches[0].methodology_id == "mom_test"

    def test_route_from_hints_multiple(self):
        matches = route_from_hints(["mom_test", "lean_startup"])
        assert len(matches) == 2
        ids = {m.methodology_id for m in matches}
        assert ids == {"mom_test", "lean_startup"}

    def test_route_from_hints_unknown_skipped(self):
        matches = route_from_hints(["mom_test", "nonexistent_thing"])
        assert len(matches) == 1
        assert matches[0].methodology_id == "mom_test"

    def test_route_from_hints_empty(self):
        matches = route_from_hints([])
        assert matches == []


class TestGraphAutoRegisters:
    """Verify graph.py creates nodes for all registry entries including devils_advocate."""

    def test_graph_has_devils_advocate_node(self):
        from src.agent.graph import graph

        # LangGraph compiled graph exposes node names
        node_names = set(graph.get_graph().nodes.keys())
        assert "specialist_devils_advocate" in node_names

    def test_graph_has_all_specialist_nodes(self):
        from src.agent.graph import graph

        node_names = set(graph.get_graph().nodes.keys())
        for agent_id in AGENT_REGISTRY:
            expected = f"specialist_{agent_id}"
            assert expected in node_names, f"Missing node: {expected}"
