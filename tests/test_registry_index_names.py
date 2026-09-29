"""Tests that each agent reads the FAISS index that ingestion writes for it."""

import importlib.util
from pathlib import Path

import pytest

from src.agent.agents.registry import AGENT_REGISTRY

INGEST_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "ingest" / "ingest_agent_knowledge.py"
)


def _load_ingest_module():
    """Import the ingest script from its file path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("ingest_agent_knowledge", INGEST_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingStore:
    """Stand-in for VectorStore that records which index names are opened."""

    opened: list[str] = []

    def __init__(self, name: str = "unified_knowledge", **kwargs):
        type(self).opened.append(name)
        self.size = 0
        self._metadata = []

    def has_source(self, source_name: str) -> bool:
        return False

    def add_documents(self, texts, embeddings, metadata_list) -> int:
        return 0


@pytest.mark.parametrize("agent_id", list(AGENT_REGISTRY.keys()))
def test_registry_index_matches_ingest(agent_id, tmp_path, monkeypatch):
    """The index an agent reads must be one that an ingest script writes."""
    ingest = _load_ingest_module()
    expected = AGENT_REGISTRY[agent_id].knowledge_store_name

    if agent_id not in ingest.VALID_AGENTS:
        # Agents without a dedicated index read the shared one.
        assert expected == "unified_knowledge"
        return

    _RecordingStore.opened = []
    monkeypatch.setattr(ingest, "VectorStore", _RecordingStore)
    monkeypatch.setattr(ingest, "AGENTS_DIR", tmp_path)

    ingest.ingest_agent_content(agent_id)

    written = [name for name in _RecordingStore.opened if name != "unified_knowledge"]
    assert written == [expected]
