"""Ingest per-agent knowledge into dedicated FAISS indices.

Each specialist agent (GTM, Finance, Marketing, etc.) gets its own FAISS index
(``{agent_id}_knowledge``) populated from two sources:

1. **Agent-specific content** in ``data/knowledge/agents/{agent_id}/`` -- domain
   documents curated for that specialist.
2. **Cross-ingestion** from the shared ``unified_knowledge`` index -- relevant
   chunks (matched by keyword in ``source_name``) are copied into the specialist
   index so agents can reference general startup wisdom without querying the
   unified store at query time.

Usage::

    python scripts/ingest/ingest_agent_knowledge.py --agent gtm
    python scripts/ingest/ingest_agent_knowledge.py --agent all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import get_settings
from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import embed_texts
from src.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

AGENTS_DIR = Path("data/knowledge/agents")
UNIFIED_STORE_DIR = get_settings().vector_store_dir

VALID_AGENTS = [
    "gtm", "finance", "marketing", "fintech", "healthcare",
    "product", "legal", "data_analytics", "bizdev",
]

#: Cross-ingestion map: ``source_name`` keyword -> list of agent IDs that should
#: receive those chunks from the unified knowledge store.
CROSS_INGEST_MAP = {
    # PG essays relevant to specific agents
    "growth": ["gtm", "product"],
    "pricing": ["gtm", "finance", "bizdev"],
    "fundraising": ["finance"],
    "do-things": ["gtm", "product"],
    "startup-ideas": ["product", "cofounder"],
    "before-the-startup": ["cofounder"],
    "hiring": ["product"],
    "equity": ["finance", "legal"],
    # Books
    "100M_Offers": ["gtm", "finance", "bizdev"],
    "The_Mom_Test": ["gtm"],
    "The_Lean_Startup": ["product", "gtm"],
    "The_Hard_Thing": ["cofounder", "product"],
    # Research
    "KPMG": ["finance", "data_analytics"],
    "venture": ["finance"],
}


def ingest_agent_content(agent_id: str) -> int:
    """Ingest content from ``data/knowledge/agents/{agent_id}/`` into the agent's dedicated store.

    Performs two ingestion passes:

    1. Reads and embeds agent-specific files from the agent's content directory.
    2. Cross-ingests matching chunks from the ``unified_knowledge`` store based
       on keyword rules in :data:`CROSS_INGEST_MAP`.

    Args:
        agent_id: Identifier of the specialist agent (e.g. ``"gtm"``, ``"finance"``).

    Returns:
        Number of new agent-specific chunks added (excludes cross-ingested chunks).
    """
    agent_dir = AGENTS_DIR / agent_id
    store = VectorStore(name=f"{agent_id}_knowledge", store_dir=UNIFIED_STORE_DIR)
    processor = DocumentProcessor()

    all_chunks = []

    # 1. Ingest agent-specific content directory
    if agent_dir.exists():
        for file_path in sorted(agent_dir.iterdir()):
            if file_path.name.startswith(".") or file_path.name == ".gitkeep":
                continue
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in (".txt", ".md", ".pdf", ".html", ".htm", ".epub"):
                continue
            if store.has_source(file_path.stem):
                logger.info(f"  Skipping {file_path.name} (already ingested)")
                continue

            chunks = processor.process_file(file_path, "agent_content")
            all_chunks.extend(chunks)
            logger.info(f"  Processed {file_path.name}: {len(chunks)} chunks")
    else:
        logger.info(f"  No content directory for {agent_id}, doing cross-ingestion only")

    # 2. Cross-ingest relevant content from the unified store
    unified = VectorStore(name="unified_knowledge", store_dir=UNIFIED_STORE_DIR)
    if unified.size > 0:
        cross_source = f"cross_{agent_id}"
        if not store.has_source(cross_source):
            cross_chunks_texts = []
            cross_chunks_meta = []

            # Scan all unified_knowledge metadata; for each chunk whose source_name
            # matches a cross-ingestion keyword targeting this agent, copy it into
            # the specialist index with a synthetic source_name for dedup tracking.
            for meta in unified._metadata:
                source_name = meta.get("source_name", "")
                for keyword, target_agents in CROSS_INGEST_MAP.items():
                    if agent_id in target_agents and keyword.lower() in source_name.lower():
                        cross_chunks_texts.append(meta.get("content", ""))
                        cross_meta = dict(meta)
                        cross_meta["source_name"] = cross_source
                        cross_meta["cross_ingested_from"] = "unified_knowledge"
                        cross_chunks_meta.append(cross_meta)
                        break  # avoid duplicates per chunk

            if cross_chunks_texts:
                logger.info(f"  Cross-ingesting {len(cross_chunks_texts)} chunks from unified_knowledge")
                embeddings = embed_texts(cross_chunks_texts)
                store.add_documents(cross_chunks_texts, embeddings, cross_chunks_meta)
        else:
            logger.info(f"  Cross-ingestion already done for {agent_id}")

    # 3. Embed and store agent-specific content
    if all_chunks:
        logger.info(f"  Embedding {len(all_chunks)} agent-specific chunks...")
        texts = [c.content for c in all_chunks]
        embeddings = embed_texts(texts)
        metadata = [c.metadata for c in all_chunks]
        added = store.add_documents(texts, embeddings, metadata)
        logger.info(f"  Added {added} chunks. Store size: {store.size}")
        return added

    logger.info(f"  Store size: {store.size}")
    return 0


def main():
    """CLI entry point: parse ``--agent`` argument and run ingestion."""
    parser = argparse.ArgumentParser(description="Ingest per-agent knowledge")
    parser.add_argument(
        "--agent",
        required=True,
        choices=VALID_AGENTS + ["all"],
        help="Agent to ingest knowledge for (or 'all')",
    )
    args = parser.parse_args()

    agents = VALID_AGENTS if args.agent == "all" else [args.agent]

    total = 0
    for agent_id in agents:
        logger.info(f"Ingesting knowledge for: {agent_id}")
        added = ingest_agent_content(agent_id)
        total += added

    logger.info(f"Done. Total new chunks added: {total}")


if __name__ == "__main__":
    main()
