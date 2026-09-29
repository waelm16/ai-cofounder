"""Learning extraction -- runs after the response, outside the graph.

The chat endpoints call ``extract_learnings`` in a background thread once the
response has been saved. It extracts structured business learnings from the
conversation using Haiku, then persists them across multiple storage layers:

- **Business facts** -> ``TimelineEvent`` entries in the DB.
- **Decisions** -> ``Decision`` entries in the DB.
- **Hypothesis updates** -> new hypotheses or evidence on existing ones.
- **User learnings** -> profile enrichment (expertise areas, preferences).
- **Domain learnings** -> embedded into the FAISS index that the named agent
  reads, so it can retrieve them later. Agent ids not in the registry are skipped.
- **Conversation record** -> stored with embedding in the ``Conversation`` table.
  No code reads these embeddings yet.

Nothing is reviewed or deduplicated before it is written. All persistence is
wrapped in try/except, so a failed persist is logged and skipped.
"""

from __future__ import annotations

import json
import logging
import struct
from datetime import datetime

from sqlalchemy.orm import Session

from src.agent.agents.registry import AGENT_REGISTRY
from src.agent.prompts.learning_extractor import LEARNING_EXTRACTOR_PROMPT
from src.agent.state import CofounderState
from src.config.llm_config import get_haiku_client
from src.config.settings import get_settings
from src.memory.database import Conversation, get_db
from src.rag.embeddings import embed_query, embed_texts
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def extract_learnings(state: CofounderState, *, db: Session | None = None) -> dict:
    """Extract structured learnings from the conversation and persist them.

    Skips extraction for trivial messages (< 50 chars). Uses Haiku to parse
    the full conversation (user message + specialist outputs + final response)
    into structured JSON, then delegates to ``_persist_learnings``.

    Args:
        state: Full graph state after synthesis.
        db: Optional SQLAlchemy session (injected in tests; auto-created otherwise).

    Returns:
        Partial state dict with ``extracted_learnings`` (parsed JSON or ``None``).
    """
    messages = state.get("messages", [])
    agent_outputs = state.get("agent_outputs", [])
    final_response = state.get("final_response", "")

    # Skip extraction for trivial messages
    user_message = ""
    if messages:
        last = messages[-1]
        user_message = last.content if hasattr(last, "content") else str(last)
    if len(user_message) < 50:
        return {"extracted_learnings": None}

    # Build conversation content for extraction
    conversation_parts = [f"User: {user_message}"]
    for output in agent_outputs:
        conversation_parts.append(
            f"\n{output['agent_name']}: {output['response']}"
        )
    conversation_parts.append(f"\nAI Cofounder (final response): {final_response}")
    conversation_content = "\n".join(conversation_parts)

    # Call Haiku for extraction
    prompt = LEARNING_EXTRACTOR_PROMPT.format(
        conversation_content=conversation_content,
        company_context=state.get("company_context", ""),
    )

    try:
        client, model = get_haiku_client()
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        learnings = json.loads(response_text)
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning("Learning extraction JSON parse failed: %s", e)
        return {"extracted_learnings": None}
    except Exception as e:
        logger.error("Learning extraction LLM call failed: %s", e)
        return {"extracted_learnings": None}

    # Persist learnings to DB and FAISS — all wrapped in try/except
    # so extraction failures never crash the graph
    _persist_learnings(state, learnings, conversation_content, db=db)

    return {"extracted_learnings": learnings}


def _persist_learnings(
    state: CofounderState,
    learnings: dict,
    conversation_content: str,
    *,
    db: Session | None = None,
) -> None:
    """Persist extracted learnings to DB and per-agent FAISS indices.

    Handles five categories of learnings independently -- a failure in one
    category does not prevent the others from being persisted.

    Args:
        state: Full graph state (used for ``user_id`` and ``session_id``).
        learnings: Parsed JSON dict from Haiku with keys like ``business_facts``,
            ``decisions``, ``hypothesis_updates``, ``user_learnings``, and
            ``domain_learnings``.
        conversation_content: Concatenated conversation text for embedding storage.
        db: Optional SQLAlchemy session.
    """
    from src.memory.company_timeline import TimelineManager
    from src.memory.decisions_tracker import DecisionTracker
    from src.memory.hypotheses_tracker import HypothesesTracker
    from src.memory.user_profile import UserProfileManager

    own_session = db is None
    if own_session:
        db = next(get_db())
    try:
        user_id = state.get("user_id", "system")
        now = datetime.utcnow()

        # Business facts → TimelineEvent
        for fact in learnings.get("business_facts", []):
            if not fact.get("fact"):
                continue
            try:
                tl = TimelineManager(db)
                tl.add_event(
                    title=fact["fact"][:200],
                    created_by=user_id,
                    date=now,
                    event_type=f"learned_{fact.get('category', 'general')}",
                    description=fact["fact"],
                    context={"importance": fact.get("importance", "medium"), "source": "conversation"},
                )
            except Exception as e:
                logger.warning("Failed to persist business fact: %s", e)

        # Decisions → Decision
        for dec in learnings.get("decisions", []):
            if not dec.get("decision"):
                continue
            try:
                dt = DecisionTracker(db)
                dt.add_decision(
                    decision=dec["decision"],
                    proposed_by=user_id,
                    category=dec.get("category"),
                    rationales={user_id: dec.get("rationale", "")},
                )
            except Exception as e:
                logger.warning("Failed to persist decision: %s", e)

        # Hypothesis updates
        for hyp in learnings.get("hypothesis_updates", []):
            try:
                ht = HypothesesTracker(db)
                if hyp.get("hypothesis_id"):
                    # Add evidence to existing hypothesis
                    ht.add_evidence(
                        hyp["hypothesis_id"],
                        hyp.get("evidence_type", "for"),
                        {"evidence": hyp.get("evidence", ""), "source": "conversation", "date": now.isoformat()},
                    )
                elif hyp.get("new_hypothesis"):
                    ht.add_hypothesis(
                        hypothesis=hyp["new_hypothesis"],
                        owner_id=user_id,
                        auto_generated=True,
                        evidence_for=[{"evidence": hyp.get("evidence", ""), "source": "conversation"}]
                        if hyp.get("evidence_type") == "for"
                        else [],
                        evidence_against=[{"evidence": hyp.get("evidence", ""), "source": "conversation"}]
                        if hyp.get("evidence_type") == "against"
                        else [],
                    )
            except Exception as e:
                logger.warning("Failed to persist hypothesis update: %s", e)

        # User profile enrichment
        user_learnings = learnings.get("user_learnings", {})
        if user_learnings.get("expertise_update") or user_learnings.get("preference_update"):
            try:
                upm = UserProfileManager(db)
                enrichment = {}
                if user_learnings.get("expertise_update"):
                    enrichment["expertise_areas"] = user_learnings["expertise_update"]
                if user_learnings.get("preference_update"):
                    enrichment["communication_style"] = {
                        p: True for p in user_learnings["preference_update"]
                    }
                if enrichment:
                    upm.enrich_profile(user_id, enrichment)
            except Exception as e:
                logger.warning("Failed to enrich user profile: %s", e)

        # Domain-specific learnings → per-agent FAISS
        settings = get_settings()
        for dl in learnings.get("domain_learnings", []):
            if not dl.get("learning") or not dl.get("agent_id"):
                continue
            # The agent id comes from model output and decides which file is
            # written, so only ids from the registry are accepted.
            agent_config = AGENT_REGISTRY.get(dl["agent_id"]) if isinstance(dl["agent_id"], str) else None
            if agent_config is None:
                logger.warning("Skipping domain learning for unknown agent_id: %r", dl["agent_id"])
                continue
            try:
                # Write to the index this agent reads at query time
                store = VectorStore(
                    name=agent_config.knowledge_store_name,
                    store_dir=settings.vector_store_dir,
                )
                text = f"{dl['learning']} Context: {dl.get('context', '')}"
                embeddings = embed_texts([text])
                store.add_documents(
                    texts=[text],
                    embeddings=embeddings,
                    metadata_list=[{
                        "source_type": "business_learning",
                        "source_name": f"conversation_{now.strftime('%Y%m%d_%H%M%S')}",
                        "agent_id": dl["agent_id"],
                        "date": now.isoformat(),
                    }],
                )
            except Exception as e:
                logger.warning("Failed to persist domain learning for %s: %s", dl.get("agent_id"), e)

        # Store conversation with embedding
        try:
            conv_embedding = embed_query(conversation_content[:2000])
            embedding_bytes = struct.pack(f"{len(conv_embedding)}f", *conv_embedding)
            conversation = Conversation(
                user_id=user_id,
                session_id=state.get("session_id", "unknown"),
                messages=[
                    {"role": "user", "content": state.get("messages", [{}])[-1].content if state.get("messages") else ""},
                    {"role": "assistant", "content": state.get("final_response", "")},
                ],
                topics_discussed=[dl.get("agent_id") for dl in learnings.get("domain_learnings", [])],
                key_insights=[f["fact"] for f in learnings.get("business_facts", []) if f.get("importance") == "high"],
                embedding=embedding_bytes,
                started_at=now,
                ended_at=now,
            )
            db.add(conversation)
            db.commit()
        except Exception as e:
            logger.warning("Failed to store conversation: %s", e)
            db.rollback()

    except Exception as e:
        logger.error("Learning persistence failed: %s", e)
        db.rollback()
    finally:
        if own_session:
            db.close()
