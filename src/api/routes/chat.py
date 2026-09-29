"""Multi-agent chat API routes.

The primary conversational interface to the AI cofounder team.  A user
message is fed into the LangGraph workflow which loads context, classifies
the query, dispatches to specialist agents in parallel, and synthesises a
unified response.  Both the user and assistant messages plus individual agent
consultations are persisted for audit and session replay.  Learning
extraction then runs in a background thread.

Endpoints:
    POST /api/chat/                        — send a message, get a response
    POST /api/chat/stream                  — same pipeline over Server-Sent Events
    GET  /api/chat/sessions/               — list the user's sessions
    GET  /api/chat/sessions/{session_id}   — retrieve session history
"""

import json
import logging
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from src.agent.graph import graph, graph_pre_synthesis
from src.agent.nodes.synthesizer import build_synthesis_prompt
from src.agent.nodes.learning_extractor import extract_learnings

logger = logging.getLogger(__name__)
from src.api.auth import get_current_user
from src.api.schemas import (
    AgentOutputResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
    ChatSessionSummary,
)
from src.memory.database import AgentConsultation, ChatMessage, SessionLocal, User, get_db

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _run_learning_extraction_background(result: dict) -> None:
    """Run learning extraction in a background thread so the user gets their response faster."""
    try:
        extract_learnings(result)
    except Exception as e:
        logger.warning("Background learning extraction failed: %s", e)


@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a message to the AI cofounder team and receive a synthesised response.

    If no ``session_id`` is provided, a new UUID session is created.  The
    LangGraph graph is invoked synchronously and the full conversation turn
    (user message, assistant response, and per-agent consultations) is
    persisted to the database.

    Args:
        request: The user's message and optional session ID.

    Returns:
        ChatResponse: Synthesised response, consulted agents, sources,
            and any learnings extracted from the conversation.

    Raises:
        HTTPException: 500 if the LangGraph execution fails.
    """
    session_id = request.session_id or str(uuid.uuid4())

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": current_user.id,
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

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    # Persist the user message
    user_msg = ChatMessage(
        session_id=session_id,
        user_id=current_user.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.flush()

    # Persist the assistant response
    assistant_msg = ChatMessage(
        session_id=session_id,
        user_id=current_user.id,
        role="assistant",
        content=result.get("final_response", ""),
        agents_consulted=result.get("agents_consulted", []),
        sources_cited=result.get("sources_cited", []),
    )
    db.add(assistant_msg)
    db.flush()

    # Persist each specialist agent's contribution for audit trail
    for output in result.get("agent_outputs", []):
        consultation = AgentConsultation(
            session_id=session_id,
            chat_message_id=assistant_msg.id,
            agent_id=output["agent_id"],
            agent_name=output["agent_name"],
            response=output["response"],
            confidence=output.get("confidence"),
            sources=output.get("sources"),
        )
        db.add(consultation)

    db.commit()

    # Run learning extraction in background — user gets response immediately
    threading.Thread(
        target=_run_learning_extraction_background,
        args=(result,),
        daemon=True,
    ).start()

    return ChatResponse(
        response=result.get("final_response", ""),
        session_id=session_id,
        agents_consulted=result.get("agents_consulted", []),
        sources=result.get("sources_cited", []),
        agent_outputs=[
            AgentOutputResponse(
                agent_id=o["agent_id"],
                agent_name=o["agent_name"],
                response=o["response"],
                sources=o.get("sources", []),
                confidence=o.get("confidence"),
            )
            for o in result.get("agent_outputs", [])
        ],
        learnings_extracted=None,
    )


@router.post("/stream")
async def chat_stream(request: Request):
    """Stream the multi-agent pipeline via Server-Sent Events.

    Phase 1: Runs the pre-synthesis graph with ``stream_mode="values"``, which
    yields one state snapshot per superstep. Status updates are sent per
    snapshot, so specialists that ran in parallel are reported together.

    Phase 2: Streams the synthesis Sonnet call token-by-token. Synthesis runs
    here, outside the graph, because it calls the Anthropic SDK directly.

    SSE event types:
        status  — progress update (e.g. "Received input from GTM Strategist...")
        token   — individual text chunk from the synthesis stream
        done    — final metadata (agents consulted, sources, session_id)
        error   — pipeline failure
    """
    from src.config.llm_config import get_sonnet_client

    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())
    user_id = request.headers.get("Authorization", "").replace("Bearer ", "").strip()

    if not message:
        return StreamingResponse(
            iter([f"event: error\ndata: {json.dumps({'detail': 'Empty message'})}\n\n"]),
            media_type="text/event-stream",
        )

    # Verify user
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return StreamingResponse(
                iter([f"event: error\ndata: {json.dumps({'detail': 'Invalid user'})}\n\n"]),
                media_type="text/event-stream",
            )
    finally:
        db.close()

    initial_state = {
        "messages": [HumanMessage(content=message)],
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

    def event_generator():
        # --- Phase 1: Run pre-synthesis graph with progress updates ---
        collected_state = None
        prev_output_count = 0
        try:
            for state_snapshot in graph_pre_synthesis.stream(initial_state, stream_mode="values"):
                collected_state = state_snapshot
                agent_outputs = state_snapshot.get("agent_outputs", [])

                if len(agent_outputs) > prev_output_count:
                    new_agents = agent_outputs[prev_output_count:]
                    for o in new_agents:
                        agent_name = o["agent_name"]
                        yield f"event: status\ndata: {json.dumps({'status': f'Received input from {agent_name}...'})}\n\n"
                    prev_output_count = len(agent_outputs)
                elif state_snapshot.get("company_context") and not agent_outputs:
                    yield f"event: status\ndata: {json.dumps({'status': 'Routing to specialist agents...'})}\n\n"
                elif not state_snapshot.get("company_context"):
                    yield f"event: status\ndata: {json.dumps({'status': 'Loading company context...'})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return

        if not collected_state or not collected_state.get("agent_outputs"):
            yield f"event: error\ndata: {json.dumps({'detail': 'No specialist outputs received'})}\n\n"
            return

        # --- Phase 2: Stream synthesis token-by-token ---
        yield f"event: status\ndata: {json.dumps({'status': 'Synthesizing response...'})}\n\n"

        prompt, all_sources, agents_consulted = build_synthesis_prompt(collected_state)
        full_response = []

        try:
            client, model = get_sonnet_client()
            with client.messages.stream(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    full_response.append(text)
                    yield f"event: token\ndata: {json.dumps({'token': text})}\n\n"
        except Exception as e:
            # Fallback: concatenate raw specialist responses
            fallback = "I encountered an error during synthesis. Here are the raw specialist inputs:\n\n"
            for output in collected_state.get("agent_outputs", []):
                fallback += f"**{output['agent_name']}**: {output['response']}\n\n"
            full_response = [fallback]
            yield f"event: token\ndata: {json.dumps({'token': fallback})}\n\n"

        final_response = "".join(full_response)

        # --- Persist to DB ---
        persist_db = SessionLocal()
        try:
            user_msg = ChatMessage(
                session_id=session_id, user_id=user_id, role="user", content=message,
            )
            persist_db.add(user_msg)
            persist_db.flush()

            assistant_msg = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=final_response,
                agents_consulted=agents_consulted,
                sources_cited=all_sources,
            )
            persist_db.add(assistant_msg)
            persist_db.flush()

            for output in collected_state.get("agent_outputs", []):
                persist_db.add(AgentConsultation(
                    session_id=session_id,
                    chat_message_id=assistant_msg.id,
                    agent_id=output["agent_id"],
                    agent_name=output["agent_name"],
                    response=output["response"],
                    confidence=output.get("confidence"),
                    sources=output.get("sources"),
                ))
            persist_db.commit()
        except Exception:
            persist_db.rollback()
        finally:
            persist_db.close()

        # Run learning extraction in background
        result_for_learnings = dict(collected_state)
        result_for_learnings["final_response"] = final_response
        threading.Thread(
            target=_run_learning_extraction_background,
            args=(result_for_learnings,),
            daemon=True,
        ).start()

        # --- Send final metadata ---
        done_data = {
            "session_id": session_id,
            "agents_consulted": agents_consulted,
            "sources": all_sources,
            "agent_outputs": [
                {
                    "agent_id": o["agent_id"],
                    "agent_name": o["agent_name"],
                    "response": o["response"],
                    "sources": o.get("sources", []),
                    "confidence": o.get("confidence"),
                }
                for o in collected_state.get("agent_outputs", [])
            ],
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/sessions/", response_model=list[ChatSessionSummary])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all chat sessions for the current user, most recent first.

    Returns a summary for each session including the first user message,
    message count, and which agents were consulted.
    """
    from sqlalchemy import func

    # Get distinct session IDs with their first message and count
    sessions = (
        db.query(
            ChatMessage.session_id,
            func.min(ChatMessage.created_at).label("created_at"),
            func.count(ChatMessage.id).label("message_count"),
        )
        .filter(ChatMessage.user_id == current_user.id)
        .group_by(ChatMessage.session_id)
        .order_by(func.min(ChatMessage.created_at).desc())
        .all()
    )

    result = []
    for sess in sessions:
        # Get the first user message for preview
        first_msg = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.session_id == sess.session_id,
                ChatMessage.role == "user",
            )
            .order_by(ChatMessage.created_at.asc())
            .first()
        )

        # Collect all agents consulted across the session
        agents = set()
        agent_msgs = (
            db.query(ChatMessage.agents_consulted)
            .filter(
                ChatMessage.session_id == sess.session_id,
                ChatMessage.agents_consulted.isnot(None),
            )
            .all()
        )
        for (consulted,) in agent_msgs:
            if consulted:
                agents.update(consulted)

        result.append(ChatSessionSummary(
            session_id=sess.session_id,
            first_message=first_msg.content[:100] if first_msg else "New session",
            message_count=sess.message_count,
            agents_consulted=list(agents),
            created_at=str(sess.created_at) if sess.created_at else None,
        ))

    return result


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the full conversation history for a chat session.

    Messages are scoped to the authenticated user and returned in
    chronological order.  The union of all agents consulted across the
    session is included.

    Args:
        session_id: UUID of the chat session.

    Returns:
        ChatSessionResponse: Ordered messages and agents consulted.

    Raises:
        HTTPException: 404 if no messages exist for the given session.
    """
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == current_user.id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")

    agents = set()
    msg_list = []
    for msg in messages:
        msg_list.append({
            "role": msg.role,
            "content": msg.content,
            "created_at": str(msg.created_at) if msg.created_at else None,
        })
        if msg.agents_consulted:
            agents.update(msg.agents_consulted)

    return ChatSessionResponse(
        session_id=session_id,
        messages=msg_list,
        agents_consulted=list(agents),
    )
