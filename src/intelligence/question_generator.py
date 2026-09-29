"""Pre-call question generator using Mom Test principles via RAG + Sonnet.

Before a customer discovery call, this module generates interview questions
grounded in Mom Test methodology.  It retrieves relevant passages from the
knowledge base (particularly "The Mom Test" book) and feeds them alongside
the current problem statement and active hypotheses to Sonnet, which produces
structured, actionable questions designed to elicit honest customer feedback.
"""

import json
import logging

from anthropic import Anthropic

from src.config.settings import get_settings
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def get_sonnet_client():
    """Create an Anthropic client configured for Sonnet.

    Returns:
        Tuple of (Anthropic client instance, Sonnet model ID string).
    """
    settings = get_settings()
    return Anthropic(api_key=settings.anthropic_api_key), settings.sonnet_model


def generate_pre_call_questions(
    customer_name: str,
    customer_company: str | None = None,
    customer_role: str | None = None,
    customer_context: dict | None = None,
    active_hypotheses: list[dict] | None = None,
    problem_statement: dict | None = None,
) -> list[dict]:
    """Generate Mom Test-style pre-call questions using Sonnet with RAG context.

    Retrieves Mom Test principles from the knowledge base, combines them with
    the customer profile and active hypotheses, and asks Sonnet to produce
    5-7 interview questions that follow Mom Test methodology (asking about
    past behavior, not opinions about the future).

    Args:
        customer_name: Name of the person being interviewed.
        customer_company: Their company name (optional).
        customer_role: Their job title or role (optional).
        customer_context: Additional context dict about the customer (optional).
        active_hypotheses: List of hypothesis dicts to validate during the call.
        problem_statement: Current problem/solution/target-customer dict.

    Returns:
        List of dicts, each with keys ``question``, ``rationale``, and
        ``targeted_hypothesis_id`` (int or None).
    """
    # Retrieve Mom Test principles from RAG
    mom_test_results = retrieve("Mom Test principles customer interview questions", top_k=5, source_type="book")
    mom_test_context = "\n".join(r["content"] for r in mom_test_results) if mom_test_results else ""

    # Build context
    customer_info = f"Customer: {customer_name}"
    if customer_company:
        customer_info += f", Company: {customer_company}"
    if customer_role:
        customer_info += f", Role: {customer_role}"
    if customer_context:
        customer_info += f"\nAdditional context: {json.dumps(customer_context)}"

    hypotheses_text = ""
    if active_hypotheses:
        hypotheses_text = "\n".join(
            f"- [H{h.get('id', '?')}] {h.get('hypothesis', '')}" for h in active_hypotheses
        )

    problem_text = ""
    if problem_statement:
        problem_text = (
            f"Problem: {problem_statement.get('problem', '')}\n"
            f"Solution: {problem_statement.get('solution', '')}\n"
            f"Target customer: {problem_statement.get('target_customer', '')}"
        )

    prompt = f"""You are an expert customer discovery interviewer trained in the Mom Test methodology.

{f"Mom Test principles from our knowledge base:{chr(10)}{mom_test_context}" if mom_test_context else "Apply Mom Test principles: ask about their life, not your idea. Ask about specifics in the past, not generics or opinions about the future."}

{f"Our problem statement:{chr(10)}{problem_text}" if problem_text else ""}

{f"Active hypotheses to validate:{chr(10)}{hypotheses_text}" if hypotheses_text else ""}

{customer_info}

Generate 5-7 pre-call questions following Mom Test principles:
- Ask about their actual behavior, not opinions
- Ask about specifics in the past, not generics about the future
- Talk less, listen more
- Push past compliments and fluff
- If targeting a specific hypothesis, note which one

Return valid JSON array where each element has:
- "question": the interview question
- "rationale": why this question matters (1 sentence)
- "targeted_hypothesis_id": hypothesis ID this targets (integer or null)

Return ONLY the JSON array, no other text."""

    client, model = get_sonnet_client()
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences that Sonnet sometimes wraps around JSON output
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)
