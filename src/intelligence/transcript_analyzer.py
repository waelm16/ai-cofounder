"""Post-call transcript analyzer using Haiku for structured extraction.

After a customer discovery call, this module sends the full transcript to
Haiku (chosen for cost efficiency on extraction tasks) and extracts structured
insights: key takeaways, pain points with severity, sentiment, willingness-to-pay
signals, next steps, new hypothesis candidates, and evidence for/against
existing hypotheses.
"""

import json
import logging

from anthropic import Anthropic

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_haiku_client():
    """Create an Anthropic client configured for Haiku.

    Returns:
        Tuple of (Anthropic client instance, Haiku model ID string).
    """
    settings = get_settings()
    return Anthropic(api_key=settings.anthropic_api_key), settings.haiku_model


def analyze_transcript(
    transcript: str,
    customer_name: str,
    customer_company: str | None = None,
    active_hypotheses: list[dict] | None = None,
) -> dict:
    """Analyze a customer call transcript using Haiku and return structured insights.

    The LLM extracts multiple categories of information in a single pass,
    mapping evidence back to existing hypotheses by ID where applicable.

    Args:
        transcript: Full text of the call transcript.
        customer_name: Name of the customer on the call.
        customer_company: Customer's company name (optional).
        active_hypotheses: List of hypothesis dicts (with ``id`` and
            ``hypothesis`` keys) to match evidence against.

    Returns:
        Dict with keys: ``key_insights``, ``pain_points``, ``sentiment``,
        ``wtp_signals``, ``next_steps``, ``new_hypotheses``,
        ``evidence_for_existing``.
    """
    hypotheses_text = ""
    if active_hypotheses:
        hypotheses_text = "\n".join(
            f"- [H{h.get('id', '?')}] {h.get('hypothesis', '')}" for h in active_hypotheses
        )

    prompt = f"""Analyze this customer call transcript and extract structured insights.

Customer: {customer_name}{f", Company: {customer_company}" if customer_company else ""}

{f"Active hypotheses (reference by ID when mapping evidence):{chr(10)}{hypotheses_text}" if hypotheses_text else ""}

TRANSCRIPT:
{transcript}

Extract the following and return as valid JSON:
{{
  "key_insights": [
    {{"insight": "...", "quote": "relevant quote from transcript", "importance": "high|medium|low"}}
  ],
  "pain_points": [
    {{"pain_point": "...", "severity": "high|medium|low", "quote": "..."}}
  ],
  "sentiment": "positive|neutral|negative|mixed",
  "wtp_signals": [
    {{"signal": "...", "strength": "strong|moderate|weak", "quote": "..."}}
  ],
  "next_steps": [
    {{"action": "...", "owner": "us|customer|both", "deadline": "..."}}
  ],
  "new_hypotheses": [
    {{"hypothesis": "...", "category": "problem|solution|market|customer", "evidence": "..."}}
  ],
  "evidence_for_existing": [
    {{"hypothesis_id": 123, "evidence_type": "for|against", "quote": "...", "source": "..."}}
  ]
}}

Return ONLY the JSON object, no other text."""

    client, model = get_haiku_client()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences that Haiku sometimes wraps around JSON output
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text)
