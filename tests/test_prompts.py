"""Tests for the prompt templates and the company name they share."""

import pytest

from src.agent.prompts import (
    CLASSIFIER_PROMPT,
    COMPANY_CONTEXT,
    COMPANY_NAME,
    LEARNING_EXTRACTOR_PROMPT,
    SPECIALIST_PROMPT,
    SYNTHESIZER_PROMPT,
)
from src.agent.prompts import company_context

PROMPTS = {
    "classifier": CLASSIFIER_PROMPT,
    "specialist": SPECIALIST_PROMPT,
    "synthesizer": SYNTHESIZER_PROMPT,
    "learning_extractor": LEARNING_EXTRACTOR_PROMPT,
}

FORMAT_ARGS = {
    "classifier": {"company_context": "CTX", "agent_descriptions": "AGENTS", "user_message": "MSG"},
    "specialist": {
        "agent_name": "NAME",
        "agent_expertise": "EXPERTISE",
        "company_context": "CTX",
        "knowledge_context": "KNOWLEDGE",
        "methodology_context": "METHOD",
        "sub_question": "SUB",
        "original_query": "ORIGINAL",
    },
    "synthesizer": {
        "company_context": "CTX",
        "knowledge_context": "KNOWLEDGE",
        "specialist_outputs": "OUTPUTS",
        "original_query": "ORIGINAL",
        "user_context": "USER",
    },
    "learning_extractor": {"conversation_content": "CONVERSATION", "company_context": "CTX"},
}


@pytest.mark.parametrize("name", list(PROMPTS))
def test_prompt_names_the_company_from_the_constant(name):
    assert COMPANY_NAME in PROMPTS[name]
    assert "[[" not in PROMPTS[name]
    assert "]]" not in PROMPTS[name]


@pytest.mark.parametrize("name", list(PROMPTS))
def test_prompt_formats_with_its_documented_fields(name):
    rendered = PROMPTS[name].format(**FORMAT_ARGS[name])
    for value in FORMAT_ARGS[name].values():
        assert value in rendered
    assert COMPANY_NAME in rendered


def test_company_context_uses_the_constant():
    assert COMPANY_CONTEXT.startswith(f"## About {COMPANY_NAME}")


def test_fill_company_replaces_markers_and_escapes_braces(monkeypatch):
    monkeypatch.setattr(company_context, "COMPANY_NAME", "Acme {Labs}")
    monkeypatch.setattr(company_context, "COMPANY_DESCRIPTION", "a maker of {widgets}")

    template = company_context.fill_company("Advising [[COMPANY_NAME]], [[COMPANY_DESCRIPTION]]. Q: {question}")

    assert template.format(question="why?") == "Advising Acme {Labs}, a maker of {widgets}. Q: why?"
