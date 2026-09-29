"""Prompt templates for the multi-agent cofounder pipeline.

Each prompt is a format-string constant used by its corresponding graph node.
The prompts are designed for tiered LLM usage:

- ``CLASSIFIER_PROMPT`` -- consumed by Haiku for cheap/fast routing.
- ``SPECIALIST_PROMPT`` -- consumed by Sonnet for domain-specific reasoning.
- ``SYNTHESIZER_PROMPT`` -- consumed by Sonnet for merging specialist outputs.
- ``LEARNING_EXTRACTOR_PROMPT`` -- consumed by Haiku for structured extraction.
- ``COMPANY_CONTEXT`` -- static company context injected into all prompts.
- ``COMPANY_NAME`` -- the company name, inserted into the four templates above.
"""

from src.agent.prompts.classifier import CLASSIFIER_PROMPT
from src.agent.prompts.company_context import COMPANY_CONTEXT, COMPANY_NAME
from src.agent.prompts.learning_extractor import LEARNING_EXTRACTOR_PROMPT
from src.agent.prompts.specialist import SPECIALIST_PROMPT
from src.agent.prompts.synthesizer import SYNTHESIZER_PROMPT

__all__ = [
    "CLASSIFIER_PROMPT",
    "LEARNING_EXTRACTOR_PROMPT",
    "SPECIALIST_PROMPT",
    "SYNTHESIZER_PROMPT",
    "COMPANY_CONTEXT",
    "COMPANY_NAME",
]
