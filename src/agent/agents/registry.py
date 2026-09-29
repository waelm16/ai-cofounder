"""Agent configuration registry -- single source of truth for all specialist agents.

Defines ``AgentConfig`` and the ``AGENT_REGISTRY`` dict that the graph builder,
specialist factory, and classifier all read from. Adding a new specialist agent
requires only a new entry here plus curated content in its knowledge directory --
no new node code is needed thanks to the factory pattern in ``specialist.py``.
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for a single specialist agent.

    Args:
        agent_id: Unique key used in routing, node naming, and FAISS index lookup.
        name: Human-readable display name (e.g. ``"GTM Strategist"``).
        description: Capabilities summary shown to the classifier for routing decisions.
        knowledge_store_name: FAISS index name under ``vector_stores/`` (without ``.faiss``).
        llm_tier: LLM tier to use -- ``"sonnet"`` (default) or ``"haiku"``.
        tools: List of tool names available to this agent (default: ``["rag_search"]``).
    """

    agent_id: str
    name: str
    description: str
    knowledge_store_name: str
    llm_tier: str = "sonnet"
    tools: list[str] = field(default_factory=lambda: ["rag_search"])


# Single source of truth: graph builder, classifier, and specialist factory all read from this.
# To add a new agent, add an entry here and populate its knowledge directory.
AGENT_REGISTRY: dict[str, AgentConfig] = {
    "cofounder": AgentConfig(
        agent_id="cofounder",
        name="AI Cofounder",
        description=(
            "General startup strategy, decision-making, and synthesis. Handles queries about "
            "overall company direction, founder methodology (lean startup, Mom Test), prioritization, "
            "and questions that don't clearly fit one specialist domain."
        ),

        knowledge_store_name="unified_knowledge",
    ),
    "gtm": AgentConfig(
        agent_id="gtm",
        name="GTM Strategist",
        description=(
            "Go-to-market strategy, launch planning, customer segmentation, ICP definition, "
            "outbound/inbound channels, cold email and outreach, pricing strategy, and customer "
            "acquisition. Handles questions about how to reach, convert, and retain customers."
        ),

        knowledge_store_name="gtm_knowledge",
    ),
    "finance": AgentConfig(
        agent_id="finance",
        name="Finance & Sales Advisor",
        description=(
            "Financial modeling, runway analysis, unit economics, fundraising strategy, pitch deck "
            "review, SaaS metrics (MRR/ARR/churn), sales pipeline management, deal qualification, "
            "and revenue forecasting. Handles questions about money, funding, and sales execution."
        ),

        knowledge_store_name="finance_knowledge",
    ),
    "marketing": AgentConfig(
        agent_id="marketing",
        name="Marketing Strategist",
        description=(
            "Brand strategy, content marketing, growth marketing, digital campaigns, SEO/SEM, "
            "social media, thought leadership, developer marketing, and messaging/positioning. "
            "Handles questions about building awareness, brand, and demand generation."
        ),

        knowledge_store_name="marketing_knowledge",
    ),
    "fintech": AgentConfig(
        agent_id="fintech",
        name="Fintech Domain Expert",
        description=(
            "Financial services industry, banking APIs, PCI-DSS compliance, SOC2 certification, "
            "fintech regulatory landscape, payment processing, and financial data security. "
            "Handles questions specific to selling into or understanding the fintech vertical."
        ),

        knowledge_store_name="fintech_knowledge",
    ),
    "healthcare": AgentConfig(
        agent_id="healthcare",
        name="Healthcare Domain Expert",
        description=(
            "HIPAA compliance, healthcare IT, EHR/EMR systems, PHI protection, healthcare "
            "regulatory landscape, FDA software regulations, and healthtech buyer personas. "
            "Handles questions specific to selling into or understanding the healthcare vertical."
        ),

        knowledge_store_name="healthcare_knowledge",
    ),
    "product": AgentConfig(
        agent_id="product",
        name="Product & Strategy Advisor",
        description=(
            "Product management, roadmap prioritization, feature scoping, competitive analysis, "
            "product-market fit assessment, MVP definition, user research methodology, and "
            "technical architecture decisions. Handles questions about what to build and why."
        ),

        knowledge_store_name="product_knowledge",
    ),
    "legal": AgentConfig(
        agent_id="legal",
        name="Legal & Compliance Advisor",
        description=(
            "AI regulation (EU AI Act, state laws), data privacy (GDPR, CCPA), industry "
            "compliance frameworks, terms of service, privacy policies, IP protection, and "
            "startup legal structure. Handles questions about legal risk and regulatory compliance."
        ),

        knowledge_store_name="legal_knowledge",
    ),
    "data_analytics": AgentConfig(
        agent_id="data_analytics",
        name="Data & Analytics Advisor",
        description=(
            "Metrics frameworks, KPI definition, market sizing (TAM/SAM/SOM), data-driven "
            "decision making, A/B testing, cohort analysis, and analytics infrastructure. "
            "Handles questions about measuring, sizing markets, and interpreting data."
        ),

        knowledge_store_name="data_analytics_knowledge",
    ),
    "bizdev": AgentConfig(
        agent_id="bizdev",
        name="Business Development Advisor",
        description=(
            "Business model design, pricing models (usage-based, tiered, per-seat), partnership "
            "strategy, channel partnerships, revenue models, customer development, and strategic "
            "alliances. Handles questions about business model, partnerships, and revenue strategy."
        ),

        knowledge_store_name="bizdev_knowledge",
    ),
    "devils_advocate": AgentConfig(
        agent_id="devils_advocate",
        name="Devil's Advocate",
        description=(
            "Critical thinking, assumption challenging, risk identification, stress-testing "
            "strategies and hypotheses, identifying blind spots, contrarian analysis, and "
            "pre-mortem exercises. Activated when the team needs pushback on optimistic "
            "assumptions, when validating high-stakes decisions, or when a plan seems too "
            "smooth. Asks 'what could go wrong?' and 'what are we not seeing?'"
        ),

        knowledge_store_name="unified_knowledge",
    ),
}
