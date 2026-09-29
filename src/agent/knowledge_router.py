"""Smart methodology-aware routing for RAG retrieval.

Maps user queries to specific book methodologies so that specialist agents
retrieve from the right source with framework-specific application instructions,
rather than searching all sources indiscriminately.

Usage from specialist nodes::

    from src.agent.knowledge_router import route_query, route_from_hints

    match = route_query(query, original_query)
    if match:
        # Use match.source_names to filter FAISS search
        # Inject match.methodology_prompt into the LLM prompt
"""

import re
from dataclasses import dataclass, field


@dataclass
class MethodologyMatch:
    """A matched methodology with source routing and LLM application instructions.

    Args:
        methodology_id: Registry key (e.g. ``"mom_test"``).
        source_names: Exact FAISS ``source_name`` values to filter retrieval to.
        source_type: FAISS ``source_type`` filter (typically ``"book"``).
        store_name: FAISS store name (typically ``"unified_knowledge"``).
        methodology_prompt: Instructions for the LLM on how to apply this framework.
        top_k: Number of chunks to retrieve.
    """

    methodology_id: str
    source_names: list[str]
    source_type: str = "book"
    store_name: str = "unified_knowledge"
    methodology_prompt: str = ""
    top_k: int = 8


METHODOLOGY_REGISTRY: dict[str, dict] = {
    "mom_test": {
        "keywords": [
            "mom test", "customer interview", "pre-call question",
            "discovery question", "customer discovery", "validation interview",
            "talk to users", "user interview",
        ],
        "match": MethodologyMatch(
            methodology_id="mom_test",
            source_names=["The_Mom_Test_-_Rob_Fitzpatrick"],
            methodology_prompt=(
                "Apply The Mom Test methodology: Talk about their life, not your idea. "
                "Ask about specifics in the past, not generics or opinions about the future. "
                "Talk less, listen more. Look for commitment and advancement signals. "
                "Never pitch during discovery. Bad data comes from compliments, fluff, and ideas. "
                "Good data comes from facts about past behavior, emotional signals, and commitment."
            ),
        ),
    },
    "100m_offers": {
        "keywords": [
            "irresistible offer", "grand slam offer", "value equation",
            "100m offer", "$100m offer", "offer creation", "offer stack",
        ],
        "match": MethodologyMatch(
            methodology_id="100m_offers",
            source_names=[
                "100M_Offers_-_Alex_Hormozi",
            ],
            methodology_prompt=(
                "Apply the Hormozi Grand Slam Offer framework: Dream Outcome x Perceived Likelihood "
                "/ (Time Delay x Effort & Sacrifice). Create an offer stack with bonuses, guarantees, "
                "urgency, and scarcity. List all obstacles to the dream outcome, turn each into a "
                "solution, then trim and stack the deliverables into an irresistible offer."
            ),
        ),
    },
    "100m_leads": {
        "keywords": [
            "lead generation", "lead magnet", "100m leads", "$100m leads",
            "cold outreach", "warm outreach", "getting strangers", "lead gen",
        ],
        "match": MethodologyMatch(
            methodology_id="100m_leads",
            source_names=[
                "100M_Leads_-_Alex_Hormozi",
            ],
            methodology_prompt=(
                "Apply Hormozi's lead generation framework: Core Four channels (warm outreach, "
                "cold outreach, content, paid ads) multiplied by (1-to-1 vs 1-to-many). Lead magnets "
                "should solve a narrow problem completely. 'Give away the secrets, sell the "
                "implementation.' Volume x Quality x Offer = Leads."
            ),
        ),
    },
    "crossing_the_chasm": {
        "keywords": [
            "crossing the chasm", "cross the chasm", "early adopter to mainstream",
            "technology adoption lifecycle", "bowling alley strategy", "beachhead segment",
            "geoffrey moore",
        ],
        "match": MethodologyMatch(
            methodology_id="crossing_the_chasm",
            source_names=[
                "Crossing_the_Chasm_-_Geoffrey_A_Moore",
            ],
            methodology_prompt=(
                "Apply Moore's Crossing the Chasm framework: Technology Adoption Lifecycle with the "
                "CHASM between early adopters and early majority. Use beachhead strategy — dominate "
                "one narrow segment and use it as a reference. Apply the whole product concept. "
                "Pragmatists buy from market leaders in their segment, so position accordingly."
            ),
        ),
    },
    "obviously_awesome": {
        "keywords": [
            "positioning framework", "positioning strategy", "category design",
            "competitive alternative", "obviously awesome", "market category",
            "april dunford",
        ],
        "match": MethodologyMatch(
            methodology_id="obviously_awesome",
            source_names=[
                "Obviously_Awesome_-_April_Dunford",
            ],
            methodology_prompt=(
                "Apply Dunford's positioning framework with 5 components: Competitive Alternatives, "
                "Unique Attributes, Value, Target Customer Segments, and Market Category. Start with "
                "competitive alternatives (what would customers do without you?), then work forward "
                "to define a unique position that makes the product's value obvious."
            ),
        ),
    },
    "product_led_growth": {
        "keywords": [
            "product-led", "plg", "product led growth",
            "freemium model", "self-serve motion", "wes bush",
        ],
        "match": MethodologyMatch(
            methodology_id="product_led_growth",
            source_names=[
                "Product-Led_Growth_-_Wes_Bush",
            ],
            methodology_prompt=(
                "Apply Wes Bush's Product-Led Growth framework: Let the product drive acquisition, "
                "conversion, and expansion. Use the freemium vs free trial vs demo decision tree. "
                "Optimize time-to-value. Prioritize product-qualified leads (PQLs) over MQLs. "
                "Apply the bowling alley onboarding framework to drive users to the aha moment."
            ),
        ),
    },
    "lean_startup": {
        "keywords": [
            "lean startup", "build measure learn", "pivot or persevere",
            "minimum viable", "validated learning",
        ],
        "match": MethodologyMatch(
            methodology_id="lean_startup",
            source_names=[
                "The_Lean_Startup_-_Eric_Ries",
            ],
            methodology_prompt=(
                "Apply Ries's Lean Startup methodology: Build-Measure-Learn loop. Identify "
                "leap-of-faith assumptions and test them first. Use innovation accounting over "
                "vanity metrics. The MVP is the smallest experiment to test the riskiest assumption. "
                "Apply pivot types (zoom-in, zoom-out, customer segment, etc.) when hypotheses fail."
            ),
        ),
    },
    "running_lean": {
        "keywords": [
            "lean canvas", "running lean", "problem-solution fit",
            "solution-market fit", "riskiest assumption",
        ],
        "match": MethodologyMatch(
            methodology_id="running_lean",
            source_names=[
                "Running_Lean_-_Ash_Maurya",
            ],
            methodology_prompt=(
                "Apply Maurya's Running Lean methodology: Lean Canvas (9 boxes). Systematic "
                "de-risking in order: Customer -> Problem -> Solution -> Revenue. Identify and test "
                "the riskiest assumption first. Progress through problem interviews, solution "
                "interviews, then MVP interviews to validate at each stage."
            ),
        ),
    },
    "lean_product_playbook": {
        "keywords": [
            "product-market fit", "lean product", "product playbook",
            "underserved needs", "kano model",
        ],
        "match": MethodologyMatch(
            methodology_id="lean_product_playbook",
            source_names=[
                "The_Lean_Product_Playbook_-_Dan_Olsen",
            ],
            methodology_prompt=(
                "Apply Olsen's Product-Market Fit Pyramid: Target Customer -> Underserved Needs -> "
                "Value Proposition -> Feature Set -> UX. Use the Importance vs Satisfaction framework "
                "to identify underserved needs where customers care deeply but existing solutions "
                "fall short. Build the value proposition around those gaps."
            ),
        ),
    },
    "zero_to_one": {
        "keywords": [
            "zero to one", "0 to 1", "contrarian truth",
            "definite optimism", "peter thiel",
        ],
        "match": MethodologyMatch(
            methodology_id="zero_to_one",
            source_names=[
                "Zero_to_One_-_Peter_Thiel",
            ],
            methodology_prompt=(
                "Apply Thiel's Zero to One thinking: Go from 0->1 (creation) not 1->n (copying). "
                "Monopoly characteristics are proprietary tech 10x better, network effects, economies "
                "of scale, and branding. Ask 'What important truth do few people agree with you on?' "
                "Pursue definite optimism — have a concrete plan, don't rely on incremental iteration."
            ),
        ),
    },
    "traction_eos": {
        "keywords": [
            "eos", "entrepreneurial operating system", "traction eos",
            "rocks", "scorecard", "quarterly rocks", "level 10 meeting",
            "accountability chart", "gino wickman",
        ],
        "match": MethodologyMatch(
            methodology_id="traction_eos",
            source_names=[
                "Traction_-_Gino_Wickman",
            ],
            methodology_prompt=(
                "Apply the Entrepreneurial Operating System (EOS) from Gino Wickman's Traction. "
                "Six Key Components: Vision, People, Data, Issues, Process, Traction. Use the "
                "Vision/Traction Organizer (V/TO), set 90-day Rocks for focus, run Level 10 Meetings, "
                "use the Accountability Chart to put the right people in the right seats, track a "
                "Scorecard of 5-15 weekly metrics, and use the Issues Solving Track (IDS: Identify, "
                "Discuss, Solve)."
            ),
        ),
    },
    "venture_deals": {
        "keywords": [
            "term sheet", "venture deal", "fundraising terms",
            "liquidation preference", "cap table", "venture capital terms",
        ],
        "match": MethodologyMatch(
            methodology_id="venture_deals",
            source_names=[
                "Venture_Deals_-_Brad_Feld",
            ],
            methodology_prompt=(
                "Apply Feld's Venture Deals framework: Separate Economics (price, liquidation "
                "preferences, pay-to-play, vesting) from Control (board seats, protective provisions, "
                "drag-along rights). Distinguish clean vs dirty term sheet terms. Understand "
                "negotiation leverage dynamics — the best leverage is a competitive process."
            ),
        ),
    },
    "hacking_growth": {
        "keywords": [
            "growth hacking", "growth experiment", "growth loop",
            "aha moment", "north star metric", "sean ellis",
            "high-tempo experimentation",
        ],
        "match": MethodologyMatch(
            methodology_id="hacking_growth",
            source_names=[
                "Hacking_Growth_-_Sean_Ellis",
            ],
            methodology_prompt=(
                "Apply Ellis's Hacking Growth framework: Find the North Star Metric. Build a growth "
                "equation unique to your product. Run high-tempo experimentation. Use the AARRR funnel: "
                "Acquisition -> Activation -> Retention -> Revenue -> Referral. Find the 'aha moment' "
                "and drive users to it faster — that is the key to unlocking sustainable growth."
            ),
        ),
    },
    "breakthrough_advertising": {
        "keywords": [
            "breakthrough advertising", "awareness stage", "market sophistication",
            "eugene schwartz", "copywriting",
        ],
        "match": MethodologyMatch(
            methodology_id="breakthrough_advertising",
            source_names=[
                "Breakthrough_Advertising_-_Eugene_Schwartz",
            ],
            methodology_prompt=(
                "Apply Schwartz's Breakthrough Advertising framework: 5 Stages of Market Awareness "
                "(unaware -> problem-aware -> solution-aware -> product-aware -> most aware) and "
                "5 Stages of Market Sophistication. Match the headline and copy approach to the "
                "prospect's current awareness level — this determines what you can and cannot say."
            ),
        ),
    },
    "cashvertising": {
        "keywords": [
            "cashvertising", "advertising psychology", "life force 8",
            "drew whitman",
        ],
        "match": MethodologyMatch(
            methodology_id="cashvertising",
            source_names=[
                "Cashvertising_-_Drew_Eric_Whitman",
            ],
            methodology_prompt=(
                "Apply Whitman's Cashvertising framework: Life Force 8 desires that drive all human "
                "behavior, plus the 9 Secondary wants. Use fear-based vs desire-based appeals depending "
                "on context. Apply psychological triggers for advertising effectiveness — urgency, "
                "social proof, authority, specificity, and emotional resonance."
            ),
        ),
    },
    "startup_owners_manual": {
        "keywords": [
            "customer development", "startup owner", "steve blank",
            "four steps", "get out of the building",
        ],
        "match": MethodologyMatch(
            methodology_id="startup_owners_manual",
            source_names=[
                "The_Startup_Owners_Manual_-_Steve_Blank",
            ],
            methodology_prompt=(
                "Apply Blank's Customer Development methodology: 4 steps — Customer Discovery -> "
                "Customer Validation -> Customer Creation -> Company Building. 'Get out of the "
                "building.' Treat the Business Model Canvas as a set of hypotheses to test. "
                "Pivot when hypotheses fail validation — do not scale before validating."
            ),
        ),
    },
    "fall_in_love_problem": {
        "keywords": [
            "fall in love with the problem", "problem not solution",
            "uri levine", "problem first",
        ],
        "match": MethodologyMatch(
            methodology_id="fall_in_love_problem",
            source_names=[
                "Fall_in_Love_with_the_Problem_Not_the_Solution_-_Uri_Levine",
            ],
            methodology_prompt=(
                "Apply Levine's problem-first methodology: Fall in love with the problem, not the "
                "solution. Develop deep problem understanding before building anything. Validate that "
                "the pain is real, frequent, and worth paying to solve before investing in solutions. "
                "The solution can change; the problem should not."
            ),
        ),
    },
    "high_growth_handbook": {
        "keywords": [
            "high growth handbook", "scaling organization",
            "executive hiring", "board management", "elad gil",
        ],
        "match": MethodologyMatch(
            methodology_id="high_growth_handbook",
            source_names=[
                "High_Growth_Handbook_-_Elad_Gil",
            ],
            methodology_prompt=(
                "Apply Gil's High Growth Handbook: Know when to hire executives vs promote from within. "
                "Follow board management best practices. Evaluate M&A considerations carefully. "
                "Manage organizational structure and culture through rapid growth phases — what works "
                "at 10 people breaks at 100, and what works at 100 breaks at 1000."
            ),
        ),
    },
    "hard_things": {
        "keywords": [
            "hard thing", "wartime ceo", "peacetime ceo",
            "ben horowitz", "the struggle",
        ],
        "match": MethodologyMatch(
            methodology_id="hard_things",
            source_names=[
                "The_Hard_Thing_About_Hard_Things_-_Ben_Horowitz",
            ],
            methodology_prompt=(
                "Apply Horowitz's Hard Things framework: Distinguish wartime vs peacetime CEO mindset. "
                "Make hard decisions with incomplete information — there is no recipe. The Struggle is "
                "normal and expected. Focus on hiring, firing, and culture-building under pressure. "
                "When things go wrong, tell the truth and face reality head-on."
            ),
        ),
    },
}


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Compile a pattern that matches a keyword as whole words.

    The keyword must not be preceded or followed by a letter or digit, so
    ``"eos"`` does not match inside ``"videos"`` and ``"0 to 1"`` does not match
    inside ``"10 to 12"``. A plural ending (``s`` or ``es``) is allowed, so
    ``"customer interview"`` still matches ``"customer interviews"``.

    Args:
        keyword: A single- or multi-word keyword from the registry.

    Returns:
        A compiled pattern for use against lowercased text.
    """
    return re.compile(
        r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?:s|es)?(?![a-z0-9])"
    )


#: Compiled keyword patterns per methodology, built once at import.
_KEYWORD_PATTERNS: dict[str, list[re.Pattern]] = {
    methodology_id: [_keyword_pattern(kw) for kw in entry["keywords"]]
    for methodology_id, entry in METHODOLOGY_REGISTRY.items()
}


def route_query(query: str, original_query: str = "") -> MethodologyMatch | None:
    """Score all methodologies against query text and return the best match.

    Both ``query`` (the sub-question) and ``original_query`` (the user's raw
    message) are checked so that methodology routing works whether triggered
    by the classifier's rewritten sub-question or the user's original phrasing.

    Scoring: each keyword found as whole words adds 1 point. The methodology
    with the most hits wins. This avoids first-match bias when multiple
    methodologies match.

    Args:
        query: The sub-question or search query.
        original_query: The user's original unmodified message.

    Returns:
        The best-scoring ``MethodologyMatch``, or ``None`` if no keywords hit.
    """
    combined = f"{query} {original_query}".lower()
    best_match: MethodologyMatch | None = None
    best_score = 0

    for methodology_id, patterns in _KEYWORD_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern.search(combined))
        if score > best_score:
            best_score = score
            best_match = METHODOLOGY_REGISTRY[methodology_id]["match"]

    return best_match


def route_from_hints(source_hints: list[str]) -> list[MethodologyMatch]:
    """Resolve classifier source_hints (methodology IDs) to MethodologyMatch objects.

    The classifier may emit hints like ``["mom_test", "lean_startup"]`` to tell
    specialist nodes which frameworks to prioritize. This function looks up each
    hint in the registry and returns the corresponding matches.

    Args:
        source_hints: List of methodology_id strings from the classifier.

    Returns:
        List of resolved ``MethodologyMatch`` objects (unrecognized hints are skipped).
    """
    matches = []
    for hint in source_hints:
        entry = METHODOLOGY_REGISTRY.get(hint)
        if entry:
            matches.append(entry["match"])
    return matches
