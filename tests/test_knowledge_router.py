"""Tests for src.agent.knowledge_router — methodology-aware RAG routing."""

import re

import pytest

from src.agent.knowledge_router import (
    METHODOLOGY_REGISTRY,
    MethodologyMatch,
    route_from_hints,
    route_query,
)


# ---------------------------------------------------------------------------
# 1. Basic keyword matching
# ---------------------------------------------------------------------------

class TestRouteQueryBasic:
    def test_mom_test_keyword_match(self):
        result = route_query("help me with mom test questions")
        assert result is not None
        assert result.methodology_id == "mom_test"

    def test_returns_methodology_match_instance(self):
        result = route_query("help me with mom test questions")
        assert isinstance(result, MethodologyMatch)


# ---------------------------------------------------------------------------
# 2. Case insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    def test_uppercase_mom_test(self):
        result = route_query("MOM TEST questions")
        assert result is not None
        assert result.methodology_id == "mom_test"

    def test_mixed_case(self):
        result = route_query("Mom Test Discovery Question")
        assert result is not None
        assert result.methodology_id == "mom_test"


# ---------------------------------------------------------------------------
# 3. Original query fallback
# ---------------------------------------------------------------------------

class TestOriginalQueryFallback:
    def test_match_via_original_query(self):
        result = route_query("some sub-question", "prepare Mom Test questions")
        assert result is not None
        assert result.methodology_id == "mom_test"

    def test_no_match_in_sub_but_match_in_original(self):
        result = route_query("generic question about pricing", "how to build a grand slam offer")
        assert result is not None
        assert result.methodology_id == "100m_offers"


# ---------------------------------------------------------------------------
# 4. All 19 methodologies — at least one keyword each
# ---------------------------------------------------------------------------

ALL_METHODOLOGY_SAMPLES = [
    ("mom_test", "mom test"),
    ("100m_offers", "grand slam offer"),
    ("100m_leads", "lead generation"),
    ("crossing_the_chasm", "crossing the chasm"),
    ("obviously_awesome", "positioning framework"),
    ("product_led_growth", "product-led"),
    ("lean_startup", "lean startup"),
    ("running_lean", "lean canvas"),
    ("lean_product_playbook", "product-market fit"),
    ("zero_to_one", "zero to one"),
    ("traction_eos", "entrepreneurial operating system"),
    ("venture_deals", "term sheet"),
    ("hacking_growth", "growth hacking"),
    ("breakthrough_advertising", "breakthrough advertising"),
    ("cashvertising", "cashvertising"),
    ("startup_owners_manual", "customer development"),
    ("fall_in_love_problem", "fall in love with the problem"),
    ("high_growth_handbook", "high growth handbook"),
    ("hard_things", "wartime ceo"),
]


class TestAllMethodologies:
    def test_registry_has_19_entries(self):
        assert len(METHODOLOGY_REGISTRY) == 19

    @pytest.mark.parametrize("methodology_id,keyword", ALL_METHODOLOGY_SAMPLES)
    def test_keyword_routes_to_correct_methodology(self, methodology_id, keyword):
        result = route_query(keyword)
        assert result is not None, f"Expected match for keyword '{keyword}'"
        assert result.methodology_id == methodology_id


# ---------------------------------------------------------------------------
# 5. No match
# ---------------------------------------------------------------------------

class TestNoMatch:
    def test_unrelated_query_returns_none(self):
        result = route_query("what's the weather today")
        assert result is None

    def test_empty_query_returns_none(self):
        result = route_query("")
        assert result is None


# ---------------------------------------------------------------------------
# 6. route_from_hints() happy path
# ---------------------------------------------------------------------------

class TestRouteFromHints:
    def test_two_valid_hints(self):
        matches = route_from_hints(["mom_test", "obviously_awesome"])
        assert len(matches) == 2
        ids = {m.methodology_id for m in matches}
        assert ids == {"mom_test", "obviously_awesome"}
        for m in matches:
            assert isinstance(m, MethodologyMatch)


# ---------------------------------------------------------------------------
# 7. route_from_hints() with invalid hints
# ---------------------------------------------------------------------------

class TestRouteFromHintsInvalid:
    def test_nonexistent_hint_returns_empty(self):
        matches = route_from_hints(["nonexistent"])
        assert matches == []

    def test_mix_valid_and_invalid(self):
        matches = route_from_hints(["mom_test", "nonexistent", "lean_startup"])
        assert len(matches) == 2
        ids = {m.methodology_id for m in matches}
        assert ids == {"mom_test", "lean_startup"}


# ---------------------------------------------------------------------------
# 7b. Keywords match whole words, not substrings
# ---------------------------------------------------------------------------

class TestWholeWordMatching:
    @pytest.mark.parametrize("query", [
        "Can you summarize these onboarding videos for the team?",  # contains "eos"
        "We hired a rockstar engineer last month",                  # contains "rocks"
        "We plan to grow from 10 to 12 people this year",           # contains "0 to 1"
    ])
    def test_keyword_inside_another_word_does_not_match(self, query):
        assert route_query(query) is None

    @pytest.mark.parametrize("query, expected", [
        ("Should we adopt EOS for our weekly meetings?", "traction_eos"),
        ("what is plg and does it fit us", "product_led_growth"),
        ("How do we go from 0 to 1 in this market?", "zero_to_one"),
    ])
    def test_short_keyword_as_a_word_matches(self, query, expected):
        result = route_query(query)
        assert result is not None
        assert result.methodology_id == expected

    @pytest.mark.parametrize("query, expected", [
        ("We ran a customer interview yesterday", "mom_test"),
        ("Help me plan five customer interviews", "mom_test"),
        ("Draft a term sheet, please.", "venture_deals"),
        ("How do I build a $100M offer?", "100m_offers"),
        ("Is product-led growth right for us?", "product_led_growth"),
        ("What goes into a level 10 meeting", "traction_eos"),
    ])
    def test_multi_word_keyword_matches(self, query, expected):
        result = route_query(query)
        assert result is not None
        assert result.methodology_id == expected

    @pytest.mark.parametrize("query", [
        "We interviewed a customer interviewer about hiring",  # "customer interviewer"
        "The term sheeting process in textile mills",          # "term sheeting"
    ])
    def test_multi_word_keyword_inside_longer_words_does_not_match(self, query):
        assert route_query(query) is None


# ---------------------------------------------------------------------------
# 8. route_from_hints() empty list
# ---------------------------------------------------------------------------

class TestRouteFromHintsEmpty:
    def test_empty_list_returns_empty(self):
        matches = route_from_hints([])
        assert matches == []


# ---------------------------------------------------------------------------
# 9. Source names correctness
# ---------------------------------------------------------------------------

# A source name must be a clean file name stem in the form ``Title_-_Author``:
# letters, digits, underscores and hyphens only.
CLEAN_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*(_[A-Za-z0-9-]+)*_-_[A-Za-z]+(_[A-Za-z]+)*$")


class TestSourceNames:
    @pytest.mark.parametrize("methodology_id", list(METHODOLOGY_REGISTRY.keys()))
    def test_source_names_are_clean_stems(self, methodology_id):
        match = METHODOLOGY_REGISTRY[methodology_id]["match"]
        for name in match.source_names:
            assert CLEAN_STEM.match(name), (
                f"Source name '{name}' for {methodology_id} is not a clean Title_-_Author stem"
            )

    def test_source_names_are_unique(self):
        names = [
            name
            for entry in METHODOLOGY_REGISTRY.values()
            for name in entry["match"].source_names
        ]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("methodology_id", list(METHODOLOGY_REGISTRY.keys()))
    def test_source_names_not_empty(self, methodology_id):
        match = METHODOLOGY_REGISTRY[methodology_id]["match"]
        assert len(match.source_names) > 0


# ---------------------------------------------------------------------------
# 10. Methodology prompt not empty
# ---------------------------------------------------------------------------

class TestMethodologyPrompt:
    @pytest.mark.parametrize("methodology_id", list(METHODOLOGY_REGISTRY.keys()))
    def test_methodology_prompt_not_empty(self, methodology_id):
        match = METHODOLOGY_REGISTRY[methodology_id]["match"]
        assert match.methodology_prompt, f"methodology_prompt is empty for {methodology_id}"
        assert len(match.methodology_prompt.strip()) > 0
