"""Unit tests for Showdown evolution and preference synthesis engine."""

from pathlib import Path
from showdown.evolve import extract_tournament_preferences, build_evolution_prompt, _parse_candidates_json
from showdown.models import (
    Candidate,
    CandidateStats,
    Match,
    TaskType,
    TriageRecord,
    TriageStatus,
    Tournament,
)


def test_preference_extraction_and_prompt():
    c1 = Candidate(id="c1", label="CODE. SYSTEMS. AGENTS.", content="CODE. SYSTEMS. AGENTS.")
    c2 = Candidate(id="c2", label="SYSTEMS. AGENTS. OPS.", content="SYSTEMS. AGENTS. OPS.")
    c3 = Candidate(id="c3", label="Code Practice", content="Software Systems Practice")

    tournament = Tournament(
        id="tagline_tourney",
        title="Brand Tagline Selection",
        prompt="Select punchy 3-word social card tagline",
        task_type=TaskType.TEXT,
        candidates=[c1, c2, c3],
        stats={
            "c1": CandidateStats(elo=1240.0, wins=3, losses=0, matches=3),
            "c2": CandidateStats(elo=1160.0, wins=1, losses=2, matches=3),
            "c3": CandidateStats(elo=1100.0, wins=0, losses=2, matches=2),
        },
        triage={
            "c1": TriageRecord(status=TriageStatus.FAVORITE, notes="Punchy, perfect tone"),
            "c2": TriageRecord(status=TriageStatus.DISLIKED, notes="Dislike the word ops"),
        },
        matches=[
            Match(
                id_a="c1",
                id_b="c2",
                winner="a",
                elo_a_before=1200,
                elo_b_before=1200,
                elo_a_after=1216,
                elo_b_after=1184,
                notes="Candidate A is much stronger, ops feels generic",
            )
        ],
    )

    prefs = extract_tournament_preferences(tournament)
    assert len(prefs["top_performers"]) >= 1
    assert prefs["top_performers"][0]["id"] == "c1"
    assert len(prefs["bottom_performers"]) >= 1
    assert any("Dislike the word ops" in n for n in prefs["notes"])

    prompt, summary = build_evolution_prompt(tournament, count=4, instructions="Keep strictly to 3 words")
    assert "CODE. SYSTEMS. AGENTS." in prompt
    assert "Dislike the word ops" in prompt
    assert "Keep strictly to 3 words" in prompt
    assert "Generate exactly 4 NEW distinct candidate variations" in prompt


def test_parse_candidates_json():
    mock_llm_output = """
```json
[
  {"label": "CODE. ARCHITECTURE. AGENTS.", "content": "CODE. ARCHITECTURE. AGENTS."},
  {"label": "SYSTEMS. RUNTIMES. AGENTS.", "content": "SYSTEMS. RUNTIMES. AGENTS."}
]
```
"""
    cands = _parse_candidates_json(mock_llm_output, next_gen=2, backend_name="mock")
    assert len(cands) == 2
    assert cands[0].label == "CODE. ARCHITECTURE. AGENTS."
    assert cands[0].generation == 2
    assert cands[0].metadata["backend"] == "mock"


def test_rejected_candidate_with_high_elo_never_in_top():
    """A candidate explicitly disliked by the user must NEVER be placed in top_performers, even with Elo >= 1200."""
    c = Candidate(id="lucky_disliked", label="Disliked Luck", content="Bad tone")
    t = Tournament(
        id="t_dislike",
        title="Dislike test",
        task_type=TaskType.TEXT,
        candidates=[c],
        stats={"lucky_disliked": CandidateStats(elo=1350.0, wins=4, losses=0, matches=4)},
        triage={"lucky_disliked": TriageRecord(status=TriageStatus.DISLIKED, notes="Hate this style")},
    )
    prefs = extract_tournament_preferences(t)
    assert len(prefs["top_performers"]) == 0
    assert len(prefs["bottom_performers"]) == 1
    assert prefs["bottom_performers"][0]["id"] == "lucky_disliked"


def test_cold_start_empty_preferences():
    """When no matches or triage records exist, candidate order must not arbitrarily designate winners."""
    cands = [Candidate(id=f"c{i}", content=f"Candidate {i}") for i in range(6)]
    t = Tournament(
        id="t_cold",
        title="Cold Start",
        task_type=TaskType.TEXT,
        candidates=cands,
        stats={c.id: CandidateStats() for c in cands},
    )
    prefs = extract_tournament_preferences(t)
    assert len(prefs["top_performers"]) == 0
    assert len(prefs["bottom_performers"]) == 0
    prompt, _ = build_evolution_prompt(t, count=3)
    assert "No clear winners yet" in prompt


def test_svg_evolution_prompt_contains_svg_guidance():
    # Arrange
    tournament = Tournament(
        id="t_svg",
        title="Logo Design",
        task_type=TaskType.SVG,
        candidates=[Candidate(id="svg1", content="<svg></svg>")],
        stats={"svg1": CandidateStats()},
    )

    # Act
    prompt, _ = build_evolution_prompt(tournament, count=2)

    # Assert
    assert "Task Type: svg" in prompt
    assert "valid, standalone inline SVG string" in prompt
