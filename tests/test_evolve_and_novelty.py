"""Comprehensive tests for chained tournament growth, structural axis divergence, and hybrid novelty injection."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from showdown.evolve import (
    _parse_candidates_json,
    build_evolution_prompt,
    execute_evolution,
    extract_tournament_preferences,
    resolve_upstream_context,
)
from showdown.judge import build_judge_prompt, evaluate_pair, run_tournament_judge
from showdown.models import (
    Candidate,
    CandidateStats,
    CreateTournamentRequest,
    EvolveRequest,
    TaskType,
    TriageRecord,
    TriageStatus,
    Tournament,
    TournamentStatus,
)
from showdown.server import app, storage


@pytest.fixture
def client(tmp_path: Path):
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path
    c = TestClient(app)
    yield c
    storage.data_dir = orig_dir


def test_resolve_upstream_context_only_accepted_winners(tmp_path: Path):
    """Upstream resolution must strictly inherit accepted winners and never provisional leaders."""
    s = storage
    s.data_dir = tmp_path

    # Parent 1: completed with accepted winner
    p1 = Tournament(
        id="stage_1",
        title="Chapter 1: The Outpost",
        prompt="Write the arrival at the outpost",
        task_type=TaskType.TEXT,
        candidates=[
            Candidate(id="p1_a", label="Draft A", content="Snow fell upon the iron gates."),
            Candidate(id="p1_b", label="Draft B", content="The outpost was dark and silent."),
        ],
        accepted_candidate_id="p1_a",
        status=TournamentStatus.COMPLETED,
    )
    s.save_tournament(p1)

    # Parent 2: active, provisional leader but NO accepted winner
    p2 = Tournament(
        id="stage_2_unsettled",
        title="Chapter 2: The Infiltration",
        task_type=TaskType.TEXT,
        candidates=[
            Candidate(id="p2_a", label="Draft A", content="They crept past the sentries."),
        ],
        stats={"p2_a": CandidateStats(elo=1300.0, wins=5)},
        accepted_candidate_id=None,
        status=TournamentStatus.ACTIVE,
    )
    s.save_tournament(p2)

    # Child tournament pointing to stage_1: resolves accepted winner
    child_1 = Tournament(
        id="stage_2_growth",
        title="Chapter 2: Continuation",
        task_type=TaskType.TEXT,
        candidates=[Candidate(id="c1", content="Next part")],
        parent_ids=["stage_1"],
    )
    ctx1 = resolve_upstream_context(child_1, s)
    assert ctx1 is not None
    assert ctx1["id"] == "p1_a"
    assert ctx1["content"] == "Snow fell upon the iron gates."
    assert ctx1["parent_title"] == "Chapter 1: The Outpost"

    # Child tournament pointing to stage_2_unsettled: does NOT fall back to provisional leader
    child_2 = Tournament(
        id="stage_3_pending",
        title="Chapter 3",
        task_type=TaskType.TEXT,
        candidates=[Candidate(id="c1", content="Part 3")],
        parent_ids=["stage_2_unsettled"],
    )
    ctx2 = resolve_upstream_context(child_2, s)
    assert ctx2 is None


def test_build_evolution_prompt_refine_vs_diverge():
    """Verify divergence prompt structural axis instructions, differing rationale, and preservation of rejections."""
    c1 = Candidate(id="c1", label="Modular Monolith", content="def app(): pass")
    c2 = Candidate(id="c2", label="Microservices Spaghetti", content="def rpc(): pass")

    t = Tournament(
        id="arch_tourney",
        title="Architecture Decision",
        prompt="Design robust backend service",
        task_type=TaskType.CODE,
        candidates=[c1, c2],
        stats={
            "c1": CandidateStats(elo=1280.0, wins=3, losses=0),
            "c2": CandidateStats(elo=1120.0, wins=0, losses=3),
        },
        triage={
            "c1": TriageRecord(status=TriageStatus.FAVORITE, notes="Clean cohesion"),
            "c2": TriageRecord(status=TriageStatus.DISLIKED, notes="Too much networking overhead"),
        },
    )

    # 1. Refine / Growth Prompt
    refine_prompt, refine_sum = build_evolution_prompt(t, count=3, mode="refine")
    assert "High-Performing Winners" in refine_prompt
    assert "Modular Monolith" in refine_prompt
    assert "Too much networking overhead" in refine_prompt
    assert "emphasize, deepen, and refine their patterns" in refine_prompt
    assert "clean, executable code" in refine_prompt

    # 2. Diverge / Novelty Prompt
    diverge_prompt, diverge_sum = build_evolution_prompt(t, count=4, mode="diverge")
    assert "Divergence & Structural Novelty Instructions" in diverge_prompt
    assert "Structural Axis Divergence" in diverge_prompt
    assert "Distinct Stances" in diverge_prompt
    assert "No Negation Collapse" in diverge_prompt
    # CRITICAL: Rejections and critiques MUST still be present in divergence mode to maintain quality floor
    assert "Low-Performing / Rejected" in diverge_prompt
    assert "Microservices Spaghetti" in diverge_prompt
    assert "Too much networking overhead" in diverge_prompt
    assert "'differs_by'" in diverge_prompt


def test_parse_candidates_json_metadata_and_wildcards():
    """_parse_candidates_json correctly populates differs_by, lineage, and wildcard flag."""
    raw_json = json.dumps([
        {
            "label": "Event-Driven Actor Architecture",
            "content": "class Actor: ...",
            "differs_by": "Shifts from synchronous request-response to reactive asynchronous message passing.",
        },
        {
            "label": "Refined Pipeline",
            "content": "class Pipeline: ...",
        },
    ])

    # Parse as divergence wildcards
    cands_diverged = _parse_candidates_json(raw_json, next_gen=3, backend_name="mock_llm", is_divergence=True, is_wildcard=True)
    assert len(cands_diverged) == 2
    assert cands_diverged[0].metadata["lineage"] == "diverged"
    assert cands_diverged[0].metadata["wildcard"] is True
    assert "reactive asynchronous message passing" in cands_diverged[0].metadata["differs_by"]
    assert cands_diverged[0].generation == 3

    # Parse as standard refinements
    cands_evolved = _parse_candidates_json(raw_json, next_gen=2, backend_name="mock_llm", is_divergence=False, is_wildcard=False)
    assert len(cands_evolved) == 2
    assert cands_evolved[0].metadata["lineage"] == "evolved"
    assert "wildcard" not in cands_evolved[0].metadata


def test_execute_evolution_hybrid_mode():
    """Hybrid mode splits count into refinements and wildcards with proper tags."""
    t = Tournament(
        id="t_hybrid",
        title="Taglines",
        prompt="Design brand taglines",
        task_type=TaskType.TEXT,
        candidates=[Candidate(id="c1", label="Fast Code", content="Fast Code")],
    )

    mock_refine_response = json.dumps([
        {"label": "Refined #1", "content": "Fast Scalable Code", "differs_by": "Deepened performance"},
        {"label": "Refined #2", "content": "Swift Resilient Code", "differs_by": "Enhanced resilience"},
        {"label": "Refined #3", "content": "Rapid Secure Code", "differs_by": "Added security focus"},
    ])
    mock_wildcard_response = json.dumps([
        {"label": "Wildcard #1", "content": "Systems Beyond Logic", "differs_by": "Shifts focus from speed to philosophical transcendence"},
        {"label": "Wildcard #2", "content": "Code that Breathes", "differs_by": "Employs biological organic metaphor rather than mechanical phrasing"},
    ])

    call_count = 0

    def mock_backend_caller(prompt, backend):
        nonlocal call_count
        call_count += 1
        if "Divergence & Structural Novelty" in prompt:
            return mock_wildcard_response
        return mock_refine_response

    with patch("showdown.evolve.call_generation_backend", side_effect=mock_backend_caller):
        res = execute_evolution(
            tournament=t,
            count=5,
            mode="hybrid",
            wildcards=2,
            backend="mock",
        )

    assert res.mode == "hybrid"
    assert res.refine_count == 3
    assert res.wildcard_count == 2
    assert len(res.new_candidates) == 5

    # Check that wildcards have wildcard metadata and differs_by
    wildcards = [c for c in res.new_candidates if c.metadata.get("wildcard")]
    assert len(wildcards) == 2
    assert wildcards[0].metadata["lineage"] == "diverged"
    assert "philosophical transcendence" in wildcards[0].metadata["differs_by"]

    refinements = [c for c in res.new_candidates if not c.metadata.get("wildcard")]
    assert len(refinements) == 3
    assert refinements[0].metadata["lineage"] == "evolved"


def test_judge_incorporates_upstream_context():
    """Automated judge prompts must include upstream context to prevent stage blindness."""
    cand_a = Candidate(id="a", content="Chapter 2 continuation option A")
    cand_b = Candidate(id="b", content="Chapter 2 continuation option B")
    upstream_context = "Stage: Act 1\nWinner: The hero arrived at the citadel gates under heavy blizzard."

    prompt = build_judge_prompt(
        candidate_a=cand_a,
        candidate_b=cand_b,
        task_prompt="Continue the scene immediately following Act 1",
        context=upstream_context,
    )

    assert "[Upstream Stage Context / Baseline]:" in prompt
    assert "The hero arrived at the citadel gates" in prompt
    assert "Continue the scene immediately following Act 1" in prompt


def test_api_create_chained_tournament_with_chain_mode(client: TestClient):
    """POST /api/tournaments correctly accepts chain_mode and populates context from parent winner."""
    # Step 1: Create parent stage
    res1 = client.post("/api/tournaments", json={
        "title": "Act I",
        "task_type": "text",
        "candidates": [
            {"id": "act1_1", "label": "Option 1", "content": "They entered the abandoned cavern."},
            {"id": "act1_2", "label": "Option 2", "content": "They bypassed the mountain path."},
        ],
    })
    assert res1.status_code == 200
    p_id = res1.json()["id"]

    # Step 2: Accept winner in parent stage
    res_accept = client.post(f"/api/tournaments/{p_id}/accept", json={
        "candidate_id": "act1_1",
        "notes": "Strongest narrative anchor",
    })
    assert res_accept.status_code == 200

    # Step 3: Create child stage with divergence strategy and empty context
    res_child = client.post("/api/tournaments", json={
        "title": "Act II: Divergent Path",
        "task_type": "text",
        "parent_ids": [p_id],
        "chain_mode": "divergence",
        "candidates": [
            {"id": "act2_1", "content": "Deep in the dark, they lit a torch."},
        ],
    })
    assert res_child.status_code == 200
    child_data = res_child.json()
    assert child_data["chain_mode"] == "divergence"
    assert "They entered the abandoned cavern." in child_data["context"]


def test_api_evolve_endpoint_modes_and_wildcards(client: TestClient):
    """POST /api/tournaments/{id}/evolve dispatches mode and wildcards."""
    res_create = client.post("/api/tournaments", json={
        "title": "Novelty Test Tournament",
        "task_type": "text",
        "candidates": [
            {"id": "c1", "label": "Winner 1", "content": "Solid baseline solution."},
        ],
    })
    t_id = res_create.json()["id"]

    mock_llm_json = json.dumps([
        {"label": "Novel Concept", "content": "Radical alternate concept.", "differs_by": "Explores decentralized paradigm."},
    ])

    with patch("showdown.evolve.call_generation_backend", return_value=mock_llm_json):
        res_evolve = client.post(f"/api/tournaments/{t_id}/evolve", json={
            "count": 1,
            "mode": "diverge",
            "backend": "mock",
        })
        assert res_evolve.status_code == 200
        data = res_evolve.json()
        assert data["mode"] == "diverge"
        assert data["wildcard_count"] == 1
        assert len(data["new_candidates"]) == 1
        assert data["new_candidates"][0]["metadata"]["wildcard"] is True
        assert data["new_candidates"][0]["metadata"]["differs_by"] == "Explores decentralized paradigm."

    # Verify tournament candidate pool updated in storage
    res_get = client.get(f"/api/tournaments/{t_id}")
    assert res_get.status_code == 200
    candidates = res_get.json()["candidates"]
    assert len(candidates) == 2
    assert any(c["metadata"].get("wildcard") for c in candidates)
