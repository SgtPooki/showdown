"""Tests for multi-generational simulation harness, logo arena, and chained narrative evaluation."""

import json
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from showdown.models import Candidate, TaskType, Tournament
from showdown.server import app
from showdown.simulate import SimulationHarness
from showdown.storage import Storage


@pytest.fixture
def client():
    return TestClient(app)


def test_generational_simulation_loop():
    """Verify N-generation self-play loop, Elo progression, and Gen N vs Gen 1 validation battle."""
    harness = SimulationHarness()

    mock_llm_json = json.dumps([
        {"label": "Gen 2 Contender #1", "content": "Superior refined concept", "differs_by": "Better ergonomics"},
        {"label": "Gen 2 Contender #2", "content": "Another advanced concept", "differs_by": "Higher throughput"},
    ])

    with patch("showdown.evolve.call_generation_backend", return_value=mock_llm_json):
        res = harness.run_generational_simulation(
            generations=2,
            candidates_per_gen=2,
            judge_rounds_per_gen=3,
            judge_backend="mock",
            generation_backend="mock",
        )

    assert res["generations_run"] == 2
    assert len(res["generation_history"]) == 2
    assert res["total_candidates"] == 4

    # Verify history structure
    gen1_hist = res["generation_history"][0]
    gen2_hist = res["generation_history"][1]
    assert gen1_hist["generation"] == 1
    assert gen2_hist["generation"] == 2
    assert gen1_hist["candidate_count"] == 2
    assert gen2_hist["candidate_count"] == 2

    # Verify validation battle occurred
    vb = res["validation_battle"]
    assert vb["matches_played"] > 0
    assert "gen_n_win_rate_pct" in vb
    assert "objective_convergence_proven" in vb
    assert "winning_candidate" in res
    assert res["winning_candidate"]["id"] is not None


def test_logo_arena_simulation():
    """Verify Scenario 2: SVG vector logo arena evaluation."""
    harness = SimulationHarness()

    res = harness.run_logo_arena_simulation(
        company_name="Vortex Compute",
        rounds=3,
        judge_backend="mock",
    )

    assert res["company_name"] == "Vortex Compute"
    assert res["candidates_count"] == 3
    assert res["matches_evaluated"] > 0
    assert "winning_logo" in res
    assert "<svg" in res["winning_logo"]["svg_content"]
    assert res["winning_logo"]["elo"] > 0


def test_chained_narrative_simulation():
    """Verify Scenario 3: Chained sequential stage narrative simulation."""
    harness = SimulationHarness()

    custom_stages = [
        {"title": "Prologue", "prompt": "Establish the dystopian space colony."},
        {"title": "Inciting Incident", "prompt": "The primary life support manifold fails."},
    ]

    res = harness.run_chained_narrative_simulation(
        project_title="ExoColony 9",
        stages_config=custom_stages,
        judge_backend="mock",
        generation_backend="mock",
    )

    assert res["project_title"] == "ExoColony 9"
    assert res["stages_completed"] == 2
    assert len(res["tournament_ids"]) == 2
    assert len(res["assembled_document"]) == 2
    assert res["assembled_document"][0]["stage"] == "Prologue"
    assert res["assembled_document"][1]["stage"] == "Inciting Incident"
    assert "## Prologue" in res["full_text"]
    assert "## Inciting Incident" in res["full_text"]


def test_api_simulations_endpoint(client: TestClient):
    """Verify POST /api/simulations endpoint for generational, logo, and chained scenarios."""
    mock_llm_json = json.dumps([
        {"label": "Gen 2 Item", "content": "Evolved text", "differs_by": "Precision"},
    ])

    with patch("showdown.evolve.call_generation_backend", return_value=mock_llm_json):
        # 1. Generational
        res_gen = client.post("/api/simulations", json={
            "scenario": "generational",
            "generations": 2,
            "candidates_per_gen": 2,
            "judge_backend": "mock",
            "generation_backend": "mock",
        })
        assert res_gen.status_code == 200
        data_gen = res_gen.json()
        assert data_gen["generations_run"] == 2

        # 2. Logo
        res_logo = client.post("/api/simulations", json={
            "scenario": "logo",
            "company": "Test Logo Inc",
            "judge_backend": "mock",
        })
        assert res_logo.status_code == 200
        data_logo = res_logo.json()
        assert data_logo["company_name"] == "Test Logo Inc"

        # 3. Chained
        res_chained = client.post("/api/simulations", json={
            "scenario": "chained",
            "company": "Spec Document",
            "judge_backend": "mock",
            "generation_backend": "mock",
        })
        assert res_chained.status_code == 200
        data_chained = res_chained.json()
        assert data_chained["stages_completed"] == 3


def test_mcp_run_simulation_tool():
    """Verify MCP showdown_run_simulation tool execution."""
    from showdown.mcp_server import showdown_run_simulation

    mock_llm_json = json.dumps([
        {"label": "MCP Gen 2", "content": "Evolved MCP text", "differs_by": "Clarity"},
    ])

    with patch("showdown.evolve.call_generation_backend", return_value=mock_llm_json):
        res = showdown_run_simulation(
            scenario="generational",
            generations=2,
            candidates_per_gen=2,
            judge_backend="mock",
            generation_backend="mock",
        )
        assert res["generations_run"] == 2
        assert "validation_battle" in res
