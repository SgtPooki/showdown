"""Unit tests for blinded evaluation mode, undo vote, and voter self-consistency."""

import pytest
from fastapi import HTTPException

from showdown.client import create_tournament as client_create_tournament
from showdown.client import undo_vote as client_undo_vote
from showdown.engine import compute_voter_consistency
from showdown.mcp_server import showdown_get_consistency, showdown_undo_vote
from showdown.models import (
    Candidate,
    CreateTournamentRequest,
    Match,
    TaskType,
    UpdateTournamentRequest,
    VoteRequest,
)
from showdown.server import (
    create_tournament,
    get_voter_consistency,
    record_vote,
    storage,
    undo_vote,
    update_tournament,
)


def test_compute_voter_consistency_pure():
    # Arrange: matches with canonical swaps and self-reversals
    matches = [
        # Voter 1: pair (c1, c2) tested twice, both times c1 is preferred (100% consistent)
        Match(id_a="c1", id_b="c2", winner="a", voter="judge_alice"),
        Match(id_a="c2", id_b="c1", winner="b", voter="judge_alice"),  # c1 is b, b won -> c1 won
        # Voter 1: pair (c2, c3) tested twice, first c2 won, second c3 won (self-reversal)
        Match(id_a="c2", id_b="c3", winner="a", voter="judge_alice"),
        Match(id_a="c2", id_b="c3", winner="b", voter="judge_alice"),
        # Voter 2: pair (c1, c2) tested once (not repeated)
        Match(id_a="c1", id_b="c2", winner="a", voter="judge_bob"),
    ]

    # Act 1: judge_alice
    res_alice = compute_voter_consistency(matches, voter="judge_alice")
    assert res_alice["repeated_pairs_count"] == 2
    assert res_alice["consistent_pairs_count"] == 1
    assert res_alice["self_reversals_count"] == 1
    assert res_alice["consistency_rate"] == 0.5

    # Act 2: judge_bob (no repeated pairs)
    res_bob = compute_voter_consistency(matches, voter="judge_bob")
    assert res_bob["repeated_pairs_count"] == 0
    assert res_bob["consistency_rate"] is None

    # Act 3: pooled / all voters
    res_all = compute_voter_consistency(matches)
    assert res_all["repeated_pairs_count"] == 2
    assert res_all["consistency_rate"] == 0.5


def test_undo_vote_recalibrates_elo(tmp_path):
    # Arrange
    storage.data_dir = tmp_path

    t_req = CreateTournamentRequest(
        title="Undo Arena",
        task_type=TaskType.TEXT,
        candidates=[
            Candidate(id="c1", label="Option 1", content="One"),
            Candidate(id="c2", label="Option 2", content="Two"),
        ],
    )
    t = create_tournament(t_req)
    assert t.stats["c1"].elo == 1200.0
    assert t.stats["c2"].elo == 1200.0
    assert len(t.matches) == 0

    # Act 1: Vote c1 over c2
    record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="human"))
    t_after_vote = storage.load_tournament(t.id)
    assert len(t_after_vote.matches) == 1
    assert t_after_vote.stats["c1"].elo > 1200.0
    assert t_after_vote.stats["c2"].elo < 1200.0

    # Act 2: Undo vote
    undo_res = undo_vote(t.id)
    assert undo_res["status"] == "undone"
    assert undo_res["remaining_matches"] == 0
    assert undo_res["undone_match"]["winner"] == "a"

    t_after_undo = storage.load_tournament(t.id)
    assert len(t_after_undo.matches) == 0
    assert t_after_undo.stats["c1"].elo == 1200.0
    assert t_after_undo.stats["c2"].elo == 1200.0
    assert t_after_undo.stats["c1"].wins == 0
    assert t_after_undo.stats["c2"].losses == 0

    # Act 3: Undo on empty matches raises 400
    with pytest.raises(HTTPException) as excinfo:
        undo_vote(t.id)
    assert excinfo.value.status_code == 400
    assert "No votes to undo" in excinfo.value.detail


def test_undo_vote_per_voter(tmp_path):
    storage.data_dir = tmp_path
    t_req = CreateTournamentRequest(
        title="Multi-Voter Undo",
        candidates=[Candidate(id="c1", content="One"), Candidate(id="c2", content="Two")],
    )
    t = create_tournament(t_req)
    # Alice votes first, then Bob votes
    record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="alice"))
    record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="b", voter="bob"))

    # Alice undos her vote specifically
    undo_res = undo_vote(t.id, voter="alice")
    assert undo_res["undone_match"]["voter"] == "alice"
    assert undo_res["undone_match"]["winner"] == "a"
    assert undo_res["remaining_matches"] == 1

    # Remaining match is Bob's vote
    t_reloaded = storage.load_tournament(t.id)
    assert len(t_reloaded.matches) == 1
    assert t_reloaded.matches[0].voter == "bob"

    # Alice trying to undo again raises 400
    with pytest.raises(HTTPException) as excinfo:
        undo_vote(t.id, voter="alice")
    assert excinfo.value.status_code == 400
    assert "No votes by 'alice'" in excinfo.value.detail


def test_blinded_mode_and_update_tournament(tmp_path):
    # Arrange
    storage.data_dir = tmp_path

    # Create tournament with blinded=True
    t = client_create_tournament(
        title="Blind Arena",
        candidates=[
            {"id": "c1", "label": "Model A (GPT-4)", "content": "Text A"},
            {"id": "c2", "label": "Model B (Claude 3.7)", "content": "Text B"},
        ],
        blinded=True,
        data_dir=str(tmp_path),
    )
    assert t.blinded is True

    # Patch tournament to toggle blinded to False
    res_patch = update_tournament(t.id, UpdateTournamentRequest(blinded=False))
    assert res_patch["status"] == "updated"
    assert res_patch["blinded"] is False

    t_reloaded = storage.load_tournament(t.id)
    assert t_reloaded.blinded is False


def test_mcp_undo_and_consistency_tools(tmp_path):
    # Arrange
    storage.data_dir = tmp_path

    t_req = CreateTournamentRequest(
        title="MCP Tools Arena",
        task_type=TaskType.TEXT,
        candidates=[
            Candidate(id="c1", label="Option 1", content="One"),
            Candidate(id="c2", label="Option 2", content="Two"),
        ],
    )
    t = create_tournament(t_req)

    # Vote twice on same pair with consistent choice
    record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="agent_codex"))
    record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="agent_codex"))

    # Check consistency via MCP tool
    cons_res = showdown_get_consistency(tournament_id=t.id, voter="agent_codex")
    assert cons_res["repeated_pairs_count"] == 1
    assert cons_res["consistent_pairs_count"] == 1
    assert cons_res["consistency_rate"] == 1.0

    # Undo last vote via MCP tool
    undo_res = showdown_undo_vote(tournament_id=t.id)
    assert undo_res["status"] == "undone"
    assert undo_res["remaining_matches"] == 1

    # Consistency now has 0 repeated pairs
    cons_after = showdown_get_consistency(tournament_id=t.id, voter="agent_codex")
    assert cons_after["repeated_pairs_count"] == 0
