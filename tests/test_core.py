"""Unit tests for Showdown core tournament engine and storage."""

import shutil
from pathlib import Path
from showdown.engine import calculate_expected_score, update_elo, select_matchup
from showdown.models import Candidate, CandidateStats, Tournament, TaskType, VoteRequest
from showdown.storage import Storage


def test_elo_math():
    # Equal ratings should have 0.5 expected score
    exp = calculate_expected_score(1200.0, 1200.0)
    assert abs(exp - 0.5) < 0.001

    # Winner gains, loser drops
    new_a, new_b = update_elo(1200.0, 1200.0, actual_score_a=1.0)
    assert new_a > 1200.0
    assert new_b < 1200.0
    assert new_a + new_b == 2400.0


def test_matchmaker_and_storage(tmp_path):
    storage = Storage(data_dir=tmp_path)

    cands = [
        Candidate(id="m1", label="Model 1", content="print('hello 1')"),
        Candidate(id="m2", label="Model 2", content="print('hello 2')"),
        Candidate(id="m3", label="Model 3", content="print('hello 3')"),
    ]

    t = Tournament(
        id="test_tourney",
        title="Test Code",
        task_type=TaskType.CODE,
        candidates=cands,
        stats={c.id: CandidateStats() for c in cands}
    )
    storage.save_tournament(t)

    loaded = storage.load_tournament("test_tourney")
    assert loaded is not None
    assert len(loaded.candidates) == 3

    pair = select_matchup(loaded)
    assert pair is not None
    assert pair[0].id != pair[1].id


def test_dynamic_k_factor():
    from showdown.engine import get_dynamic_k_factor
    assert get_dynamic_k_factor(1, 2) == 48.0
    assert get_dynamic_k_factor(6, 8) == 32.0
    assert get_dynamic_k_factor(20, 25) == 24.0


def test_accept_and_wait_endpoints(tmp_path):
    from showdown.client import create_tournament, accept_candidate, wait_for_tournament
    from showdown.models import TournamentStatus

    t = create_tournament(
        title="Session Test",
        items=[
            {"id": "a", "label": "Option A", "content": "print('A')"},
            {"id": "b", "label": "Option B", "content": "print('B')"},
        ],
        task_type="diff",
        tournament_id="session_tourney",
        data_dir=str(tmp_path),
    )
    assert t.task_type.value == "diff"

    # Accept winner
    res = accept_candidate("session_tourney", candidate_id="a", data_dir=str(tmp_path))
    assert res["status"] == "completed"
    assert res["accepted_candidate_id"] == "a"

    # Wait endpoint should resolve completed immediately
    wait_res = wait_for_tournament("session_tourney", timeout=2, data_dir=str(tmp_path))
    assert wait_res["completed"] is True
    assert wait_res["accepted_candidate"]["id"] == "a"


def test_kto_and_jsonl_export(tmp_path):
    import json
    from showdown.server import export_tournament, storage
    from showdown.models import Candidate, CandidateStats, TriageRecord, TriageStatus, Tournament, TaskType

    orig_dir = storage.data_dir
    storage.data_dir = tmp_path
    try:
        t = Tournament(
            id="export_tourney",
            title="Export Tourney",
            task_type=TaskType.TEXT,
            candidates=[
                Candidate(id="c1", label="Winner", content="Great copy"),
                Candidate(id="c2", label="Loser", content="Bad copy"),
            ],
            stats={"c1": CandidateStats(), "c2": CandidateStats()},
            triage={
                "c1": TriageRecord(status=TriageStatus.LIKED),
                "c2": TriageRecord(status=TriageStatus.DISLIKED),
            },
        )
        storage.save_tournament(t)

        kto_data = export_tournament("export_tourney", format="kto")
        assert len(kto_data) == 2
        assert any(row["label"] is True and row["candidate_id"] == "c1" for row in kto_data)
        assert any(row["label"] is False and row["candidate_id"] == "c2" for row in kto_data)

        # JSONL export returns Response or formatted string
        resp = export_tournament("export_tourney", format="kto", jsonl=True)
        lines = [json.loads(line) for line in resp.body.decode().strip().split("\n")]
        assert len(lines) == 2
    finally:
        storage.data_dir = orig_dir
