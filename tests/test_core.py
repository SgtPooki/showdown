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
    print("All core unit tests passed!")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_elo_math()
        test_matchmaker_and_storage(Path(td))
