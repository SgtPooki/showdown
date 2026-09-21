"""Tests for Bradley-Terry MLE estimation, Bayesian uncertainty bands, and stopping criteria."""

import pytest
from fastapi.testclient import TestClient

from showdown.engine import (
    _invert_matrix,
    check_convergence,
    fit_bradley_terry,
    replay_stats,
    select_matchup,
)
from showdown.models import Candidate, CandidateStats, Match, Tournament
from showdown.server import app


def test_invert_matrix():
    # 2x2 identity matrix
    mat = [[1.0, 0.0], [0.0, 1.0]]
    inv = _invert_matrix(mat)
    assert round(inv[0][0], 4) == 1.0
    assert round(inv[0][1], 4) == 0.0
    assert round(inv[1][0], 4) == 0.0
    assert round(inv[1][1], 4) == 1.0

    # 2x2 symmetric positive-definite
    mat2 = [[2.0, 1.0], [1.0, 2.0]]
    # Determinant = 3, Inverse = [[2/3, -1/3], [-1/3, 2/3]]
    inv2 = _invert_matrix(mat2)
    assert round(inv2[0][0], 4) == round(2.0 / 3.0, 4)
    assert round(inv2[0][1], 4) == round(-1.0 / 3.0, 4)
    assert round(inv2[1][1], 4) == round(2.0 / 3.0, 4)


def test_fit_bradley_terry_transitivity():
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
        Candidate(id="c3", label="Option 3", content="Text 3"),
    ]
    # c1 beats c2 consistently, c2 beats c3 consistently
    matches = [
        Match(id_a="c1", id_b="c2", winner="a"),
        Match(id_a="c1", id_b="c2", winner="a"),
        Match(id_a="c2", id_b="c3", winner="a"),
        Match(id_a="c2", id_b="c3", winner="a"),
        Match(id_a="c1", id_b="c3", winner="a"),
    ]

    results = fit_bradley_terry(candidates, matches)
    assert "c1" in results and "c2" in results and "c3" in results

    # Transitivity must hold: c1 > c2 > c3
    assert results["c1"]["bt_elo"] > results["c2"]["bt_elo"]
    assert results["c2"]["bt_elo"] > results["c3"]["bt_elo"]

    # Uncertainty must be positive
    for cid in ("c1", "c2", "c3"):
        assert results[cid]["bt_uncertainty"] > 0
        assert results[cid]["bt_ci_lower"] < results[cid]["bt_elo"] < results[cid]["bt_ci_upper"]


def test_bradley_terry_order_invariance():
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
    ]
    matches_order_1 = [
        Match(id_a="c1", id_b="c2", winner="a"),
        Match(id_a="c1", id_b="c2", winner="b"),
        Match(id_a="c1", id_b="c2", winner="a"),
    ]
    matches_order_2 = [
        Match(id_a="c1", id_b="c2", winner="b"),
        Match(id_a="c1", id_b="c2", winner="a"),
        Match(id_a="c1", id_b="c2", winner="a"),
    ]

    res1 = fit_bradley_terry(candidates, matches_order_1)
    res2 = fit_bradley_terry(candidates, matches_order_2)

    # MLE fit must be invariant to sequence permutations
    assert res1["c1"]["bt_elo"] == res2["c1"]["bt_elo"]
    assert res1["c2"]["bt_elo"] == res2["c2"]["bt_elo"]
    assert res1["c1"]["bt_uncertainty"] == res2["c1"]["bt_uncertainty"]


def test_uncertainty_shrinks_with_sample_size():
    cands = [Candidate(id=f"c{i}", content=f"Option {i}") for i in range(4)]
    few_matches = [
        Match(id_a="c0", id_b="c1", winner="a"),
        Match(id_a="c2", id_b="c3", winner="a"),
    ]
    many_matches = []
    for i in range(4):
        for j in range(i + 1, 4):
            for _ in range(3):
                many_matches.append(Match(id_a=f"c{i}", id_b=f"c{j}", winner="a"))
                many_matches.append(Match(id_a=f"c{i}", id_b=f"c{j}", winner="b"))

    res_few = fit_bradley_terry(cands, few_matches)
    res_many = fit_bradley_terry(cands, many_matches)

    # Comprehensive match evidence substantially shrinks Fisher information standard error
    assert res_many["c0"]["bt_uncertainty"] < res_few["c0"]["bt_uncertainty"]


def test_replay_stats_populates_bt_fields():
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
    ]
    matches = [
        Match(id_a="c1", id_b="c2", winner="a"),
        Match(id_a="c1", id_b="c2", winner="a"),
    ]

    stats = replay_stats(candidates, matches)
    assert stats["c1"].bt_elo is not None
    assert stats["c1"].bt_uncertainty is not None
    assert stats["c1"].bt_ci_lower is not None
    assert stats["c1"].bt_ci_upper is not None
    assert stats["c1"].bt_elo > stats["c2"].bt_elo


def test_select_matchup_info_gain():
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
        Candidate(id="c3", label="Option 3", content="Text 3"),
    ]
    tourney = Tournament(
        id="t-info-gain",
        title="Test Task",
        prompt="Testing info gain matchmaking",
        candidates=candidates,
        stats={
            "c1": CandidateStats(elo=1300, bt_elo=1300, bt_uncertainty=30.0, matches=4),
            "c2": CandidateStats(elo=1290, bt_elo=1290, bt_uncertainty=35.0, matches=4),
            "c3": CandidateStats(elo=900, bt_elo=900, bt_uncertainty=10.0, matches=10),
        },
        matches=[],
    )

    # In info_gain mode, c1 and c2 should be selected because they have high uncertainty and close ratings at the top
    pair = select_matchup(tourney, mode="info_gain")
    assert pair is not None
    pair_ids = {pair[0].id, pair[1].id}
    assert pair_ids == {"c1", "c2"}


def test_check_convergence_bayesian_z_score():
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
    ]

    # Well-separated case: difference is 100, pooled SE is ~20 => Z = 5.0 >> 1.645
    separated_stats = {
        "c1": CandidateStats(elo=1350, matches=5, bt_elo=1350, bt_uncertainty=15.0),
        "c2": CandidateStats(elo=1200, matches=5, bt_elo=1200, bt_uncertainty=15.0),
    }
    is_conv, conf = check_convergence(candidates, separated_stats)
    assert is_conv is True
    assert conf >= 0.95

    # Ambiguous case: difference is 10, pooled SE is 30 => Z = 0.33 < 1.645
    ambiguous_stats = {
        "c1": CandidateStats(elo=1210, matches=5, bt_elo=1210, bt_uncertainty=20.0),
        "c2": CandidateStats(elo=1200, matches=5, bt_elo=1200, bt_uncertainty=20.0),
    }
    is_conv_amb, conf_amb = check_convergence(candidates, ambiguous_stats)
    assert is_conv_amb is False
    assert conf_amb < 0.95

    # High uncertainty case: diff is 40 points, but SE is large (±150) => Z = 0.19 << 1.645. Must NOT converge.
    high_se_stats = {
        "c1": CandidateStats(elo=1240, matches=3, bt_elo=1240, bt_uncertainty=150.0),
        "c2": CandidateStats(elo=1200, matches=3, bt_elo=1200, bt_uncertainty=150.0),
    }
    is_conv_high_se, _ = check_convergence(candidates, high_se_stats)
    assert is_conv_high_se is False


def test_unplayed_candidate_neutral_rating():
    """Verify that an unplayed candidate remains at neutral 1200.0 and does not float above active candidates."""
    cands = [
        Candidate(id="A", content="A"),
        Candidate(id="B", content="B"),
        Candidate(id="C", content="C"),
        Candidate(id="D", content="D"),
    ]
    matches = [
        Match(id_a="A", id_b="B", winner="a"),
        Match(id_a="A", id_b="C", winner="a"),
        Match(id_a="B", id_b="C", winner="a"),
    ]
    res = fit_bradley_terry(cands, matches)
    assert res["A"]["bt_elo"] > res["B"]["bt_elo"] > res["C"]["bt_elo"]
    assert res["D"]["bt_elo"] == 1200.0
    # Unplayed candidate has highest prior uncertainty
    assert res["D"]["bt_uncertainty"] > res["A"]["bt_uncertainty"]
    assert res["D"]["bt_uncertainty"] > res["B"]["bt_uncertainty"]


def test_api_info_gain_mode_and_stats():
    client = TestClient(app)
    create_res = client.post(
        "/api/tournaments",
        json={
            "title": "MLE Test",
            "prompt": "Evaluate BT MLE",
            "candidates": [
                {"id": "cand_a", "label": "A", "content": "A"},
                {"id": "cand_b", "label": "B", "content": "B"},
            ],
        },
    )
    assert create_res.status_code == 200
    t_id = create_res.json()["id"]

    # Request matchup with info_gain mode
    matchup_res = client.get(f"/api/tournaments/{t_id}/matchup?mode=info_gain")
    assert matchup_res.status_code == 200
    m_data = matchup_res.json()
    assert m_data["mode"] == "info_gain"

    cand_a = m_data["candidate_a"]["id"]
    cand_b = m_data["candidate_b"]["id"]

    # Record 3 consecutive votes for cand_a
    for _ in range(3):
        vote_res = client.post(
            f"/api/tournaments/{t_id}/vote",
            json={
                "id_a": cand_a,
                "id_b": cand_b,
                "winner": "a",
                "voter": "evaluator-1",
            },
        )
        assert vote_res.status_code == 200

    # Fetch tournament and verify Bradley-Terry fields are present
    t_res = client.get(f"/api/tournaments/{t_id}")
    assert t_res.status_code == 200
    stats = t_res.json()["stats"]
    assert stats[cand_a]["bt_elo"] > stats[cand_b]["bt_elo"]
    assert stats[cand_a]["bt_uncertainty"] is not None
    assert stats[cand_a]["bt_ci_lower"] is not None
    assert stats[cand_a]["bt_ci_upper"] is not None


def test_bradley_terry_balanced_prior_unequal_matches():
    """Verify that ties with unequal match counts do not bias ratings toward less-played options."""
    cands = [
        Candidate(id="A", content="Option A"),
        Candidate(id="B", content="Option B"),
        Candidate(id="C", content="Option C"),
    ]
    # 20 A-B ties, 2 B-C ties (everyone has 0 decisive wins)
    matches = [Match(id_a="A", id_b="B", winner="tie") for _ in range(20)] + [
        Match(id_a="B", id_b="C", winner="tie") for _ in range(2)
    ]
    res = fit_bradley_terry(cands, matches)
    # Balanced prior must yield identical 1200.0 ratings across all three
    assert res["A"]["bt_elo"] == 1200.0
    assert res["B"]["bt_elo"] == 1200.0
    assert res["C"]["bt_elo"] == 1200.0


def test_check_convergence_bt_order_reversal():
    """Verify that check_convergence uses Bradley-Terry ordering rather than sequential Elo when BT fields exist."""
    candidates = [
        Candidate(id="c1", label="Option 1", content="Text 1"),
        Candidate(id="c2", label="Option 2", content="Text 2"),
    ]
    # In sequential Elo with recency bias, c2 may end with higher elo, while c1 has higher overall BT MLE
    stats = {
        "c1": CandidateStats(elo=1180.0, bt_elo=1240.0, bt_uncertainty=12.0, matches=16),
        "c2": CandidateStats(elo=1220.0, bt_elo=1160.0, bt_uncertainty=12.0, matches=16),
    }
    is_conv, conf = check_convergence(candidates, stats)
    # Must evaluate c1 as top and c2 as second (diff = +80, not -80), yielding high confidence
    assert is_conv is True
    assert conf >= 0.95

