"""Unit tests for LLM-as-a-judge automated tournament runner and position-bias mitigation."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from showdown.engine import compute_inter_annotator_agreement, select_matchup
from showdown.judge import (
    DEFAULT_RUBRICS,
    build_judge_prompt,
    evaluate_pair,
    parse_judge_output,
    run_tournament_judge,
)
from showdown.models import Candidate, CandidateStats, Match, TaskType, Tournament
from showdown.server import app, storage


@pytest.fixture
def sample_tournament():
    return Tournament(
        id="test_judge_tourney",
        title="Judge Test Tournament",
        prompt="Write a clean Python singleton.",
        task_type=TaskType.CODE,
        candidates=[
            Candidate(id="c1", label="Option 1", content="class Singleton:\n    _inst = None"),
            Candidate(id="c2", label="Option 2", content="def singleton(cls):\n    instances = {}"),
            Candidate(id="c3", label="Option 3", content="import threading\nclass ThreadSafeSingleton: pass"),
        ],
        stats={
            "c1": CandidateStats(elo=1200.0, matches=0),
            "c2": CandidateStats(elo=1200.0, matches=0),
            "c3": CandidateStats(elo=1200.0, matches=0),
        },
    )


def test_parse_judge_output():
    # 1. Valid fenced JSON
    raw_1 = '```json\n{"winner": "A", "critique": "Candidate A is thread-safe and idiomatic."}\n```'
    w1, c1 = parse_judge_output(raw_1)
    assert w1 == "a"
    assert "thread-safe" in c1

    # 2. Bare JSON in response text
    raw_2 = 'Here is my evaluation:\n{"winner": "B", "critique": "Candidate B has lower overhead."}\nHope this helps!'
    w2, c2 = parse_judge_output(raw_2)
    assert w2 == "b"
    assert "lower overhead" in c2

    # 3. Tie decision
    raw_3 = '{"winner": "TIE", "critique": "Both implementations are essentially equivalent."}'
    w3, c3 = parse_judge_output(raw_3)
    assert w3 == "tie"

    # 4. Regex fallback matching "Candidate A"
    raw_4 = 'After careful review, winner: Candidate A. Candidate A handles edge cases.'
    w4, c4 = parse_judge_output(raw_4)
    assert w4 == "a"

    # 5. Unparseable output returns None
    raw_5 = 'I cannot decide between these two candidates because the model refused to evaluate.'
    w5, c5 = parse_judge_output(raw_5)
    assert w5 is None


def test_build_judge_prompt(sample_tournament):
    c1, c2 = sample_tournament.candidates[0], sample_tournament.candidates[1]
    prompt = build_judge_prompt(
        candidate_a=c1,
        candidate_b=c2,
        task_prompt=sample_tournament.prompt,
        rubric=None,
        task_type=sample_tournament.task_type,
    )
    assert "Write a clean Python singleton." in prompt
    assert "Candidate A:" in prompt
    assert "Candidate B:" in prompt
    assert "algorithmic efficiency" in prompt  # from default code rubric


def test_evaluate_pair_consistent_win():
    c1 = Candidate(id="c1", content="Fast and correct")
    c2 = Candidate(id="c2", content="Buggy and slow")

    # Presentation 1 (c1 vs c2): Judge picks A (c1)
    # Presentation 2 (c2 vs c1): Judge picks B (c1)
    side_effects = [
        '{"winner": "A", "critique": "A is significantly faster."}',
        '{"winner": "B", "critique": "B is significantly faster."}',
    ]
    with patch("showdown.judge.call_judge_backend", side_effect=side_effects):
        res = evaluate_pair(c1, c2, task_prompt="Test", backend="omp", swap_positions=True)
        assert res["winner"] == "a"
        assert res["swapped_consistent"] is True
        assert "[Consistent]" in res["critique"]


def test_evaluate_pair_position_bias_abstains_without_moving_elo(sample_tournament):
    c1 = sample_tournament.candidates[0]
    c2 = sample_tournament.candidates[1]

    # Position bias: Judge always picks the first option (slot 1) regardless of content
    side_effects = [
        '{"winner": "A", "critique": "First candidate looked better."}',
        '{"winner": "A", "critique": "First candidate looked better."}',
    ]
    with patch("showdown.judge.call_judge_backend", side_effect=side_effects):
        res = evaluate_pair(c1, c2, task_prompt="Test", backend="omp", swap_positions=True)
        assert res["winner"] == "abstain"
        assert res["swapped_consistent"] is False
        assert "[Position-Bias Contradiction]" in res["critique"]

    # Now verify that when run through tournament judge, abstain does NOT alter Elo or matches
    with patch("showdown.judge.call_judge_backend", side_effect=side_effects):
        judge_res = run_tournament_judge(
            tournament=sample_tournament,
            rounds=1,
            backend="omp",
            swap_positions=True,
        )
        assert judge_res.matches_evaluated == 1
        assert judge_res.contradictions == 1
        # Crucial invariant: ratings did not shift
        assert sample_tournament.stats["c1"].elo == 1200.0
        assert sample_tournament.stats["c2"].elo == 1200.0
        assert sample_tournament.stats["c1"].matches == 0
        assert sample_tournament.stats["c2"].matches == 0


def test_evaluate_pair_nondeterminism_abstains():
    c1 = Candidate(id="c1", content="Alpha")
    c2 = Candidate(id="c2", content="Beta")

    # Presentation 1: Tie. Presentation 2: Candidate A (c2)
    side_effects = [
        '{"winner": "TIE", "critique": "Unsure on first read."}',
        '{"winner": "A", "critique": "A is definitely better."}',
    ]
    with patch("showdown.judge.call_judge_backend", side_effect=side_effects):
        res = evaluate_pair(c1, c2, task_prompt="Test", swap_positions=True)
        assert res["winner"] == "abstain"
        assert res["swapped_consistent"] is False
        assert "[Judge Nondeterminism]" in res["critique"]


def test_evaluate_pair_unparseable_raises():
    c1 = Candidate(id="c1", content="Alpha")
    c2 = Candidate(id="c2", content="Beta")

    with patch("showdown.judge.call_judge_backend", return_value="I apologize, but I am unable to answer."):
        with pytest.raises(ValueError, match="could not be parsed"):
            evaluate_pair(c1, c2, task_prompt="Test", swap_positions=True)


def test_evaluate_pair_no_swap():
    c1 = Candidate(id="c1", content="Alpha")
    c2 = Candidate(id="c2", content="Beta")

    with patch("showdown.judge.call_judge_backend", return_value='{"winner": "B", "critique": "B won"}'):
        res = evaluate_pair(c1, c2, task_prompt="Test", swap_positions=False)
        assert res["winner"] == "b"
        assert res["swapped_consistent"] is True


def test_run_tournament_judge_loop(sample_tournament):
    with patch(
        "showdown.judge.call_judge_backend",
        return_value='{"winner": "A", "critique": "Option A is cleaner."}',
    ):
        res = run_tournament_judge(
            tournament=sample_tournament,
            rounds=3,
            backend="omp",
            swap_positions=False,
        )

        assert res.matches_evaluated == 3
        assert len(sample_tournament.matches) == 3
        assert sample_tournament.matches[0].voter == "judge:omp"
        assert "Option A is cleaner" in sample_tournament.matches[0].notes


def test_human_vs_judge_agreement_computation():
    matches = [
        # Pair (c1, c2): Human and Judge agree that c1 wins
        Match(id_a="c1", id_b="c2", winner="a", voter="human"),
        Match(id_a="c1", id_b="c2", winner="a", voter="judge:claude"),

        # Pair (c2, c3): Human says c2 wins, Judge says c3 wins (reversal)
        Match(id_a="c2", id_b="c3", winner="a", voter="human"),
        Match(id_a="c2", id_b="c3", winner="b", voter="judge:claude"),
    ]

    agreement_data = compute_inter_annotator_agreement(matches)
    assert agreement_data["overall_agreement_rate"] == 0.5
    assert agreement_data["human_vs_judge"] is not None
    assert agreement_data["human_vs_judge"]["shared_pairs_evaluated"] == 2
    assert agreement_data["human_vs_judge"]["agreements"] == 1
    assert agreement_data["human_vs_judge"]["agreement_rate"] == 0.5


def test_select_matchup_routing_modes(sample_tournament):
    # Add a match with position-bias contradiction note
    sample_tournament.matches.append(
        Match(
            id_a="c1",
            id_b="c3",
            winner="abstain",
            voter="judge:omp",
            notes="[Position-Bias Contradiction] Picked slot 1 both times",
        )
    )

    # In controversial mode, (c1, c3) should be prioritized
    pair = select_matchup(sample_tournament, mode="controversial")
    assert pair is not None
    pair_ids = {pair[0].id, pair[1].id}
    assert pair_ids == {"c1", "c3"}

    # In close mode, candidates with closest ratings are paired
    sample_tournament.stats["c1"].elo = 1250.0
    sample_tournament.stats["c1"].matches = 2
    sample_tournament.stats["c2"].elo = 1205.0
    sample_tournament.stats["c2"].matches = 2
    sample_tournament.stats["c3"].elo = 1200.0
    sample_tournament.stats["c3"].matches = 2

    pair_close = select_matchup(sample_tournament, mode="close")
    assert pair_close is not None
    pair_close_ids = {pair_close[0].id, pair_close[1].id}
    # c2 and c3 have delta 5, closer than c1 and c2 (delta 45)
    assert pair_close_ids == {"c2", "c3"}


def test_judge_api_endpoint(tmp_path, sample_tournament):
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        storage.save_tournament(sample_tournament)
        client = TestClient(app)

        with patch(
            "showdown.judge.call_judge_backend",
            return_value='{"winner": "A", "critique": "Solid syntax."}',
        ):
            resp = client.post(
                f"/api/tournaments/{sample_tournament.id}/judge",
                json={
                    "backend": "omp",
                    "rounds": 2,
                    "swap_positions": False,
                    "mode": "active",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["tournament_id"] == sample_tournament.id
            assert data["matches_evaluated"] == 2
            assert data["voter"] == "judge:omp"
            assert len(data["results"]) == 2
    finally:
        storage.data_dir = orig_dir
