"""Unit tests for multi-tenant voter slicing, convergence, and chained tournaments."""

from showdown.engine import check_convergence, replay_stats, select_matchup
from showdown.models import (
    AcceptCandidateRequest,
    Candidate,
    CandidateStats,
    CreateTournamentRequest,
    Match,
    TaskType,
    Tournament,
    VoteRequest,
)
from showdown.server import (
    accept_candidate,
    create_tournament,
    get_matchup,
    get_tournament,
    get_tournament_chain,
    record_vote,
    storage,
)
from showdown.storage import Storage


def test_replay_stats_voter_slicing():
    # Arrange
    cands = [
        Candidate(id="c1", label="Option 1", content="One"),
        Candidate(id="c2", label="Option 2", content="Two"),
        Candidate(id="c3", label="Option 3", content="Three"),
    ]
    matches = [
        Match(
            id_a="c1",
            id_b="c2",
            winner="a",
            elo_a_before=1200.0,
            elo_b_before=1200.0,
            elo_a_after=1224.0,
            elo_b_after=1176.0,
            voter="evaluator_alpha",
        ),
        Match(
            id_a="c2",
            id_b="c3",
            winner="b",
            elo_a_before=1176.0,
            elo_b_before=1200.0,
            elo_a_after=1152.0,
            elo_b_after=1224.0,
            voter="evaluator_beta",
        ),
    ]

    # Act
    stats_alpha = replay_stats(cands, matches, voter="evaluator_alpha")
    stats_beta = replay_stats(cands, matches, voter="evaluator_beta")
    stats_pooled = replay_stats(cands, matches, voter="pooled")

    # Assert
    # Alpha only voted on c1 vs c2
    assert stats_alpha["c1"].matches == 1
    assert stats_alpha["c1"].wins == 1
    assert stats_alpha["c1"].elo > 1200.0
    assert stats_alpha["c2"].matches == 1
    assert stats_alpha["c2"].losses == 1
    assert stats_alpha["c2"].elo < 1200.0
    assert stats_alpha["c3"].matches == 0
    assert stats_alpha["c3"].elo == 1200.0

    # Beta only voted on c2 vs c3
    assert stats_beta["c1"].matches == 0
    assert stats_beta["c1"].elo == 1200.0
    assert stats_beta["c2"].matches == 1
    assert stats_beta["c2"].losses == 1
    assert stats_beta["c3"].matches == 1
    assert stats_beta["c3"].wins == 1
    assert stats_beta["c3"].elo > 1200.0

    # Pooled includes all matches
    assert stats_pooled["c1"].matches == 1
    assert stats_pooled["c2"].matches == 2
    assert stats_pooled["c3"].matches == 1


def test_check_convergence_detection():
    # Arrange
    cands = [
        Candidate(id="c1", label="Option 1", content="One"),
        Candidate(id="c2", label="Option 2", content="Two"),
    ]

    # Case 1: No matches played yet
    stats_empty = {c.id: CandidateStats() for c in cands}
    converged_empty, conf_empty = check_convergence(cands, stats_empty, min_matches_per_cand=2)
    assert converged_empty is False
    assert conf_empty == 0.0

    # Case 2: Matches played but margin too small
    stats_close = {
        "c1": CandidateStats(elo=1210.0, matches=3, wins=2),
        "c2": CandidateStats(elo=1200.0, matches=3, losses=2),
    }
    converged_close, conf_close = check_convergence(cands, stats_close, min_matches_per_cand=2, lead_margin=35.0)
    assert converged_close is False
    assert conf_close < 1.0

    # Case 3: Statistically separated winner
    stats_separated = {
        "c1": CandidateStats(elo=1255.0, matches=4, wins=4),
        "c2": CandidateStats(elo=1180.0, matches=4, losses=4),
    }
    converged_sep, conf_sep = check_convergence(cands, stats_separated, min_matches_per_cand=2, lead_margin=35.0)
    assert converged_sep is True
    assert conf_sep == 1.0


def test_small_pool_least_matched_selection():
    # Arrange: 4 candidates, one has 0 matches, others have 2
    cands = [
        Candidate(id="c1", label="Option 1", content="One"),
        Candidate(id="c2", label="Option 2", content="Two"),
        Candidate(id="c3", label="Option 3", content="Three"),
        Candidate(id="c4", label="Option 4", content="Four"),
    ]
    t = Tournament(
        id="small_pool_test",
        title="Small Pool",
        task_type=TaskType.TEXT,
        candidates=cands,
        stats={
            "c1": CandidateStats(matches=2),
            "c2": CandidateStats(matches=0),
            "c3": CandidateStats(matches=2),
            "c4": CandidateStats(matches=2),
        },
    )

    # Act
    pair = select_matchup(t)

    # Assert: In small pool (<= 6), candidate with 0 matches must be selected
    assert pair is not None
    pair_ids = {pair[0].id, pair[1].id}
    assert "c2" in pair_ids


def test_server_voter_slicing_and_chaining(tmp_path):
    # Arrange
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        # Step 1: Create Parent Tournament (Stage 1)
        req_stage1 = CreateTournamentRequest(
            title="Act I: The Beginning",
            task_type=TaskType.TEXT,
            candidates=[
                Candidate(id="p1", label="Hook A", content="The ship fell silently into orbit."),
                Candidate(id="p2", label="Hook B", content="Alarms pierced the quiet bridge."),
            ],
        )
        stage1 = create_tournament(req_stage1)
        assert stage1.id is not None

        # Step 2: Record votes with distinct voter tags
        record_vote(
            stage1.id,
            VoteRequest(
                id_a="p1",
                id_b="p2",
                winner="a",
                voter="voter_human",
                notes="Better atmospheric tension",
            ),
        )
        record_vote(
            stage1.id,
            VoteRequest(
                id_a="p1",
                id_b="p2",
                winner="b",
                voter="voter_bot",
                notes="More kinetic start",
            ),
        )

        # Step 3: Verify voter list and sliced stats via endpoint
        t_all = get_tournament(stage1.id)
        assert sorted(t_all.voters) == ["voter_bot", "voter_human"]

        t_human = get_tournament(stage1.id, voter="voter_human")
        assert t_human.stats["p1"].wins == 1
        assert t_human.stats["p1"].elo > 1200.0

        t_bot = get_tournament(stage1.id, voter="voter_bot")
        assert t_bot.stats["p2"].wins == 1
        assert t_bot.stats["p2"].elo > 1200.0

        # Step 4: Verify matchup convergence metadata
        matchup_data = get_matchup(stage1.id)
        assert "converged" in matchup_data
        assert "confidence" in matchup_data

        # Step 5: Accept Stage 1 winner
        accept_candidate(stage1.id, AcceptCandidateRequest(candidate_id="p1"))

        # Step 6: Create Child Tournament (Stage 2) chained to Stage 1
        req_stage2 = CreateTournamentRequest(
            title="Act II: The Escalation",
            task_type=TaskType.TEXT,
            candidates=[
                Candidate(id="e1", label="Escalation A", content="Thrusters fired, barely clearing the debris field."),
                Candidate(id="e2", label="Escalation B", content="The derelict hull loomed closer in the darkness."),
            ],
            parent_ids=[stage1.id],
            context="The ship fell silently into orbit.",
        )
        stage2 = create_tournament(req_stage2)
        assert stage2.parent_ids == [stage1.id]
        assert stage2.context == "The ship fell silently into orbit."

        # Accept Stage 2 winner
        accept_candidate(stage2.id, AcceptCandidateRequest(candidate_id="e1"))

        # Step 7: Call chain lineage endpoint on Stage 2
        chain_data = get_tournament_chain(stage2.id)

        assert chain_data["count"] == 2
        assert len(chain_data["stages"]) == 2
        # Chronological lineage: stage1 first, then stage2
        assert chain_data["stages"][0]["tournament_id"] == stage1.id
        assert chain_data["stages"][0]["accepted_candidate"]["id"] == "p1"
        assert chain_data["stages"][1]["tournament_id"] == stage2.id
        assert chain_data["stages"][1]["accepted_candidate"]["id"] == "e1"

        expected_text = (
            "The ship fell silently into orbit.\n\n"
            "Thrusters fired, barely clearing the debris field."
        )
        assert chain_data["assembled_content"] == expected_text

    finally:
        storage.data_dir = orig_dir
