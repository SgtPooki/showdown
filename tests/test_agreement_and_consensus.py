"""Unit tests for inter-annotator agreement and consensus-gated DPO export."""

from showdown.engine import compute_inter_annotator_agreement
from showdown.models import (
    Candidate,
    CreateTournamentRequest,
    Match,
    TaskType,
    VoteRequest,
)
from showdown.server import (
    create_tournament,
    export_tournament,
    get_inter_annotator_agreement,
    record_vote,
    storage,
)
from showdown.storage import Storage


def test_compute_inter_annotator_agreement_pure():
    # Arrange: 3 matches across 2 shared pairs
    # Pair (c1, c2): both alpha and beta agree (winner = c1)
    # Pair (c2, c3): alpha picked c2, beta picked c3 (reversal)
    matches = [
        # Pair (c1, c2)
        Match(id_a="c1", id_b="c2", winner="a", voter="evaluator_alpha"),
        Match(id_a="c2", id_b="c1", winner="b", voter="evaluator_beta"),  # beta picked c1 too
        # Pair (c2, c3)
        Match(id_a="c2", id_b="c3", winner="a", voter="evaluator_alpha"),  # alpha picked c2
        Match(id_a="c2", id_b="c3", winner="b", voter="evaluator_beta"),   # beta picked c3 (reversal)
    ]

    # Act
    res = compute_inter_annotator_agreement(matches)

    # Assert
    assert res["voters"] == ["evaluator_alpha", "evaluator_beta"]
    assert res["shared_pairs_evaluated"] == 2
    assert len(res["evaluator_pairs"]) == 1

    pair_stat = res["evaluator_pairs"][0]
    assert pair_stat["agreements"] == 1
    assert pair_stat["reversals"] == 1
    assert pair_stat["agreement_rate"] == 0.5
    assert pair_stat["reversal_rate"] == 0.5
    assert res["overall_agreement_rate"] == 0.5


def test_agreement_endpoint_and_consensus_export(tmp_path):
    # Arrange
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        t_req = CreateTournamentRequest(
            title="Consensus Arena",
            task_type=TaskType.TEXT,
            candidates=[
                Candidate(id="c1", label="Option 1", content="One"),
                Candidate(id="c2", label="Option 2", content="Two"),
                Candidate(id="c3", label="Option 3", content="Three"),
            ],
        )
        t = create_tournament(t_req)

        # Pair c1 vs c2: Both voters agree on c1
        record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="eval_1"))
        record_vote(t.id, VoteRequest(id_a="c1", id_b="c2", winner="a", voter="eval_2"))

        # Pair c2 vs c3: eval_1 picked c2, eval_2 picked c3 (disagreement)
        record_vote(t.id, VoteRequest(id_a="c2", id_b="c3", winner="a", voter="eval_1"))
        record_vote(t.id, VoteRequest(id_a="c2", id_b="c3", winner="b", voter="eval_2"))

        # 1. Test agreement endpoint
        agree_data = get_inter_annotator_agreement(t.id)
        assert agree_data["tournament_id"] == t.id
        assert agree_data["overall_agreement_rate"] == 0.5
        assert agree_data["shared_pairs_evaluated"] == 2

        # 2. Test voter-filtered DPO export (only eval_1)
        dpo_eval1 = export_tournament(t.id, format="dpo", voter="eval_1")
        assert len(dpo_eval1) == 2
        assert all(row["voter"] == "eval_1" for row in dpo_eval1)

        # 3. Test strict consensus export (must have unanimous agreement and >=2 voters)
        dpo_strict = export_tournament(t.id, format="dpo", consensus="strict")
        # Only pair (c1, c2) had 100% agreement between eval_1 and eval_2
        assert len(dpo_strict) == 1
        assert dpo_strict[0]["chosen_id"] == "c1"
        assert dpo_strict[0]["rejected_id"] == "c2"
        assert dpo_strict[0]["agreement_rate"] == 1.0
        assert dpo_strict[0]["votes_chosen"] == 2
        assert dpo_strict[0]["votes_rejected"] == 0

        # 4. Test min_agreement threshold (e.g. min 0.8)
        dpo_threshold = export_tournament(t.id, format="dpo", min_agreement=0.8)
        assert len(dpo_threshold) == 1
        assert dpo_threshold[0]["chosen_id"] == "c1"

    finally:
        storage.data_dir = orig_dir
