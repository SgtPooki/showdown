"""Tests for multi-tournament dataset aggregation, train/val splits, and alignment adapters."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from showdown.export import export_dataset, find_tournament_lineages, format_jsonl
from showdown.models import Candidate, CandidateStats, Match, TaskType, Tournament, TriageRecord, TriageStatus
from showdown.server import app
from showdown.storage import Storage


def create_sample_tournaments() -> list[Tournament]:
    c1 = Candidate(id="c1", label="Opt 1", content="print('hello')")
    c2 = Candidate(id="c2", label="Opt 2", content="print('world')")
    c3 = Candidate(id="c3", label="Opt 3", content="print('foo')")

    t1 = Tournament(
        id="t-code-1",
        title="Code Style 1",
        prompt="Write a hello script",
        task_type=TaskType.CODE,
        candidates=[c1, c2],
        stats={
            "c1": CandidateStats(elo=1300.0, bt_elo=1300.0),
            "c2": CandidateStats(elo=1100.0, bt_elo=1100.0),
        },
        matches=[
            Match(id_a="c1", id_b="c2", winner="a", voter="alice", notes="Cleaner syntax"),
            Match(id_a="c1", id_b="c2", winner="a", voter="bob", notes="Better naming"),
        ],
        triage={
            "c1": TriageRecord(status=TriageStatus.FAVORITE, notes="Top choice"),
            "c2": TriageRecord(status=TriageStatus.DISLIKED, notes="Too messy"),
        },
    )

    t2 = Tournament(
        id="t-code-2",
        title="Code Style 2 (Chained)",
        prompt="Write a foo script",
        task_type=TaskType.CODE,
        parent_ids=["t-code-1"],  # Chained child
        candidates=[c2, c3],
        stats={
            "c2": CandidateStats(elo=1250.0, bt_elo=1250.0),
            "c3": CandidateStats(elo=1150.0, bt_elo=1150.0),
        },
        matches=[
            Match(id_a="c2", id_b="c3", winner="a", voter="alice", notes="More robust"),
        ],
    )

    t3 = Tournament(
        id="t-text-1",
        title="Text Copy",
        prompt="Draft a slogan",
        task_type=TaskType.TEXT,
        candidates=[
            Candidate(id="t1", content="Slogan A"),
            Candidate(id="t2", content="Slogan B"),
        ],
        stats={
            "t1": CandidateStats(elo=1220.0, bt_elo=1220.0),
            "t2": CandidateStats(elo=1180.0, bt_elo=1180.0),
        },
        matches=[
            Match(id_a="t1", id_b="t2", winner="a", voter="carol", notes="Catchy"),
        ],
    )

    return [t1, t2, t3]


def test_find_tournament_lineages():
    tourneys = create_sample_tournaments()
    lineages = find_tournament_lineages(tourneys)
    # t-code-1 and t-code-2 must belong to the exact same cluster root
    assert lineages["t-code-1"] == lineages["t-code-2"]
    # t-text-1 is independent
    assert lineages["t-text-1"] != lineages["t-code-1"]


def test_export_dataset_dpo_and_dedup():
    tourneys = create_sample_tournaments()
    # Duplicate match added
    tourneys[0].matches.append(Match(id_a="c1", id_b="c2", winner="a", voter="alice", notes="Duplicate"))

    # With dedup=True (default), duplicates should collapse
    recs_dedup = export_dataset(tourneys, format="dpo", dedup=True)
    recs_nodedup = export_dataset(tourneys, format="dpo", dedup=False)

    assert len(recs_dedup) < len(recs_nodedup)
    first = recs_dedup[0]
    assert "prompt" in first and "chosen" in first and "rejected" in first
    assert "critique" in first


def test_export_dataset_pairwise_margins():
    tourneys = create_sample_tournaments()
    recs = export_dataset(tourneys, format="pairwise_margins")
    assert len(recs) > 0
    first = recs[0]
    assert "margin" in first
    # In t1, c1 (1300) beat c2 (1100), so margin should be 200.0
    assert first["margin"] == 200.0


def test_export_dataset_kto():
    tourneys = create_sample_tournaments()
    kto_recs = export_dataset(tourneys, format="kto")
    assert len(kto_recs) == 2  # c1 is liked (label=True), c2 is disliked (label=False)
    labels = {r["label"] for r in kto_recs}
    assert labels == {True, False}


def test_export_dataset_filter_task_type():
    tourneys = create_sample_tournaments()
    code_recs = export_dataset(tourneys, format="dpo", task_type="code")
    text_recs = export_dataset(tourneys, format="dpo", task_type="text")

    for r in code_recs:
        assert r["task_type"] == "code"
    for r in text_recs:
        assert r["task_type"] == "text"


def test_export_dataset_lineage_split():
    tourneys = create_sample_tournaments()
    result = export_dataset(tourneys, format="dpo", split=0.6, split_by="lineage", seed=42)

    assert isinstance(result, dict)
    assert "train" in result and "val" in result
    assert "split_stats" in result

    train_tids = {r["tournament_id"] for r in result["train"]}
    val_tids = {r["tournament_id"] for r in result["val"]}

    # Crucial test for lineage safety: t-code-1 and t-code-2 must NEVER be split across train and val
    if "t-code-1" in train_tids:
        assert "t-code-2" in train_tids
        assert "t-code-2" not in val_tids
    elif "t-code-1" in val_tids:
        assert "t-code-2" in val_tids
        assert "t-code-2" not in train_tids


def test_api_export_all():
    client = TestClient(app)
    # Test JSON endpoint
    res = client.get("/api/export?format=dpo")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)

    # Test JSONL endpoint
    res_jsonl = client.get("/api/export?format=dpo&jsonl=true")
    assert res_jsonl.status_code == 200
    lines = [line for line in res_jsonl.text.strip().split("\n") if line]
    assert len(lines) == len(data)

    # Test with split
    res_split = client.get("/api/export?format=dpo&split=0.8")
    assert res_split.status_code == 200
    split_data = res_split.json()
    assert "train" in split_data and "val" in split_data


def test_mcp_export_dataset_tool():
    from showdown.mcp_server import mcp_export_dataset
    res = mcp_export_dataset(format="dpo", split=0.8)
    assert "train" in res and "val" in res
    assert "split_stats" in res
