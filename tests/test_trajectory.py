"""Unit and integration tests for Agent Trajectory evaluation, step triage, and PRM/SPO dataset export."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from showdown.export import export_dataset
from showdown.models import (
    Candidate,
    CandidateStats,
    CreateTournamentRequest,
    StepAnnotationRequest,
    StepAnnotationTag,
    TaskType,
    Tournament,
    VoteRequest,
)
from showdown.server import app, record_vote, storage
from showdown.trajectories import (
    compute_trajectory_summary,
    extract_candidate_steps,
    parse_trajectory,
)


def sample_trajectory_content():
    return json.dumps({
        "steps": [
            {
                "step_index": 0,
                "thought": "Examine the failing unit test.",
                "tool_name": "view_file",
                "tool_args": {"path": "tests/test_foo.py"},
                "tool_output": "def test_foo(): assert foo() == 42",
                "duration_seconds": 0.75,
                "tokens": {"prompt": 120, "completion": 30, "total": 150},
                "status": "success",
            },
            {
                "step_index": 1,
                "thought": "Run wrong destructive command.",
                "tool_name": "run_command",
                "tool_args": {"cmd": "rm -rf ."},
                "tool_output": "Permission denied",
                "duration_seconds": 1.25,
                "tokens": {"total": 80},
                "status": "error",
            },
        ],
        "final_artifact": "def foo(): return 42",
    })


def test_parse_trajectory():
    content = sample_trajectory_content()
    steps = parse_trajectory(content)
    assert steps is not None
    assert len(steps) == 2
    assert steps[0].step_index == 0
    assert steps[0].tool_name == "view_file"
    assert steps[0].duration_seconds == 0.75
    assert steps[0].tokens["total"] == 150
    assert steps[1].status == "error"

    # Non-json returns None
    assert parse_trajectory("Just regular prose text") is None

    # Test falsy fields survive parsing (0 duration, 0 tokens, empty tool_args, False/empty tool_output)
    falsy_steps = parse_trajectory([
        {
            "step_index": 0,
            "tool_name": "bash",
            "tool_args": {},
            "tool_output": False,
            "duration_seconds": 0.0,
            "tokens": 0,
            "status": "failed",
        }
    ])
    assert falsy_steps is not None
    assert falsy_steps[0].duration_seconds == 0.0
    assert falsy_steps[0].tool_args == {}
    assert falsy_steps[0].tool_output == "False"
    assert falsy_steps[0].status == "error"


def test_compute_trajectory_summary():
    steps = parse_trajectory(sample_trajectory_content())
    summary = compute_trajectory_summary(steps)

    assert summary.total_steps == 2
    assert summary.total_tool_calls == 2
    assert summary.total_duration_seconds == 2.0
    assert summary.total_tokens == 230
    assert summary.error_count == 1


def test_trajectory_tournament_endpoints(tmp_path):
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        client = TestClient(app)

        cand_a = Candidate(
            id="agent-a",
            label="Agent A",
            content=sample_trajectory_content(),
        )
        cand_b = Candidate(
            id="agent-b",
            label="Agent B",
            content=json.dumps([
                {
                    "step_index": 0,
                    "thought": "Direct edit",
                    "tool_name": "replace_file",
                    "status": "success",
                    "duration_seconds": 0.5,
                }
            ]),
        )

        create_req = {
            "title": "Agent Debugging Arena",
            "prompt": "Fix failing assertion",
            "task_type": "trajectory",
            "candidates": [cand_a.model_dump(), cand_b.model_dump()],
        }
        res = client.post("/api/tournaments", json=create_req)
        assert res.status_code == 200
        t_data = res.json()
        t_id = t_data["id"]

        # 1. Fetch trajectory for candidate A
        res = client.get(f"/api/tournaments/{t_id}/candidates/agent-a/trajectory")
        assert res.status_code == 200
        traj_data = res.json()
        assert len(traj_data["steps"]) == 2
        assert traj_data["summary"]["error_count"] == 1

        # 2. Annotate step 0 as exemplary
        ann_req = {"tag": "exemplary", "notes": "Great initial exploration", "voter": "russell"}
        res = client.post(f"/api/tournaments/{t_id}/candidates/agent-a/steps/0/annotate", json=ann_req)
        assert res.status_code == 200
        ann_data = res.json()
        assert ann_data["annotation"]["tag"] == "exemplary"
        assert ann_data["annotation"]["voter"] == "russell"

        # 3. Nonexistent step index returns 404
        bad_ann_res = client.post(f"/api/tournaments/{t_id}/candidates/agent-a/steps/999/annotate", json=ann_req)
        assert bad_ann_res.status_code == 404
        assert "Step index 999 not found" in bad_ann_res.json()["detail"]

        # 4. Another voter annotates step 0 as well -> both voters must be preserved!
        bob_req = {"tag": "inefficient", "notes": "Could be faster", "voter": "bob"}
        res = client.post(f"/api/tournaments/{t_id}/candidates/agent-a/steps/0/annotate", json=bob_req)
        assert res.status_code == 200

        # 5. Annotate step 1 as incorrect
        ann_req2 = {"tag": "incorrect", "notes": "Dangerous rm command", "voter": "russell"}
        res = client.post(f"/api/tournaments/{t_id}/candidates/agent-a/steps/1/annotate", json=ann_req2)
        assert res.status_code == 200

        # 6. Verify annotations appear in trajectory endpoint
        res = client.get(f"/api/tournaments/{t_id}/candidates/agent-a/trajectory")
        steps = res.json()["steps"]
        assert len(steps[0]["annotations"]) == 2
        voters_step0 = {a["voter"] for a in steps[0]["annotations"]}
        assert voters_step0 == {"russell", "bob"}
        assert steps[1]["annotations"][0]["tag"] == "incorrect"

        # 7. Export SPO / PRM dataset
        res = client.get("/api/export?format=spo")
        assert res.status_code == 200
        spo_data = res.json()
        assert len(spo_data) == 3
        tags = {r["tag"] for r in spo_data}
        assert tags == {"exemplary", "inefficient", "incorrect"}

        # 8. Delete annotation for russell only; bob must remain
        del_res = client.delete(f"/api/tournaments/{t_id}/candidates/agent-a/steps/0/annotate?voter=russell")
        assert del_res.status_code == 200
        res = client.get(f"/api/tournaments/{t_id}/candidates/agent-a/trajectory")
        assert len(res.json()["steps"][0]["annotations"]) == 1
        assert res.json()["steps"][0]["annotations"][0]["voter"] == "bob"

    finally:
        storage.data_dir = orig_dir


def test_mcp_trajectory_tools(tmp_path):
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        from showdown.mcp_server import showdown_annotate_step, showdown_create_tournament, showdown_get_trajectory

        res = showdown_create_tournament(
            title="MCP Trajectory Arena",
            task_type="trajectory",
            candidates=[
                {"id": "c1", "content": sample_trajectory_content()},
                {"id": "c2", "content": json.dumps([{"step_index": 0, "thought": "fast"}])},
            ],
        )
        t_id = res["tournament_id"]

        traj_res = showdown_get_trajectory(tournament_id=t_id, candidate_id="c1")
        assert "steps" in traj_res
        assert len(traj_res["steps"]) == 2

        ann_res = showdown_annotate_step(
            tournament_id=t_id,
            candidate_id="c1",
            step_index=0,
            tag="exemplary",
            notes="Very systematic",
            voter="agent_eval",
        )
        assert ann_res["status"] == "annotated"
        assert ann_res["annotation"]["tag"] == "exemplary"
    finally:
        storage.data_dir = orig_dir
