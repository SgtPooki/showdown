"""Unit tests for Showdown MCP (Model Context Protocol) server."""

import pytest
from showdown.mcp_server import server
from showdown.models import TaskType
from showdown.server import storage
from showdown.storage import Storage


@pytest.mark.anyio
async def test_mcp_list_tools():
    # Act
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    # Assert
    expected = [
        "showdown_create_tournament",
        "showdown_wait_for_decision",
        "showdown_get_leaderboard",
        "showdown_record_vote",
        "showdown_evolve_candidates",
        "showdown_get_chain",
        "showdown_export_dataset",
    ]
    for exp in expected:
        assert exp in tool_names, f"Missing MCP tool: {exp}"


@pytest.mark.anyio
async def test_mcp_create_vote_and_leaderboard(tmp_path):
    # Arrange
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        # 1. Create Tournament via MCP
        create_res = await server.call_tool(
            "showdown_create_tournament",
            {
                "title": "MCP Test Arena",
                "prompt": "Pick the best tagline",
                "task_type": "text",
                "candidates": [
                    {"id": "c1", "label": "Candidate A", "content": "Fast as light"},
                    {"id": "c2", "label": "Candidate B", "content": "Solid as granite"},
                ],
            },
        )
        assert create_res.is_error is False
        t_data = create_res.structured_content["result"]
        t_id = t_data["tournament_id"]
        assert "http://localhost:8000/?t=" in t_data["url"]
        assert t_data["candidate_count"] == 2

        # 2. Record Vote via MCP
        vote_res = await server.call_tool(
            "showdown_record_vote",
            {
                "tournament_id": t_id,
                "id_a": "c1",
                "id_b": "c2",
                "winner": "a",
                "voter": "claude-agent",
                "notes": "Much more energetic",
            },
        )
        assert vote_res.is_error is False
        vote_data = vote_res.structured_content["result"]
        assert vote_data["status"] == "recorded"
        assert vote_data["elo_a_after"] > 1200.0
        assert vote_data["elo_b_after"] < 1200.0

        # 3. Get Leaderboard via MCP
        lb_res = await server.call_tool(
            "showdown_get_leaderboard",
            {"tournament_id": t_id},
        )
        assert lb_res.is_error is False
        lb_data = lb_res.structured_content["result"]
        assert len(lb_data["leaderboard"]) == 2
        assert lb_data["leaderboard"][0]["id"] == "c1"
        assert lb_data["available_voters"] == ["claude-agent"]

        # 4. Sliced Leaderboard via MCP
        lb_sliced_res = await server.call_tool(
            "showdown_get_leaderboard",
            {"tournament_id": t_id, "voter": "claude-agent"},
        )
        assert lb_sliced_res.is_error is False
        lb_sliced_data = lb_sliced_res.structured_content["result"]
        assert lb_sliced_data["voter_filter"] == "claude-agent"
        assert lb_sliced_data["leaderboard"][0]["id"] == "c1"

        # 5. Export DPO Dataset via MCP
        exp_res = await server.call_tool(
            "showdown_export_dataset",
            {"tournament_id": t_id, "format": "dpo"},
        )
        assert exp_res.is_error is False
        exp_data = exp_res.structured_content["result"]
        assert exp_data["count"] == 1
        assert exp_data["records"][0]["chosen"] == "Fast as light"
        assert exp_data["records"][0]["rejected"] == "Solid as granite"

    finally:
        storage.data_dir = orig_dir


@pytest.mark.anyio
async def test_mcp_wait_and_chain(tmp_path):
    # Arrange
    orig_dir = storage.data_dir
    storage.data_dir = tmp_path

    try:
        # Create Stage 1
        res1 = await server.call_tool(
            "showdown_create_tournament",
            {
                "title": "Stage 1",
                "prompt": "Hook",
                "candidates": [
                    {"id": "h1", "content": "The reactor flared blue."},
                    {"id": "h2", "content": "Silence engulfed the bridge."},
                ],
            },
        )
        s1_id = res1.structured_content["result"]["tournament_id"]

        # Accept winner on Stage 1
        from showdown.server import accept_candidate as s_accept
        from showdown.models import AcceptCandidateRequest
        s_accept(s1_id, AcceptCandidateRequest(candidate_id="h1"))

        # Test wait_for_decision returns completed immediately
        wait_res = await server.call_tool(
            "showdown_wait_for_decision",
            {"tournament_id": s1_id, "timeout": 2},
        )
        assert wait_res.is_error is False
        wait_data = wait_res.structured_content["result"]
        assert wait_data["completed"] is True
        assert wait_data["accepted_candidate"]["id"] == "h1"

        # Create Stage 2 chained to Stage 1
        res2 = await server.call_tool(
            "showdown_create_tournament",
            {
                "title": "Stage 2",
                "prompt": "Escalation",
                "parent_ids": [s1_id],
                "context": "The reactor flared blue.",
                "candidates": [
                    {"id": "e1", "content": "Containment shields shattered instantly."},
                ],
            },
        )
        s2_id = res2.structured_content["result"]["tournament_id"]
        s_accept(s2_id, AcceptCandidateRequest(candidate_id="e1"))

        # Get Chain via MCP
        chain_res = await server.call_tool(
            "showdown_get_chain",
            {"tournament_id": s2_id},
        )
        assert chain_res.is_error is False
        chain_data = chain_res.structured_content["result"]
        assert chain_data["count"] == 2
        assert "The reactor flared blue.\n\nContainment shields shattered instantly." in chain_data["assembled_content"]

    finally:
        storage.data_dir = orig_dir
