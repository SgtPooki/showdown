"""Tests for Extensible Agent Providers, Multi-Agent Fan-Out, and Cross-Model Leaderboard."""

import json
import os
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from showdown.models import (
    Candidate,
    CandidateStats,
    Match,
    TaskType,
    Tournament,
)
from showdown.providers import (
    AgentProvider,
    ClaudeCliProvider,
    ClaudeProvider,
    CodexCliProvider,
    CodexProvider,
    CursorCliProvider,
    CursorProvider,
    MockProvider,
    OmpCliProvider,
    OmpProvider,
    OpenAICompatibleProvider,
    ProviderRegistry,
    compute_provider_leaderboard,
    registry,
)
from showdown.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_provider_protocol_and_aliases():
    """Verify class hierarchy, protocol conformance, and backward-compatible aliases."""
    assert issubclass(ClaudeCliProvider, AgentProvider)
    assert issubclass(CodexCliProvider, AgentProvider)
    assert issubclass(CursorProvider, AgentProvider)
    assert issubclass(OmpProvider, AgentProvider)
    assert issubclass(OpenAICompatibleProvider, AgentProvider)

    # Aliases
    assert ClaudeProvider is ClaudeCliProvider
    assert CodexProvider is CodexCliProvider
    assert CursorCliProvider is CursorProvider
    assert OmpCliProvider is OmpProvider

    # Mock provider generates valid content and supports both generate() and call()
    mock_p = MockProvider()
    assert mock_p.is_available() is True
    out_gen = mock_p.generate("Give me candidate ideas")
    out_call = mock_p.call("Give me candidate ideas")
    assert json.loads(out_gen)
    assert out_gen == out_call

    # Judge prompt handling
    judge_out = mock_p.generate("Judge the winner between candidate A and candidate B")
    parsed_judge = json.loads(judge_out)
    assert parsed_judge.get("winner") == "a"


def test_cli_providers_availability_and_invocation():
    """Test availability checks and subprocess dispatch for CLI providers."""
    with patch("shutil.which", return_value=None):
        claude = ClaudeCliProvider()
        codex = CodexCliProvider()
        cursor = CursorProvider()
        omp = OmpProvider()

        assert claude.is_available() is False
        assert codex.is_available() is False
        assert cursor.is_available() is False
        assert omp.is_available() is False

        with pytest.raises(RuntimeError, match="Claude CLI"):
            claude.generate("test")
        with pytest.raises(RuntimeError, match="Codex CLI"):
            codex.generate("test")
        with pytest.raises(RuntimeError, match="Cursor Agent CLI"):
            cursor.generate("test")
        with pytest.raises(RuntimeError, match="OMP CLI"):
            omp.generate("test")

    # When binaries exist
    with patch("shutil.which", return_value="/usr/local/bin/dummy"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='[{"label": "Test", "content": "Ok"}]')

            claude = ClaudeCliProvider()
            assert claude.is_available() is True
            res = claude.generate("hello claude")
            assert res == '[{"label": "Test", "content": "Ok"}]'
            assert mock_run.call_args[0][0][:2] == ["claude", "-p"]

            cursor = CursorProvider()
            assert cursor.is_available() is True
            cursor.generate("hello cursor")
            assert mock_run.call_args[0][0][:4] == ["cursor-agent", "-p", "--output-format", "text"]

            omp = OmpProvider()
            assert omp.is_available() is True
            omp.generate("hello omp")
            assert mock_run.call_args[0][0][:3] == ["omp", "-p", "--model=homelab-default"]


def test_openai_compatible_provider():
    """Test OpenAICompatibleProvider URL, auth headers, and execution."""
    prov = OpenAICompatibleProvider(
        provider_id="custom-vllm",
        display_name="Custom vLLM",
        base_url="http://vllm.homelab:8000/v1",
        api_key="secret-token",
        model="qwen-2.5-coder",
    )
    assert prov.id == "custom-vllm"
    assert prov.base_url == "http://vllm.homelab:8000/v1"
    assert prov.is_available() is True

    fake_response_data = json.dumps({
        "choices": [{"message": {"content": '[{"label": "API Cand", "content": "Generated via vLLM"}]'}}]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response_data
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        out = prov.generate("generate something")
        assert "Generated via vLLM" in out


def test_registry_resolution_and_aliases():
    """Test registry lookup, aliases, priority resolution, and error handling."""
    reg = ProviderRegistry()

    # Aliases
    assert reg.get("homelab").id == "omp"
    assert reg.get("homelab-default").id == "omp"
    assert reg.get("claude-code").id == "claude"
    assert reg.get("cursor-agent").id == "cursor"

    # Requested unknown provider raises ValueError
    with pytest.raises(ValueError, match="Unknown provider 'non-existent'"):
        reg.resolve_provider("non-existent")

    # Prefer homelab environment variable
    with patch.dict(os.environ, {"SHOWDOWN_PREFER_HOMELAB": "1"}):
        with patch.object(reg.get("omp"), "is_available", return_value=True):
            resolved = reg.resolve_provider("auto")
            assert resolved.id == "omp"


def test_multi_agent_fan_out_evolution():
    """Test multi-agent evolution fanning out across multiple providers in parallel."""
    from showdown.evolve import execute_evolution

    t = Tournament(
        id="t_fanout",
        title="Taglines",
        prompt="Design brand taglines",
        task_type=TaskType.TEXT,
        candidates=[Candidate(id="c1", label="Base", content="Base Content")],
    )

    claude_response = json.dumps([
        {"label": "Claude 1", "content": "Content Claude 1", "differs_by": "Claude style"},
        {"label": "Claude 2", "content": "Content Claude 2", "differs_by": "Claude focus"},
    ])
    omp_response = json.dumps([
        {"label": "OMP 1", "content": "Content OMP 1", "differs_by": "Homelab speed"},
    ])

    def mock_backend(prompt, backend):
        if backend == "claude":
            return claude_response
        elif backend == "omp":
            return omp_response
        return "[]"

    with patch.object(registry.get("claude"), "is_available", return_value=True):
        with patch.object(registry.get("omp"), "is_available", return_value=True):
            with patch("showdown.evolve.call_generation_backend", side_effect=mock_backend):
                res = execute_evolution(
                    tournament=t,
                    count=3,
                    providers=["claude", "omp"],
                    mode="refine",
                )

                assert len(res.new_candidates) == 3
                assert "claude" in res.providers_used
                assert "omp" in res.providers_used
                assert "Multi-agent fan-out" in res.summary

                # Verify metadata tags on generated candidates
                claude_cands = [c for c in res.new_candidates if c.metadata.get("provider") == "claude"]
                omp_cands = [c for c in res.new_candidates if c.metadata.get("provider") == "omp"]
                assert len(claude_cands) == 2
                assert len(omp_cands) == 1
                assert claude_cands[0].metadata["provider_name"] == "Anthropic Claude (CLI)"
                assert omp_cands[0].metadata["provider_name"] == "OMP Homelab (CLI)"


def test_api_providers_endpoints(client: TestClient):
    """Test GET /api/providers and GET /api/providers/leaderboard."""
    res = client.get("/api/providers")
    assert res.status_code == 200
    providers = res.json()
    assert isinstance(providers, list)
    p_ids = [p["id"] for p in providers]
    assert "claude" in p_ids
    assert "codex" in p_ids
    assert "cursor" in p_ids
    assert "omp" in p_ids

    # Leaderboard endpoint returns 200
    res_lb = client.get("/api/providers/leaderboard")
    assert res_lb.status_code == 200
    assert isinstance(res_lb.json(), list)


def test_cross_model_leaderboard_computation():
    """Verify Elo ratings, win rates, and accepted winners are computed across models."""
    cand_claude = Candidate(
        id="c_claude",
        label="Claude Idea",
        content="...",
        metadata={"provider": "claude"},
    )
    cand_codex = Candidate(
        id="c_codex",
        label="Codex Idea",
        content="...",
        metadata={"provider": "codex"},
    )
    cand_intra_claude = Candidate(
        id="c_claude2",
        label="Claude Alt",
        content="...",
        metadata={"provider": "claude"},
    )

    t = Tournament(
        id="t_lb_test",
        title="Leaderboard Test",
        task_type=TaskType.TEXT,
        candidates=[cand_claude, cand_codex, cand_intra_claude],
        accepted_candidate_id="c_claude",
        matches=[
            # Claude beats Codex
            Match(id_a="c_claude", id_b="c_codex", winner="a"),
            # Codex beats Claude in round 2
            Match(id_a="c_claude", id_b="c_codex", winner="b"),
            # Claude beats Codex in round 3
            Match(id_a="c_claude", id_b="c_codex", winner="a"),
            # Intra-provider comparison (Claude vs Claude2) - should NOT affect provider Elo
            Match(id_a="c_claude", id_b="c_intra_claude", winner="a"),
        ],
    )

    leaderboard = compute_provider_leaderboard([t])
    assert len(leaderboard) >= 2

    claude_stats = next(item for item in leaderboard if item["provider_id"] == "claude")
    codex_stats = next(item for item in leaderboard if item["provider_id"] == "codex")

    assert claude_stats["matches"] == 3
    assert claude_stats["wins"] == 2
    assert claude_stats["losses"] == 1
    assert claude_stats["win_rate"] == 66.7
    assert claude_stats["accepted_winners"] == 1
    assert claude_stats["elo"] > 1200.0

    assert codex_stats["matches"] == 3
    assert codex_stats["wins"] == 1
    assert codex_stats["losses"] == 2
    assert codex_stats["win_rate"] == 33.3
    assert codex_stats["elo"] < 1200.0


def test_mcp_provider_tools():
    """Verify MCP tools for listing providers and leaderboard."""
    from showdown.mcp_server import (
        showdown_get_provider_leaderboard,
        showdown_list_providers,
    )

    providers_res = showdown_list_providers()
    assert "providers" in providers_res
    assert providers_res["total_count"] >= 4

    lb_res = showdown_get_provider_leaderboard()
    assert "leaderboard" in lb_res
    assert isinstance(lb_res["leaderboard"], list)
