"""Model Context Protocol (MCP) server for Showdown."""

import json
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from showdown.client import accept_candidate, create_tournament, wait_for_tournament
from showdown.models import Candidate, CandidateStats, TaskType, VoteRequest
from showdown.server import (
    evolve_tournament_endpoint,
    export_tournament,
    get_matchup,
    get_tournament,
    get_tournament_chain,
    record_vote,
    storage,
)
from showdown.storage import Storage

server = MCPServer("showdown", version="0.1.0")


@server.tool()
def showdown_create_tournament(
    title: str,
    candidates: List[Dict[str, Any]],
    prompt: Optional[str] = None,
    task_type: str = "text",
    tournament_id: Optional[str] = None,
    parent_ids: Optional[List[str]] = None,
    context: Optional[str] = None,
    port: int = 8000,
) -> Dict[str, Any]:
    """
    Create a new Showdown ranking tournament with candidate outputs.

    Args:
        title: Short title describing the comparison arena.
        candidates: List of candidate objects, each having 'id', 'content', and optional 'label'.
        prompt: The evaluation prompt or goal provided to generators.
        task_type: Type of output: 'text', 'markdown', 'code', 'diff', 'image', 'svg', or 'json'.
        tournament_id: Optional custom slug or ID.
        parent_ids: Optional list of parent tournament IDs to chain sequential stages.
        context: Optional upstream context text from parent winners.
        port: Web server port for browser interaction URL (default 8000).

    Returns:
        Metadata including tournament ID, candidate count, and browser URL.
    """
    t = create_tournament(
        title=title,
        candidates=candidates,
        prompt=prompt,
        task_type=task_type,
        tournament_id=tournament_id,
        parent_ids=parent_ids,
        context=context,
    )
    return {
        "tournament_id": t.id,
        "title": t.title,
        "task_type": t.task_type.value,
        "candidate_count": len(t.candidates),
        "url": f"http://localhost:{port}/?t={t.id}",
        "message": f"Tournament '{t.title}' created. Human evaluator can rank at http://localhost:{port}/?t={t.id}",
    }


@server.tool()
def showdown_wait_for_decision(
    tournament_id: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Block until a human evaluator accepts a winning candidate in the Showdown arena.

    Args:
        tournament_id: ID of the tournament to wait on.
        timeout: Maximum seconds to block waiting for human decision (default 120).

    Returns:
        Status ('completed' or 'timeout'), accepted candidate details, and user notes.
    """
    return wait_for_tournament(tournament_id=tournament_id, timeout=timeout)


@server.tool()
def showdown_get_leaderboard(
    tournament_id: str,
    voter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve candidate rankings, Elo ratings, win/loss records, and convergence status.

    Args:
        tournament_id: ID of the tournament.
        voter: Optional evaluator name to retrieve sliced Elo ratings for that specific judge.

    Returns:
        Ranked candidates list, active voters, and convergence status.
    """
    t = get_tournament(tournament_id=tournament_id, voter=voter)
    ranked = sorted(
        [
            {
                "id": c.id,
                "label": c.label or c.id,
                "elo": round(t.stats.get(c.id, CandidateStats()).elo, 1),
                "record": f"{t.stats.get(c.id, CandidateStats()).wins}W-{t.stats.get(c.id, CandidateStats()).losses}L-{t.stats.get(c.id, CandidateStats()).ties}T",
                "matches": t.stats.get(c.id, CandidateStats()).matches,
                "is_accepted_winner": t.accepted_candidate_id == c.id,
                "content_preview": c.content[:160] + "..." if len(c.content) > 160 else c.content,
            }
            for c in t.candidates
        ],
        key=lambda x: x["elo"],
        reverse=True,
    )
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1

    matchup_info = get_matchup(tournament_id) if len(t.candidates) >= 2 else {}

    return {
        "tournament_id": t.id,
        "title": t.title,
        "voter_filter": voter or "pooled",
        "available_voters": t.voters,
        "accepted_candidate_id": t.accepted_candidate_id,
        "converged": matchup_info.get("converged", False),
        "confidence": matchup_info.get("confidence", 0.0),
        "leaderboard": ranked,
    }


@server.tool()
def showdown_record_vote(
    tournament_id: str,
    id_a: str,
    id_b: str,
    winner: str,
    voter: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a pairwise head-to-head vote between two candidates.

    Args:
        tournament_id: ID of the tournament.
        id_a: ID of candidate A.
        id_b: ID of candidate B.
        winner: One of 'a', 'b', 'tie', or 'both_bad'.
        voter: Judge or evaluator tag (e.g. 'claude-code', 'human', 'omp:homelab-default').
        notes: Optional critique note explaining the decision.

    Returns:
        Updated Elo ratings for both candidates.
    """
    res = record_vote(
        tournament_id=tournament_id,
        vote=VoteRequest(
            id_a=id_a,
            id_b=id_b,
            winner=winner,
            voter=voter,
            notes=notes,
        ),
    )
    return res


@server.tool()
def showdown_evolve_candidates(
    tournament_id: str,
    count: int = 5,
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesize Generation N+1 candidate outputs based on human preference convergence.

    Args:
        tournament_id: ID of the tournament.
        count: Number of mutated candidates to generate (default 5).
        instructions: Optional additional creative direction for mutation.

    Returns:
        List of newly generated candidates injected into the tournament pool.
    """
    from showdown.models import EvolveRequest
    res = evolve_tournament_endpoint(
        tournament_id=tournament_id,
        req=EvolveRequest(count=count, instructions=instructions),
    )
    return {
        "tournament_id": tournament_id,
        "new_candidates": [c.model_dump() for c in res.new_candidates],
        "top_performers": res.top_performers,
        "rejected_performers": res.rejected_performers,
    }


@server.tool()
def showdown_get_chain(
    tournament_id: str,
) -> Dict[str, Any]:
    """
    Traverse parent tournament lineage and assemble multi-stage outputs into a single document.

    Args:
        tournament_id: ID of the downstream tournament stage.

    Returns:
        Chronological list of stages with accepted winners, context, and assembled full text.
    """
    return get_tournament_chain(tournament_id=tournament_id)


@server.tool()
def showdown_get_agreement(
    tournament_id: str,
) -> Dict[str, Any]:
    """
    Calculate and retrieve inter-annotator agreement metrics across judges in a tournament.

    Args:
        tournament_id: ID of the tournament.

    Returns:
        Agreement rates, shared pairs counts, reversals, and pairwise evaluator comparisons.
    """
    from showdown.server import get_inter_annotator_agreement
    return get_inter_annotator_agreement(tournament_id=tournament_id)


@server.tool()
def showdown_export_dataset(
    tournament_id: str,
    format: str = "dpo",
    voter: Optional[str] = None,
    consensus: Optional[str] = None,
    min_agreement: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Export tournament match judgments into standardized preference dataset formats.

    Args:
        tournament_id: ID of the tournament.
        format: Dataset format: 'dpo' (Direct Preference Optimization), 'kto' (Kahneman-Tversky Optimization), or 'leaderboard'.
        voter: Optional evaluator filter to export only judgments from that specific judge.
        consensus: Optional consensus mode ('strict' for unanimous with >=2 judges, or 'majority').
        min_agreement: Optional minimum agreement threshold (0.5 to 1.0).

    Returns:
        List of preference records with agreement provenance metadata.
    """
    records = export_tournament(
        tournament_id=tournament_id,
        format=format,
        voter=voter,
        consensus=consensus,
        min_agreement=min_agreement,
    )
    return {
        "tournament_id": tournament_id,
        "format": format,
        "count": len(records),
        "records": records,
    }


def run_mcp_server(transport: str = "stdio", data_dir: Optional[str] = None) -> None:
    """Run the Showdown MCP server."""
    if data_dir:
        storage.data_dir = Storage(data_dir=data_dir).data_dir
    server.run(transport=transport)


if __name__ == "__main__":
    run_mcp_server()
