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
    blinded: bool = False,
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
        blinded: If True, mask candidate metadata to eliminate evaluation bias.
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
        blinded=blinded,
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
        "prompt_used": res.prompt_used,
        "summary": res.summary,
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


@server.tool()
def showdown_undo_vote(
    tournament_id: str,
) -> Dict[str, Any]:
    """
    Undo the last vote recorded in a tournament and recalibrate Elo ratings.

    Args:
        tournament_id: ID of the tournament.

    Returns:
        Status and remaining match count.
    """
    from showdown.server import undo_vote as server_undo_vote
    return server_undo_vote(tournament_id=tournament_id)


@server.tool()
def showdown_get_consistency(
    tournament_id: str,
    voter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate voter self-consistency metrics across repeated matchups.

    Args:
        tournament_id: ID of the tournament.
        voter: Optional judge name to filter consistency calculation.

    Returns:
        Consistency rate, repeated pair counts, and self-reversals.
    """
    from showdown.server import get_voter_consistency
    return get_voter_consistency(tournament_id=tournament_id, voter=voter)


@server.tool()
def showdown_run_judge(
    tournament_id: str,
    rounds: int = 5,
    backend: str = "auto",
    rubric: Optional[str] = None,
    swap_positions: bool = True,
    voter: Optional[str] = None,
    mode: str = "active",
    stop_on_convergence: bool = False,
) -> Dict[str, Any]:
    """
    Run automated LLM-as-a-judge comparison rounds on a tournament.

    Args:
        tournament_id: ID of the tournament to judge.
        rounds: Number of pairwise comparisons to evaluate (default 5).
        backend: LLM backend ('auto', 'omp', 'claude', 'codex', 'openai').
        rubric: Optional custom evaluation criteria or guidelines.
        swap_positions: Whether to evaluate both A vs B and B vs A to mitigate position bias (default True).
        voter: Custom evaluator identifier tag (defaults to judge:<backend>).
        mode: Matchup selection strategy ('active', 'info_gain', 'controversial', 'close').
        stop_on_convergence: If True, halts evaluation early once the top candidate statistically separates.

    Returns:
        Summary of matches evaluated, consistency, convergence status, and results.
    """
    from showdown.client import run_judge
    return run_judge(
        tournament_id=tournament_id,
        rounds=rounds,
        backend=backend,
        rubric=rubric,
        swap_positions=swap_positions,
        voter=voter,
        mode=mode,
        stop_on_convergence=stop_on_convergence,
    )


@server.tool(name="showdown_export_dataset", description="Export aggregated preference datasets across tournaments for DPO, KTO, or reward model fine-tuning with train/val splitting.")
def mcp_export_dataset(
    tournament_id: Optional[str] = None,
    format: str = "dpo",
    task_type: Optional[str] = None,
    voter: Optional[str] = None,
    consensus: Optional[str] = None,
    min_agreement: Optional[float] = None,
    dedup: bool = True,
    include_critique: bool = True,
    split: Optional[float] = None,
    split_by: str = "lineage",
) -> Dict[str, Any]:
    """
    Export preference pairs across tournaments or from a single tournament.

    Args:
        tournament_id: Optional tournament ID. If omitted, aggregates across all tournaments.
        format: Export format ('dpo', 'kto', 'pairwise_margins').
        task_type: Filter by task type ('code', 'text', 'svg', 'markdown').
        voter: Filter matches by evaluator tag.
        consensus: Multi-annotator consensus requirement ('strict' or 'majority').
        min_agreement: Minimum inter-annotator agreement threshold (0.5 to 1.0).
        dedup: Whether to deduplicate identical prompt/chosen/rejected pairs.
        include_critique: Whether to include evaluator notes/critiques.
        split: Optional train/val ratio (e.g. 0.8 for 80% train / 20% val).
        split_by: Split partitioning strategy ('lineage' or 'random').

    Returns:
        Aggregated dataset records or train/val dictionary with split stats.
    """
    from showdown.export import export_dataset
    if tournament_id:
        t = storage.load_tournament(tournament_id)
        if not t:
            return {"error": f"Tournament '{tournament_id}' not found"}
        tournaments = [t]
    else:
        tournaments = storage.list_tournaments()

    try:
        result = export_dataset(
            tournaments=tournaments,
            format=format,
            task_type=task_type,
            voter=voter,
            consensus=consensus,
            min_agreement=min_agreement,
            dedup=dedup,
            include_critique=include_critique,
            split=split,
            split_by=split_by,
        )
        if isinstance(result, list):
            return {"record_count": len(result), "records": result}
        return result
    except ValueError as e:
        return {"error": str(e)}


@server.tool()
def showdown_get_trajectory(
    tournament_id: str,
    candidate_id: str,
) -> Dict[str, Any]:
    """
    Retrieve structured trajectory steps and execution summary for an agent candidate.

    Args:
        tournament_id: Tournament ID.
        candidate_id: Candidate ID.

    Returns:
        Dictionary containing candidate ID, steps, and execution summary.
    """
    from showdown.server import get_candidate_trajectory
    try:
        return get_candidate_trajectory(tournament_id, candidate_id)
    except Exception as e:
        return {"error": str(e)}


@server.tool()
def showdown_annotate_step(
    tournament_id: str,
    candidate_id: str,
    step_index: int,
    tag: str,
    notes: Optional[str] = None,
    voter: Optional[str] = "human",
) -> Dict[str, Any]:
    """
    Add or update step-level commentary / tag for Process Reward Models (PRM) & Step-level Preference Optimization (SPO).

    Args:
        tournament_id: Tournament ID.
        candidate_id: Candidate ID.
        step_index: Integer index of the trajectory step.
        tag: 'exemplary', 'inefficient', 'incorrect', or 'neutral'.
        notes: Optional explanation or critique of this specific tool invocation.
        voter: Identifier of the reviewer.

    Returns:
        Updated annotation details and summary.
    """
    from showdown.models import StepAnnotationRequest, StepAnnotationTag
    from showdown.server import annotate_candidate_step
    try:
        tag_enum = StepAnnotationTag(tag.lower())
    except ValueError:
        return {"error": f"Invalid tag '{tag}'. Must be 'exemplary', 'inefficient', 'incorrect', or 'neutral'."}

    try:
        req = StepAnnotationRequest(tag=tag_enum, notes=notes, voter=voter)
        return annotate_candidate_step(tournament_id, candidate_id, step_index, req)
    except Exception as e:
        return {"error": str(e)}


def run_mcp_server(transport: str = "stdio", data_dir: Optional[str] = None) -> None:
    """Run the Showdown MCP server."""
    if data_dir:
        storage.data_dir = Storage(data_dir=data_dir).data_dir
    server.run(transport=transport)


if __name__ == "__main__":
    run_mcp_server()
