"""Python SDK client for interacting with Showdown programmatically."""

import time
from typing import Any, Dict, List, Optional, Union
from showdown.models import (
    AcceptCandidateRequest,
    Candidate,
    CreateTournamentRequest,
    TaskType,
    Tournament,
    TournamentStatus,
)
from showdown.storage import Storage


def create_tournament(
    title: str,
    candidates: Optional[List[Dict[str, Any]]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    task_type: Union[str, TaskType] = "text",
    tournament_id: Optional[str] = None,
    parent_ids: Optional[List[str]] = None,
    context: Optional[str] = None,
    chain_mode: Optional[str] = "growth",
    blinded: bool = False,
    data_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Tournament:
    """
    Create a new Showdown tournament programmatically.

    Args:
        title: Display title for the tournament.
        candidates: List of dicts with 'id', 'content', and optional 'label'/'metadata'.
        items: Alias for 'candidates' to support Universal Skills standard.
        prompt: The evaluation prompt or goal given to the LLMs/generators.
        task_type: 'text', 'markdown', 'code', 'diff', 'image', 'svg', or 'json'.
        tournament_id: Optional custom slug/id.
        parent_ids: Optional parent tournament IDs to chain from.
        context: Optional context text from upstream tournaments.
        chain_mode: Chaining strategy: 'growth' (continuity) or 'divergence' (contrast).
        blinded: If True, candidate metadata is obscured during evaluation.
        data_dir: Optional custom data storage directory.
        overwrite: If True, overwrite existing tournament with the same ID.

    Returns:
        The created Tournament instance.
    """
    storage = Storage(data_dir=data_dir)

    raw_candidates = candidates if candidates is not None else (items or [])

    parsed_candidates = [
        Candidate(
            id=str(c.get("id", f"cand_{idx}")),
            label=c.get("label"),
            content=str(c.get("content", "")),
            metadata=c.get("metadata", {}),
        )
        for idx, c in enumerate(raw_candidates)
    ]

    resolved_task_type = task_type if isinstance(task_type, TaskType) else TaskType(str(task_type).lower())

    req = CreateTournamentRequest(
        id=tournament_id,
        title=title,
        prompt=prompt,
        task_type=resolved_task_type,
        candidates=parsed_candidates,
        parent_ids=parent_ids or [],
        context=context,
        chain_mode=chain_mode,
        blinded=blinded,
        overwrite=overwrite,
    )

    from showdown.server import create_tournament as server_create_tournament
    import showdown.server
    original_storage = showdown.server.storage
    try:
        if data_dir:
            showdown.server.storage = storage
        return server_create_tournament(req)
    finally:
        showdown.server.storage = original_storage


def accept_candidate(
    tournament_id: str,
    candidate_id: str,
    notes: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Accept a candidate as the winning selection of a tournament."""
    from showdown.server import accept_candidate as server_accept_candidate
    import showdown.server
    storage = Storage(data_dir=data_dir)
    original_storage = showdown.server.storage
    try:
        if data_dir:
            showdown.server.storage = storage
        return server_accept_candidate(
            tournament_id=tournament_id,
            req=AcceptCandidateRequest(candidate_id=candidate_id, notes=notes),
        )
    finally:
        showdown.server.storage = original_storage


def wait_for_tournament(
    tournament_id: str,
    timeout: int = 30,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Block until an accepted candidate is chosen or return the current leader on timeout."""
    from showdown.server import wait_for_completion
    import showdown.server
    storage = Storage(data_dir=data_dir)
    original_storage = showdown.server.storage
    try:
        if data_dir:
            showdown.server.storage = storage
        return wait_for_completion(tournament_id=tournament_id, timeout=timeout)
    finally:
        showdown.server.storage = original_storage


def undo_vote(
    tournament_id: str,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Rollback the last recorded vote in a tournament and recalibrate Elo ratings."""
    from showdown.server import undo_vote as server_undo_vote
    import showdown.server
    storage = Storage(data_dir=data_dir)
    original_storage = showdown.server.storage
    try:
        if data_dir:
            showdown.server.storage = storage
        return server_undo_vote(tournament_id=tournament_id)
    finally:
        showdown.server.storage = original_storage


def run_judge(
    tournament_id: str,
    rounds: int = 5,
    backend: Optional[str] = "auto",
    rubric: Optional[str] = None,
    swap_positions: bool = True,
    voter: Optional[str] = None,
    mode: str = "active",
    stop_on_convergence: bool = False,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run automated LLM-as-a-judge tournament rounds."""
    from showdown.models import JudgeRequest
    from showdown.server import judge_tournament_endpoint
    import showdown.server

    req = JudgeRequest(
        rounds=rounds,
        backend=backend,
        rubric=rubric,
        swap_positions=swap_positions,
        voter=voter,
        mode=mode,
        stop_on_convergence=stop_on_convergence,
    )

    if data_dir:
        storage = Storage(data_dir=data_dir)
        original_storage = showdown.server.storage
        showdown.server.storage = storage
        try:
            res = judge_tournament_endpoint(tournament_id=tournament_id, req=req)
            return res.model_dump()
        finally:
            showdown.server.storage = original_storage

    res = judge_tournament_endpoint(tournament_id=tournament_id, req=req)
    return res.model_dump()


def evolve_tournament(
    tournament_id: str,
    count: int = 5,
    instructions: Optional[str] = None,
    backend: Optional[str] = "auto",
    providers: Optional[List[str]] = None,
    mode: str = "refine",
    wildcards: Optional[int] = None,
    chain_mode: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize new candidate variations with refine, diverge, hybrid, or multi-agent fan-out."""
    from showdown.models import EvolveRequest
    from showdown.server import evolve_tournament_endpoint
    import showdown.server

    req = EvolveRequest(
        count=count,
        instructions=instructions,
        backend=backend,
        providers=providers,
        mode=mode,
        wildcards=wildcards,
        chain_mode=chain_mode,
    )

    if data_dir:
        storage = Storage(data_dir=data_dir)
        original_storage = showdown.server.storage
        showdown.server.storage = storage
        try:
            res = evolve_tournament_endpoint(tournament_id=tournament_id, req=req)
            return res.model_dump()
        finally:
            showdown.server.storage = original_storage

    res = evolve_tournament_endpoint(tournament_id=tournament_id, req=req)
    return res.model_dump()


def list_providers() -> List[Dict[str, Any]]:
    """List registered agent providers and host detection status."""
    from showdown.providers import registry
    return [p.to_dict() for p in registry.list_all()]


def get_provider_leaderboard(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Aggregate model/provider leaderboard across tournaments."""
    from showdown.providers import compute_provider_leaderboard
    storage = Storage(data_dir=data_dir)
    return compute_provider_leaderboard(storage.list_tournaments())

