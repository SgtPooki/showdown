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
        task_type: 'text', 'markdown', 'code', 'diff', 'image', or 'json'.
        tournament_id: Optional custom slug/id.
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
