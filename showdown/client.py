"""Python SDK client for interacting with Showdown programmatically."""

import time
from typing import Any, Dict, List, Optional
from showdown.models import Candidate, CreateTournamentRequest, TaskType, Tournament
from showdown.storage import Storage


def create_tournament(
    title: str,
    candidates: List[Dict[str, Any]],
    prompt: Optional[str] = None,
    task_type: str = "text",
    tournament_id: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tournament:
    """
    Create a new Showdown tournament programmatically.

    Args:
        title: Display title for the tournament.
        candidates: List of dicts with 'id', 'content', and optional 'label'/'metadata'.
        prompt: The evaluation prompt or goal given to the LLMs/generators.
        task_type: 'text', 'markdown', 'code', 'image', or 'json'.
        tournament_id: Optional custom slug/id.
        data_dir: Optional custom data storage directory.

    Returns:
        The created Tournament instance.
    """
    storage = Storage(data_dir=data_dir)
    t_id = tournament_id or f"tournament_{int(time.time())}"

    parsed_candidates = [
        Candidate(
            id=str(c.get("id", f"cand_{idx}")),
            label=c.get("label"),
            content=str(c.get("content", "")),
            metadata=c.get("metadata", {}),
        )
        for idx, c in enumerate(candidates)
    ]

    req = CreateTournamentRequest(
        id=t_id,
        title=title,
        prompt=prompt,
        task_type=TaskType(task_type.lower()),
        candidates=parsed_candidates,
    )

    from showdown.server import create_tournament as server_create_tournament

    return server_create_tournament(req)
