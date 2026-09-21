"""Realtime Server-Sent Events (SSE) streaming hub and evolution progress pipeline."""

import asyncio
from collections import defaultdict
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from fastapi.responses import StreamingResponse

from showdown.models import Candidate, EvolveRequest, EvolveResponse, Tournament
from showdown.storage import Storage


def format_sse(event: str, data: Any) -> str:
    """Format an event and payload into SSE protocol string."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


class TournamentEventHub:
    """Pub/sub hub distributing live tournament match and Elo updates to SSE subscribers."""

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, tournament_id: str) -> asyncio.Queue:
        """Register a new SSE client subscriber queue for a specific tournament."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers[tournament_id].add(q)
        return q

    def unsubscribe(self, tournament_id: str, queue: asyncio.Queue) -> None:
        """Remove an SSE client subscriber queue."""
        if tournament_id in self._subscribers:
            self._subscribers[tournament_id].discard(queue)
            if not self._subscribers[tournament_id]:
                del self._subscribers[tournament_id]

    def publish(self, tournament_id: str, event_type: str, data: Any) -> None:
        """Broadcast an event to all subscribers listening to this tournament."""
        if tournament_id not in self._subscribers:
            return

        message = format_sse(event_type, data)
        dead_queues = set()
        for q in self._subscribers[tournament_id]:
            try:
                q.put_nowait(message)
            except (asyncio.QueueFull, Exception):
                dead_queues.add(q)

        for q in dead_queues:
            self._subscribers[tournament_id].discard(q)


# Global default hub instance
event_hub = TournamentEventHub()


async def stream_tournament_events(
    tournament_id: str,
    timeout: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """Yield live SSE stream messages for a tournament until client disconnects or timeout expires."""
    queue = event_hub.subscribe(tournament_id)
    try:
        # Initial ping handshake
        yield format_sse("connected", {"tournament_id": tournament_id, "status": "listening"})
        while True:
            try:
                poll_interval = timeout if timeout is not None else 15.0
                msg = await asyncio.wait_for(queue.get(), timeout=poll_interval)
                yield msg
            except asyncio.TimeoutError:
                if timeout is not None:
                    break
                yield format_sse("ping", {"status": "alive"})
            except asyncio.CancelledError:
                break
    finally:
        event_hub.unsubscribe(tournament_id, queue)


async def stream_evolution_progress(
    tournament: Tournament,
    req: EvolveRequest,
    storage: Storage,
) -> AsyncGenerator[str, None]:
    """
    Run candidate evolution and yield SSE lifecycle progress events:
    - synthesizing_preferences
    - provider_dispatched
    - candidate_ready (streamed per item)
    - complete
    """
    from showdown.evolve import (
        build_evolution_prompt,
        call_generation_backend,
        _parse_candidates_json,
        resolve_upstream_context,
    )
    from showdown.providers import registry

    # Step 1: Synthesizing preferences
    yield format_sse(
        "synthesizing_preferences",
        {
            "tournament_id": tournament.id,
            "mode": req.mode or "refine",
            "count": req.count,
            "instructions": req.instructions,
        },
    )
    await asyncio.sleep(0.01)

    upstream_context = resolve_upstream_context(tournament, storage)
    existing_gens = [c.generation for c in tournament.candidates if hasattr(c, "generation")]
    next_gen = (max(existing_gens) + 1) if existing_gens else 2
    resolved_mode = (req.mode or "refine").lower()

    # Step 2: Provider Resolution and Dispatch
    if req.providers and len(req.providers) > 1:
        resolved_providers = []
        for pid in req.providers:
            p = registry.get(pid)
            if p and p.is_available():
                resolved_providers.append(p)
            elif p:
                yield format_sse("error", {"detail": f"Provider '{pid}' is not available on host."})
                return
            else:
                yield format_sse("error", {"detail": f"Unknown provider '{pid}'."})
                return

        if not resolved_providers:
            yield format_sse("error", {"detail": "No available providers."})
            return

        n_prov = len(resolved_providers)
        counts_per_prov = [req.count // n_prov + (1 if i < (req.count % n_prov) else 0) for i in range(n_prov)]
        total_wildcards = (
            min(max(1, req.wildcards), req.count)
            if req.wildcards is not None
            else (max(1, req.count // 3) if resolved_mode == "hybrid" else 0)
        )
        wildcards_remaining = total_wildcards if resolved_mode == "hybrid" else 0

        tasks_info = []
        provider_work = []
        for i, prov in enumerate(resolved_providers):
            p_total = counts_per_prov[i]
            if p_total == 0:
                continue
            if resolved_mode == "diverge":
                r_alloc, w_alloc = 0, p_total
            elif resolved_mode == "refine":
                r_alloc, w_alloc = p_total, 0
            else:  # hybrid
                w_alloc = min(p_total, wildcards_remaining)
                r_alloc = p_total - w_alloc
                wildcards_remaining = max(0, wildcards_remaining - w_alloc)

            tasks_info.append({"provider": prov.id, "display_name": prov.display_name, "refine": r_alloc, "wildcards": w_alloc})
            provider_work.append((prov, r_alloc, w_alloc))

        yield format_sse(
            "provider_dispatched",
            {
                "providers": [p.id for p in resolved_providers],
                "fan_out": True,
                "tasks": tasks_info,
            },
        )
        await asyncio.sleep(0.01)

        # Worker loop streaming results as each provider finishes
        combined_cands: List[Candidate] = []
        all_prompts: List[str] = []

        def _execute_worker(prov, r_cnt, w_cnt):
            prov_cands = []
            prov_prompts = []
            if r_cnt > 0:
                r_prompt, _ = build_evolution_prompt(
                    tournament=tournament,
                    count=r_cnt,
                    instructions=req.instructions,
                    mode="refine",
                    chain_mode=req.chain_mode,
                    upstream_context=upstream_context,
                )
                r_raw = call_generation_backend(r_prompt, prov.id)
                r_parsed = _parse_candidates_json(
                    r_raw,
                    next_gen,
                    backend_name=prov.id,
                    is_divergence=False,
                    is_wildcard=False,
                    provider=prov.id,
                    model=prov.model,
                )
                prov_cands.extend(r_parsed)
                prov_prompts.append(r_prompt)

            if w_cnt > 0:
                w_prompt, _ = build_evolution_prompt(
                    tournament=tournament,
                    count=w_cnt,
                    instructions=req.instructions,
                    mode="diverge",
                    chain_mode=req.chain_mode,
                    upstream_context=upstream_context,
                )
                w_raw = call_generation_backend(w_prompt, prov.id)
                w_parsed = _parse_candidates_json(
                    w_raw,
                    next_gen,
                    backend_name=prov.id,
                    is_divergence=True,
                    is_wildcard=True,
                    provider=prov.id,
                    model=prov.model,
                )
                prov_cands.extend(w_parsed)
                prov_prompts.append(w_prompt)

            return prov, prov_cands, prov_prompts

        loop = asyncio.get_running_loop()
        # Execute in thread pool to prevent blocking async event loop
        tasks = [
            loop.run_in_executor(None, _execute_worker, prov, r_cnt, w_cnt)
            for prov, r_cnt, w_cnt in provider_work
        ]

        for completed_coro in asyncio.as_completed(tasks):
            prov, cands, p_list = await completed_coro
            all_prompts.extend(p_list)
            for c in cands:
                combined_cands.append(c)
                yield format_sse("candidate_ready", c.model_dump())
                await asyncio.sleep(0.01)

        refine_actual = sum(1 for c in combined_cands if not c.metadata.get("wildcard"))
        wildcard_actual = sum(1 for c in combined_cands if c.metadata.get("wildcard"))
        prov_ids = [p.id for p in resolved_providers]
        summary = (
            f"Multi-agent fan-out across {len(prov_ids)} providers ({', '.join(prov_ids)}): "
            f"generated {len(combined_cands)} candidates ({refine_actual} refined, {wildcard_actual} wildcards)."
        )

        response = EvolveResponse(
            prompt_used="\n\n--- [FAN-OUT MULTI-AGENT] ---\n\n".join(all_prompts),
            new_candidates=combined_cands,
            summary=summary,
            mode=resolved_mode,
            refine_count=refine_actual,
            wildcard_count=wildcard_actual,
            providers_used=prov_ids,
        )

    else:
        # SINGLE PROVIDER PATH
        target_backend = req.providers[0] if (req.providers and len(req.providers) == 1) else (req.backend or "auto")
        try:
            prov = registry.resolve_provider(target_backend)
        except Exception as e:
            yield format_sse("error", {"detail": str(e)})
            return

        yield format_sse(
            "provider_dispatched",
            {
                "providers": [prov.id],
                "fan_out": False,
                "display_name": prov.display_name,
                "model": prov.model,
            },
        )
        await asyncio.sleep(0.01)

        def _execute_single():
            from showdown.evolve import execute_evolution
            return execute_evolution(
                tournament=tournament,
                count=req.count,
                instructions=req.instructions,
                backend=prov.id,
                mode=resolved_mode,
                wildcards=req.wildcards,
                chain_mode=req.chain_mode,
                storage=storage,
            )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, _execute_single)

        for c in response.new_candidates:
            yield format_sse("candidate_ready", c.model_dump())
            await asyncio.sleep(0.01)

    # Persist newly generated candidates to tournament storage
    for cand in response.new_candidates:
        tournament.candidates.append(cand)
    storage.save_tournament(tournament)

    # Notify any live match SSE listeners
    event_hub.publish(
        tournament.id,
        "candidates_added",
        {
            "tournament_id": tournament.id,
            "new_count": len(response.new_candidates),
            "total_candidates": len(tournament.candidates),
            "mode": response.mode,
            "providers_used": response.providers_used,
        },
    )

    # Yield completion payload
    yield format_sse("complete", response.model_dump())
