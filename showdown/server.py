"""FastAPI server for Showdown output ranking arena."""

import json
import re
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from showdown.realtime import (
    event_hub,
    stream_evolution_progress,
    stream_tournament_events,
)

from showdown.engine import (
    apply_bradley_terry_stats,
    check_convergence,
    compute_inter_annotator_agreement,
    compute_voter_consistency,
    fit_bradley_terry,
    get_dynamic_k_factor,
    replay_stats,
    select_matchup,
    update_elo,
)
from showdown.models import (
    AcceptCandidateRequest,
    AddCandidatesRequest,
    Candidate,
    CandidateStats,
    CreateTournamentRequest,
    EvolveRequest,
    EvolveResponse,
    Match,
    TournamentStatus,
    TriageRecord,
    TriageRequest,
    TriageStatus,
    Tournament,
    UpdateTournamentRequest,
    VoteRequest,
    JudgeRequest,
    JudgeResponse,
    ProviderInfo,
    ProviderLeaderboardEntry,
    StepAnnotation,
    StepAnnotationRequest,
    StepAnnotationTag,
)
from showdown.providers import compute_provider_leaderboard, registry
from showdown.storage import Storage

app = FastAPI(title="Showdown Arena", description="Minimalist Human & LLM Output Ranking Arena")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = Storage()
WEB_DIR = Path(__file__).parent / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = WEB_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text())
    return HTMLResponse("<h1>Showdown Arena</h1><p>Web interface not found.</p>")


@app.get("/api/tournaments", response_model=List[Tournament])
def list_tournaments():
    return storage.list_tournaments()


@app.post("/api/tournaments", response_model=Tournament)
def create_tournament(req: CreateTournamentRequest):
    if req.id:
        t_id = req.id
        if storage.has_tournament(t_id) and not req.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Tournament with ID '{t_id}' already exists. Provide a unique ID or set overwrite=true."
            )
    else:
        # Generate clean human-readable slug with collision protection
        clean_title = re.sub(r'[^a-zA-Z0-9_]+', '_', req.title.lower().strip()).strip('_')[:32]
        if not clean_title:
            clean_title = "tournament"
        t_id = f"{clean_title}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        while storage.has_tournament(t_id):
            t_id = f"{clean_title}_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    resolved_context = req.context
    if (not resolved_context or not resolved_context.strip()) and req.parent_ids:
        from showdown.evolve import resolve_upstream_context
        up = resolve_upstream_context(req.parent_ids, storage)
        if up and up.get("content"):
            resolved_context = f"Stage: {up.get('parent_title', 'Parent Stage')}\nWinner ({up.get('label', '')}):\n{up.get('content', '')}"

    now = time.time()
    tournament = Tournament(
        id=t_id,
        title=req.title,
        prompt=req.prompt,
        task_type=req.task_type,
        candidates=req.candidates,
        parent_ids=req.parent_ids,
        chain_mode=req.chain_mode or "growth",
        context=resolved_context,
        blinded=req.blinded,
        created_at=now,
        updated_at=now,
    )
    for c in tournament.candidates:
        tournament.stats[c.id] = CandidateStats()

    apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

    with storage.lock_tournament(t_id):
        storage.save_tournament(tournament)
    return tournament


@app.get("/api/tournaments/{tournament_id}", response_model=Tournament)
def get_tournament(tournament_id: str, voter: Optional[str] = None):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    # Extract all distinct voters present in match history
    tournament.voters = sorted(list({m.voter for m in tournament.matches if m.voter}))

    # If a specific evaluator is requested, replay ratings for that evaluator on the fly
    if voter and voter.strip().lower() not in ("all", "pooled", "*"):
        tournament.stats = replay_stats(tournament.candidates, tournament.matches, voter=voter.strip())
    elif any(s.bt_elo is None for s in tournament.stats.values()):
        apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

    return tournament


@app.delete("/api/tournaments/{tournament_id}")
def delete_tournament(tournament_id: str):
    with storage.lock_tournament(tournament_id):
        success = storage.delete_tournament(tournament_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {"status": "deleted", "id": tournament_id}


@app.get("/api/tournaments/{tournament_id}/matchup")
def get_matchup(
    tournament_id: str,
    mode: str = "active",
):
    if mode not in ("active", "controversial", "close", "info_gain"):
        raise HTTPException(status_code=400, detail=f"Invalid matchup mode '{mode}'. Must be 'active', 'controversial', 'close', or 'info_gain'.")

    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    if any(s.bt_elo is None for s in tournament.stats.values()):
        apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

    pair = select_matchup(tournament, mode=mode)
    if not pair:
        raise HTTPException(status_code=204, detail="Not enough candidates for a matchup")

    converged, confidence = check_convergence(tournament.candidates, tournament.stats)
    cand_a, cand_b = pair
    return {
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "elo_a": tournament.stats[cand_a.id].elo,
        "elo_b": tournament.stats[cand_b.id].elo,
        "converged": converged,
        "confidence": confidence,
        "mode": mode,
    }


@app.post("/api/tournaments/{tournament_id}/vote")
def record_vote(tournament_id: str, vote: VoteRequest):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        if vote.id_a not in tournament.stats or vote.id_b not in tournament.stats:
            raise HTTPException(status_code=400, detail="Invalid candidate IDs")

        stat_a = tournament.stats[vote.id_a]
        stat_b = tournament.stats[vote.id_b]

        elo_a_before = stat_a.elo
        elo_b_before = stat_b.elo

        k = get_dynamic_k_factor(stat_a.matches, stat_b.matches)

        if vote.winner == "a":
            score_a = 1.0
            stat_a.wins += 1
            stat_b.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a, k_factor=k)
        elif vote.winner == "b":
            score_a = 0.0
            stat_b.wins += 1
            stat_a.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a, k_factor=k)
        elif vote.winner == "tie":
            score_a = 0.5
            stat_a.ties += 1
            stat_b.ties += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a, k_factor=k)
        else:  # both_bad
            stat_a.losses += 1
            stat_b.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            penalty = round(k / 2.0, 2)
            elo_a_after = max(100.0, round(elo_a_before - penalty, 2))
            elo_b_after = max(100.0, round(elo_b_before - penalty, 2))

        stat_a.elo = elo_a_after
        stat_b.elo = elo_b_after

        match_record = Match(
            id_a=vote.id_a,
            id_b=vote.id_b,
            winner=vote.winner,
            elo_a_before=elo_a_before,
            elo_b_before=elo_b_before,
            elo_a_after=elo_a_after,
            elo_b_after=elo_b_after,
            voter=vote.voter,
            notes=vote.notes,
            timestamp=time.time(),
        )
        tournament.matches.append(match_record)
        tournament.updated_at = time.time()

        # Update Bradley-Terry parameters across matches
        apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

        storage.save_tournament(tournament)

    event_hub.publish(
        tournament_id,
        "match_recorded",
        {
            "tournament_id": tournament_id,
            "match": match_record.model_dump(),
            "elo_a_after": elo_a_after,
            "elo_b_after": elo_b_after,
            "matches_count": len(tournament.matches),
        },
    )

    return {
        "status": "recorded",
        "elo_a_after": elo_a_after,
        "elo_b_after": elo_b_after,
    }


@app.post("/api/tournaments/{tournament_id}/undo")
def undo_vote(tournament_id: str, voter: Optional[str] = None):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        if not tournament.matches:
            raise HTTPException(status_code=400, detail="No votes to undo")

        if voter and voter.strip().lower() not in ("all", "pooled", "*"):
            v_clean = voter.strip()
            idx = None
            for i in range(len(tournament.matches) - 1, -1, -1):
                if tournament.matches[i].voter == v_clean:
                    idx = i
                    break
            if idx is None:
                raise HTTPException(status_code=400, detail=f"No votes by '{v_clean}' to undo")
            undone_match = tournament.matches.pop(idx)
        else:
            undone_match = tournament.matches.pop()

        tournament.stats = replay_stats(tournament.candidates, tournament.matches)
        tournament.updated_at = time.time()
        storage.save_tournament(tournament)

    event_hub.publish(
        tournament_id,
        "vote_undone",
        {
            "tournament_id": tournament_id,
            "undone_match": undone_match.model_dump(),
            "remaining_matches": len(tournament.matches),
        },
    )

    return {
        "status": "undone",
        "remaining_matches": len(tournament.matches),
        "undone_match": {
            "id_a": undone_match.id_a,
            "id_b": undone_match.id_b,
            "winner": undone_match.winner,
            "voter": undone_match.voter,
        },
    }


@app.patch("/api/tournaments/{tournament_id}")
def update_tournament(tournament_id: str, req: UpdateTournamentRequest):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        if req.title is not None:
            tournament.title = req.title
        if req.prompt is not None:
            tournament.prompt = req.prompt
        if req.blinded is not None:
            tournament.blinded = req.blinded
        if req.context is not None:
            tournament.context = req.context

        tournament.updated_at = time.time()
        storage.save_tournament(tournament)

    return {
        "status": "updated",
        "tournament_id": tournament.id,
        "blinded": tournament.blinded,
    }


@app.get("/api/tournaments/{tournament_id}/consistency")
def get_voter_consistency(tournament_id: str, voter: Optional[str] = None):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    data = compute_voter_consistency(tournament.matches, voter=voter)
    data["tournament_id"] = tournament.id
    return data


@app.post("/api/tournaments/{tournament_id}/triage")
def record_triage(tournament_id: str, req: TriageRequest):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        triage_rec = TriageRecord(
            status=req.status,
            notes=req.notes,
            updated_at=time.time(),
        )
        tournament.triage[req.candidate_id] = triage_rec
        tournament.updated_at = time.time()
        storage.save_tournament(tournament)

    event_hub.publish(
        tournament_id,
        "triage_updated",
        {
            "tournament_id": tournament_id,
            "candidate_id": req.candidate_id,
            "record": triage_rec.model_dump(),
        },
    )

    return {"status": "saved", "candidate_id": req.candidate_id}


@app.post("/api/tournaments/{tournament_id}/candidates", response_model=Tournament)
def add_candidates(tournament_id: str, req: AddCandidatesRequest):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        existing_ids = {c.id for c in tournament.candidates}
        for c in req.candidates:
            if c.id in existing_ids:
                continue
            tournament.candidates.append(c)
            tournament.stats[c.id] = CandidateStats()

        apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

        tournament.updated_at = time.time()
        storage.save_tournament(tournament)
    return tournament


@app.get("/api/tournaments/{tournament_id}/candidates/{candidate_id}/trajectory")
def get_candidate_trajectory(tournament_id: str, candidate_id: str):
    """Retrieve parsed trajectory steps and execution summary for a candidate."""
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    cand = next((c for c in tournament.candidates if c.id == candidate_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    from showdown.trajectories import compute_trajectory_summary, extract_candidate_steps
    steps = extract_candidate_steps(cand)
    summary = compute_trajectory_summary(steps)
    return {
        "candidate_id": candidate_id,
        "steps": [s.model_dump() for s in steps],
        "summary": summary.model_dump(),
    }


@app.post("/api/tournaments/{tournament_id}/candidates/{candidate_id}/steps/{step_index}/annotate")
def annotate_candidate_step(tournament_id: str, candidate_id: str, step_index: int, req: StepAnnotationRequest):
    """Add or update step-level commentary / tag for Process Reward Models & SPO."""
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        cand = next((c for c in tournament.candidates if c.id == candidate_id), None)
        if not cand:
            raise HTTPException(status_code=404, detail="Candidate not found")

        from showdown.trajectories import compute_trajectory_summary, extract_candidate_steps

        # Validate that the candidate has this step_index
        existing_steps = extract_candidate_steps(cand)
        valid_indices = {s.step_index for s in existing_steps}
        if step_index not in valid_indices:
            raise HTTPException(
                status_code=404,
                detail=f"Step index {step_index} not found on candidate '{candidate_id}'. Available steps: {sorted(list(valid_indices))}",
            )

        annotation = StepAnnotation(
            step_index=step_index,
            tag=req.tag,
            notes=req.notes,
            voter=req.voter or "human",
            timestamp=time.time(),
        )

        meta = cand.metadata.setdefault("step_annotations", {})
        key = str(step_index)
        existing_list = meta.setdefault(key, [])
        voter_id = req.voter or "human"
        # Overwrite previous annotation by this voter on this step
        meta[key] = [a for a in existing_list if a.get("voter") != voter_id]
        meta[key].append(annotation.model_dump())

        tournament.updated_at = time.time()
        storage.save_tournament(tournament)

        steps = extract_candidate_steps(cand)
        summary = compute_trajectory_summary(steps)

    return {
        "status": "annotated",
        "candidate_id": candidate_id,
        "step_index": step_index,
        "annotation": annotation.model_dump(),
        "summary": summary.model_dump(),
    }


@app.delete("/api/tournaments/{tournament_id}/candidates/{candidate_id}/steps/{step_index}/annotate")
def delete_candidate_step_annotation(tournament_id: str, candidate_id: str, step_index: int, voter: Optional[str] = "human"):
    """Delete a step-level annotation."""
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        cand = next((c for c in tournament.candidates if c.id == candidate_id), None)
        if not cand:
            raise HTTPException(status_code=404, detail="Candidate not found")

        from showdown.trajectories import extract_candidate_steps
        existing_steps = extract_candidate_steps(cand)
        valid_indices = {s.step_index for s in existing_steps}
        if step_index not in valid_indices:
            raise HTTPException(
                status_code=404,
                detail=f"Step index {step_index} not found on candidate '{candidate_id}'",
            )

        meta = cand.metadata.get("step_annotations", {})
        key = str(step_index)
        if key in meta:
            target_voter = voter or "human"
            meta[key] = [a for a in meta[key] if a.get("voter") != target_voter]
            if not meta[key]:
                del meta[key]

        tournament.updated_at = time.time()
        storage.save_tournament(tournament)

    return {"status": "deleted", "candidate_id": candidate_id, "step_index": step_index}


@app.post("/api/tournaments/{tournament_id}/evolve", response_model=EvolveResponse)
def evolve_tournament_endpoint(tournament_id: str, req: EvolveRequest):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    try:
        from showdown.evolve import execute_evolution
        evolve_res = execute_evolution(
            tournament=tournament,
            count=req.count,
            instructions=req.instructions,
            backend=req.backend,
            providers=req.providers,
            mode=req.mode or "refine",
            wildcards=req.wildcards,
            chain_mode=req.chain_mode,
            storage=storage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with storage.lock_tournament(tournament_id):
        # Reload fresh tournament to avoid overwriting votes or triage recorded during generation
        fresh_tournament = storage.load_tournament(tournament_id) or tournament
        existing_ids = {c.id for c in fresh_tournament.candidates}
        for c in evolve_res.new_candidates:
            if c.id not in existing_ids:
                fresh_tournament.candidates.append(c)
                fresh_tournament.stats[c.id] = CandidateStats()

        fresh_tournament.updated_at = time.time()
        storage.save_tournament(fresh_tournament)

    event_hub.publish(
        tournament_id,
        "candidates_added",
        {
            "tournament_id": tournament_id,
            "new_count": len(evolve_res.new_candidates),
            "total_candidates": len(fresh_tournament.candidates),
            "mode": evolve_res.mode,
            "providers_used": evolve_res.providers_used,
        },
    )

    return evolve_res


@app.post("/api/tournaments/{tournament_id}/judge", response_model=JudgeResponse)
def judge_tournament_endpoint(tournament_id: str, req: JudgeRequest):
    if req.mode not in ("active", "controversial", "close"):
        raise HTTPException(status_code=400, detail=f"Invalid judge mode '{req.mode}'. Must be 'active', 'controversial', or 'close'.")

    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    try:
        from showdown.judge import run_tournament_judge
        judge_res = run_tournament_judge(
            tournament=tournament,
            rounds=req.rounds,
            backend=req.backend,
            rubric=req.rubric,
            swap_positions=req.swap_positions,
            voter=req.voter,
            mode=req.mode,
            stop_on_convergence=req.stop_on_convergence,
            storage=storage,
        )
        return judge_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tournaments/{tournament_id}/accept")
def accept_candidate(tournament_id: str, req: AcceptCandidateRequest):
    with storage.lock_tournament(tournament_id):
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        cand_map = {c.id: c for c in tournament.candidates}
        if req.candidate_id not in cand_map:
            raise HTTPException(status_code=400, detail="Candidate ID not found in tournament")

        tournament.accepted_candidate_id = req.candidate_id
        tournament.status = TournamentStatus.COMPLETED
        tournament.updated_at = time.time()

        tournament.triage[req.candidate_id] = TriageRecord(
            status=TriageStatus.FAVORITE,
            notes=req.notes or "Selected as tournament winner",
            updated_at=time.time(),
        )
        storage.save_tournament(tournament)

    accepted_cand = cand_map[req.candidate_id]
    stat = tournament.stats.get(req.candidate_id, CandidateStats())

    event_hub.publish(
        tournament_id,
        "candidate_accepted",
        {
            "tournament_id": tournament_id,
            "candidate_id": req.candidate_id,
            "notes": req.notes,
        },
    )

    return {
        "status": "completed",
        "accepted_candidate_id": req.candidate_id,
        "candidate": accepted_cand.model_dump(),
        "elo": stat.elo,
        "wins": stat.wins,
        "losses": stat.losses,
        "notes": req.notes,
    }


@app.get("/api/tournaments/{tournament_id}/chain")
def get_tournament_chain(tournament_id: str):
    """
    Traverse parent lineage to assemble multi-stage chained tournaments into a coherent document.
    """
    stages = []
    curr_id = tournament_id
    visited = set()

    while curr_id and curr_id not in visited:
        visited.add(curr_id)
        t = storage.load_tournament(curr_id)
        if not t:
            break
        cand_map = {c.id: c for c in t.candidates}
        accepted = cand_map.get(t.accepted_candidate_id) if t.accepted_candidate_id else None
        stages.append({
            "tournament_id": t.id,
            "title": t.title,
            "prompt": t.prompt,
            "task_type": t.task_type.value if hasattr(t.task_type, "value") else str(t.task_type),
            "accepted_candidate": accepted.model_dump() if accepted else None,
            "context": t.context,
            "parent_ids": t.parent_ids,
        })
        curr_id = t.parent_ids[0] if t.parent_ids else None

    stages.reverse()
    assembled = "\n\n".join(
        [s["accepted_candidate"]["content"] for s in stages if s.get("accepted_candidate") and s["accepted_candidate"].get("content")]
    )
    return {
        "stages": stages,
        "assembled_content": assembled,
        "count": len(stages),
    }


@app.get("/api/tournaments/{tournament_id}/wait")
def wait_for_completion(
    tournament_id: str,
    timeout: int = Query(default=30, ge=1, le=120),
    poll_interval: float = 0.5,
):
    """
    Long-polling endpoint for autonomous agents to block until a human or judge
    selects an accepted winning candidate or completes the tournament.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        if tournament.status == TournamentStatus.COMPLETED and tournament.accepted_candidate_id:
            cand_map = {c.id: c for c in tournament.candidates}
            accepted_cand = cand_map.get(tournament.accepted_candidate_id)
            stat = tournament.stats.get(tournament.accepted_candidate_id, CandidateStats())
            return {
                "completed": True,
                "status": tournament.status.value,
                "accepted_candidate": accepted_cand.model_dump() if accepted_cand else None,
                "elo": stat.elo if stat else None,
                "matches_played": len(tournament.matches),
            }

        time.sleep(poll_interval)

    # Return current leading candidate if timeout expires
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    sorted_cands = sorted(
        tournament.candidates,
        key=lambda c: tournament.stats.get(c.id, CandidateStats()).elo,
        reverse=True,
    )
    top_cand = sorted_cands[0] if sorted_cands else None
    top_stat = tournament.stats.get(top_cand.id, CandidateStats()) if top_cand else None

    return {
        "completed": False,
        "status": tournament.status.value,
        "accepted_candidate": None,
        "leading_candidate": top_cand.model_dump() if top_cand else None,
        "leading_elo": top_stat.elo if top_stat else None,
        "matches_played": len(tournament.matches),
    }


@app.get("/api/tournaments/{tournament_id}/agreement")
def get_inter_annotator_agreement(tournament_id: str):
    """
    Calculate and return inter-annotator agreement metrics across all judges in this tournament.
    """
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    agreement_data = compute_inter_annotator_agreement(tournament.matches)
    agreement_data["tournament_id"] = tournament.id
    agreement_data["title"] = tournament.title
    return agreement_data


@app.get("/api/tournaments/{tournament_id}/export")
def export_tournament(
    tournament_id: str,
    format: str = "dpo",
    voter: Optional[str] = None,
    consensus: Optional[str] = None,
    min_agreement: Optional[float] = None,
    dedup: bool = True,
    include_critique: bool = True,
    download: bool = False,
    jsonl: bool = False,
):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    from showdown.export import export_dataset, format_jsonl

    if format == "leaderboard":
        sorted_cands = sorted(
            tournament.candidates,
            key=lambda c: tournament.stats.get(c.id, CandidateStats()).elo,
            reverse=True,
        )
        content = [
            {
                "rank": i + 1,
                "id": c.id,
                "label": c.label,
                "elo": tournament.stats.get(c.id, CandidateStats()).elo,
                "matches": tournament.stats.get(c.id, CandidateStats()).matches,
                "wins": tournament.stats.get(c.id, CandidateStats()).wins,
                "losses": tournament.stats.get(c.id, CandidateStats()).losses,
                "ties": tournament.stats.get(c.id, CandidateStats()).ties,
            }
            for i, c in enumerate(sorted_cands)
        ]
        filename = f"{tournament.id}_leaderboard.{'jsonl' if jsonl else 'json'}"
    elif format == "raw":
        content = tournament.model_dump()
        filename = f"{tournament.id}.json"
    else:
        try:
            content = export_dataset(
                tournaments=[tournament],
                format=format,
                voter=voter,
                consensus=consensus,
                min_agreement=min_agreement,
                dedup=dedup,
                include_critique=include_critique,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        filename = f"{tournament.id}_{format}.{'jsonl' if jsonl else 'json'}"

    if jsonl and isinstance(content, list):
        body = format_jsonl(content)
        media_type = "application/x-ndjson"
    else:
        body = json.dumps(content, indent=2)
        media_type = "application/json"

    if download:
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=body, media_type=media_type, headers=headers)

    if jsonl:
        return Response(content=body, media_type=media_type)

    return content


@app.get("/api/export")
def export_all(
    format: str = "dpo",
    task_type: Optional[str] = None,
    voter: Optional[str] = None,
    consensus: Optional[str] = None,
    min_agreement: Optional[float] = None,
    dedup: bool = True,
    include_critique: bool = True,
    split: Optional[float] = None,
    split_by: str = "lineage",
    seed: int = 42,
    download: bool = False,
    jsonl: bool = False,
):
    """Multi-tournament dataset aggregation and train/val partitioning."""
    from showdown.export import export_dataset, format_jsonl

    try:
        tournaments = storage.list_tournaments()
        data = export_dataset(
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
            seed=seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = f"showdown_{format}_{task_type or 'all'}.{'jsonl' if jsonl else 'json'}"

    if jsonl:
        if isinstance(data, list):
            body = format_jsonl(data)
        elif isinstance(data, dict) and "train" in data and "val" in data:
            tagged_records = [{"split": "train", **r} for r in data["train"]] + [{"split": "val", **r} for r in data["val"]]
            body = format_jsonl(tagged_records)
        else:
            body = json.dumps(data, indent=2)
        media_type = "application/x-ndjson"
    else:
        body = json.dumps(data, indent=2)
        media_type = "application/json"

    if download:
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=body, media_type=media_type, headers=headers)

    if jsonl:
        return Response(content=body, media_type=media_type)

    return data


@app.get("/api/providers", response_model=List[ProviderInfo])
def list_providers():
    """List all registered agent providers and their host availability."""
    return [ProviderInfo(**p.to_dict()) for p in registry.list_all()]


@app.get("/api/providers/leaderboard", response_model=List[ProviderLeaderboardEntry])
def get_provider_leaderboard():
    """Cross-tournament model/provider leaderboard based on historical matches."""
    tournaments = storage.list_tournaments()
    return compute_provider_leaderboard(tournaments)


@app.get("/api/tournaments/{tournament_id}/events")
async def tournament_events(
    tournament_id: str,
    timeout: Optional[float] = Query(None),
):
    """Server-Sent Events (SSE) stream broadcasting live match and Elo updates for a tournament."""
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    return StreamingResponse(
        stream_tournament_events(tournament_id, timeout=timeout),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tournaments/{tournament_id}/evolve/stream")
@app.post("/api/tournaments/{tournament_id}/evolve/stream")
async def evolve_tournament_stream(
    tournament_id: str,
    req: Optional[EvolveRequest] = Body(None),
    count: int = Query(5, ge=1, le=20),
    instructions: Optional[str] = Query(None),
    backend: Optional[str] = Query("auto"),
    providers: Optional[List[str]] = Query(None),
    mode: str = Query("refine"),
    wildcards: Optional[int] = Query(None),
    chain_mode: Optional[str] = Query(None),
):
    """Stream candidate evolution lifecycle progress events via Server-Sent Events (SSE)."""
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    evolve_req = req or EvolveRequest(
        count=count,
        instructions=instructions,
        backend=backend,
        providers=providers,
        mode=mode,
        wildcards=wildcards,
        chain_mode=chain_mode,
    )

    return StreamingResponse(
        stream_evolution_progress(tournament, evolve_req, storage),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


