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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from showdown.engine import (
    check_convergence,
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
    VoteRequest,
)
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
_tournament_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


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

    now = time.time()
    tournament = Tournament(
        id=t_id,
        title=req.title,
        prompt=req.prompt,
        task_type=req.task_type,
        candidates=req.candidates,
        parent_ids=req.parent_ids,
        context=req.context,
        created_at=now,
        updated_at=now,
    )
    for c in tournament.candidates:
        tournament.stats[c.id] = CandidateStats()

    with _tournament_locks[t_id]:
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

    return tournament


@app.delete("/api/tournaments/{tournament_id}")
def delete_tournament(tournament_id: str):
    with _tournament_locks[tournament_id]:
        success = storage.delete_tournament(tournament_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {"status": "deleted", "id": tournament_id}


@app.get("/api/tournaments/{tournament_id}/matchup")
def get_matchup(tournament_id: str):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    pair = select_matchup(tournament)
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
    }


@app.post("/api/tournaments/{tournament_id}/vote")
def record_vote(tournament_id: str, vote: VoteRequest):
    with _tournament_locks[tournament_id]:
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

        storage.save_tournament(tournament)

    return {
        "status": "recorded",
        "elo_a_after": elo_a_after,
        "elo_b_after": elo_b_after,
    }


@app.post("/api/tournaments/{tournament_id}/triage")
def record_triage(tournament_id: str, req: TriageRequest):
    with _tournament_locks[tournament_id]:
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        tournament.triage[req.candidate_id] = TriageRecord(
            status=req.status,
            notes=req.notes,
            updated_at=time.time(),
        )
        tournament.updated_at = time.time()
        storage.save_tournament(tournament)
    return {"status": "saved", "candidate_id": req.candidate_id}


@app.post("/api/tournaments/{tournament_id}/candidates", response_model=Tournament)
def add_candidates(tournament_id: str, req: AddCandidatesRequest):
    with _tournament_locks[tournament_id]:
        tournament = storage.load_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")

        existing_ids = {c.id for c in tournament.candidates}
        for c in req.candidates:
            if c.id in existing_ids:
                continue
            tournament.candidates.append(c)
            tournament.stats[c.id] = CandidateStats()

        tournament.updated_at = time.time()
        storage.save_tournament(tournament)
    return tournament


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
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with _tournament_locks[tournament_id]:
        # Reload fresh tournament to avoid overwriting votes or triage recorded during generation
        fresh_tournament = storage.load_tournament(tournament_id) or tournament
        existing_ids = {c.id for c in fresh_tournament.candidates}
        for c in evolve_res.new_candidates:
            if c.id not in existing_ids:
                fresh_tournament.candidates.append(c)
                fresh_tournament.stats[c.id] = CandidateStats()

        fresh_tournament.updated_at = time.time()
        storage.save_tournament(fresh_tournament)
    return evolve_res


@app.post("/api/tournaments/{tournament_id}/accept")
def accept_candidate(tournament_id: str, req: AcceptCandidateRequest):
    with _tournament_locks[tournament_id]:
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


@app.get("/api/tournaments/{tournament_id}/export")
def export_tournament(
    tournament_id: str,
    format: str = "dpo",
    download: bool = False,
    jsonl: bool = False,
):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    cand_map = {c.id: c for c in tournament.candidates}

    if format == "dpo":
        pairs = []
        for m in tournament.matches:
            if m.winner in ("a", "b"):
                chosen_id = m.id_a if m.winner == "a" else m.id_b
                rejected_id = m.id_b if m.winner == "a" else m.id_a
                chosen_cand = cand_map.get(chosen_id)
                rejected_cand = cand_map.get(rejected_id)
                if chosen_cand and rejected_cand:
                    pairs.append({
                        "prompt": tournament.prompt or tournament.title,
                        "chosen": chosen_cand.content,
                        "rejected": rejected_cand.content,
                        "chosen_id": chosen_id,
                        "rejected_id": rejected_id,
                        "timestamp": m.timestamp,
                    })
        content = pairs
        filename = f"{tournament.id}_dpo.{'jsonl' if jsonl else 'json'}"

    elif format == "kto":
        kto_records = []
        for cid, t_rec in tournament.triage.items():
            cand = cand_map.get(cid)
            if not cand:
                continue
            if t_rec.status in (TriageStatus.LIKED, TriageStatus.FAVORITE):
                kto_records.append({
                    "prompt": tournament.prompt or tournament.title,
                    "completion": cand.content,
                    "label": True,
                    "candidate_id": cid,
                    "status": t_rec.status.value,
                })
            elif t_rec.status == TriageStatus.DISLIKED:
                kto_records.append({
                    "prompt": tournament.prompt or tournament.title,
                    "completion": cand.content,
                    "label": False,
                    "candidate_id": cid,
                    "status": t_rec.status.value,
                })
        content = kto_records
        filename = f"{tournament.id}_kto.{'jsonl' if jsonl else 'json'}"

    elif format == "leaderboard":
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

    else:
        content = tournament.model_dump()
        filename = f"{tournament.id}.json"

    if jsonl and isinstance(content, list):
        body = "\n".join(json.dumps(row) for row in content)
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
