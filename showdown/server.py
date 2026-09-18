"""FastAPI server for Showdown output ranking arena."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from showdown.engine import select_matchup, update_elo
from showdown.models import (
    CandidateStats,
    CreateTournamentRequest,
    Match,
    TriageRecord,
    TriageRequest,
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
    t_id = req.id or f"tournament_{int(time.time())}"
    now = time.time()
    tournament = Tournament(
        id=t_id,
        title=req.title,
        prompt=req.prompt,
        task_type=req.task_type,
        candidates=req.candidates,
        created_at=now,
        updated_at=now,
    )
    for c in tournament.candidates:
        tournament.stats[c.id] = CandidateStats()

    storage.save_tournament(tournament)
    return tournament


@app.get("/api/tournaments/{tournament_id}", response_model=Tournament)
def get_tournament(tournament_id: str):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return tournament


@app.delete("/api/tournaments/{tournament_id}")
def delete_tournament(tournament_id: str):
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

    cand_a, cand_b = pair
    return {
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "elo_a": tournament.stats[cand_a.id].elo,
        "elo_b": tournament.stats[cand_b.id].elo,
    }


@app.post("/api/tournaments/{tournament_id}/vote")
def record_vote(tournament_id: str, vote: VoteRequest):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    if vote.id_a not in tournament.stats or vote.id_b not in tournament.stats:
        raise HTTPException(status_code=400, detail="Invalid candidate IDs")

    stat_a = tournament.stats[vote.id_a]
    stat_b = tournament.stats[vote.id_b]

    elo_a_before = stat_a.elo
    elo_b_before = stat_b.elo

    if vote.winner == "a":
        score_a = 1.0
        stat_a.wins += 1
        stat_b.losses += 1
    elif vote.winner == "b":
        score_a = 0.0
        stat_b.wins += 1
        stat_a.losses += 1
    elif vote.winner == "tie":
        score_a = 0.5
        stat_a.ties += 1
        stat_b.ties += 1
    else:  # both_bad
        score_a = 0.5

    stat_a.matches += 1
    stat_b.matches += 1

    elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a)
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


@app.get("/api/tournaments/{tournament_id}/export")
def export_tournament(
    tournament_id: str,
    format: str = Query("dpo", pattern="^(dpo|leaderboard|matches|raw)$"),
    download: bool = Query(False),
):
    tournament = storage.load_tournament(tournament_id)
    if not tournament:
        raise HTTPException(status_code=404, detail="Tournament not found")

    cand_map = {c.id: c for c in tournament.candidates}

    if format == "dpo":
        # Direct Preference Optimization pairs
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
        media_type = "application/json"
        filename = f"{tournament.id}_dpo.json"

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
        media_type = "application/json"
        filename = f"{tournament.id}_leaderboard.json"

    else:
        content = tournament.model_dump()
        media_type = "application/json"
        filename = f"{tournament.id}.json"

    if download:
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=json.dumps(content, indent=2), media_type=media_type, headers=headers)

    return content
