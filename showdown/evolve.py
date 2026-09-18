"""Preference synthesis and candidate evolution engine for Showdown."""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from showdown.models import (
    Candidate,
    CandidateStats,
    EvolveResponse,
    TaskType,
    TriageStatus,
    Tournament,
)


def extract_tournament_preferences(tournament: Tournament) -> Dict[str, Any]:
    """Analyze tournament rankings, triage status, and notes to summarize user preferences."""
    cand_map = {c.id: c for c in tournament.candidates}
    stats = tournament.stats
    triage = tournament.triage

    # Sort candidates by Elo
    sorted_cands = sorted(
        tournament.candidates,
        key=lambda c: stats.get(c.id, CandidateStats()).elo,
        reverse=True,
    )

    # Collect notes from triage and matches
    user_notes: List[str] = []
    for cid, t_rec in triage.items():
        if t_rec.notes and t_rec.notes.strip():
            cand_label = cand_map.get(cid, Candidate(id=cid, content="")).label or cid
            user_notes.append(f"On '{cand_label}' ({t_rec.status.value}): {t_rec.notes.strip()}")

    for m in tournament.matches:
        if m.notes and m.notes.strip():
            cand_a = cand_map.get(m.id_a)
            cand_b = cand_map.get(m.id_b)
            label_a = cand_a.label or m.id_a if cand_a else m.id_a
            label_b = cand_b.label or m.id_b if cand_b else m.id_b
            user_notes.append(f"On matchup [{label_a} vs {label_b}] (winner: {m.winner}): {m.notes.strip()}")

    # Identify winners / favorites / top Elo
    top_performers = []
    bottom_performers = []

    for c in sorted_cands:
        s = stats.get(c.id, CandidateStats())
        t_rec = triage.get(c.id)
        is_starred = t_rec and t_rec.status in (TriageStatus.FAVORITE, TriageStatus.LIKED)
        is_rejected = t_rec and t_rec.status == TriageStatus.DISLIKED

        # Explicitly rejected candidates must never enter top_performers regardless of Elo
        if is_rejected:
            bottom_performers.append({
                "id": c.id,
                "label": c.label or c.id,
                "content": c.content,
                "elo": round(s.elo, 1),
                "record": f"{s.wins}W-{s.losses}L-{s.ties}T",
                "notes": t_rec.notes if t_rec else None,
            })
        elif is_starred:
            top_performers.append({
                "id": c.id,
                "label": c.label or c.id,
                "content": c.content,
                "elo": round(s.elo, 1),
                "record": f"{s.wins}W-{s.losses}L-{s.ties}T",
                "notes": t_rec.notes if t_rec else None,
            })
        elif s.matches > 0 and s.elo > 1200:
            top_performers.append({
                "id": c.id,
                "label": c.label or c.id,
                "content": c.content,
                "elo": round(s.elo, 1),
                "record": f"{s.wins}W-{s.losses}L-{s.ties}T",
                "notes": t_rec.notes if t_rec else None,
            })
        elif s.matches > 0 and s.elo < 1200:
            bottom_performers.append({
                "id": c.id,
                "label": c.label or c.id,
                "content": c.content,
                "elo": round(s.elo, 1),
                "record": f"{s.wins}W-{s.losses}L-{s.ties}T",
                "notes": t_rec.notes if t_rec else None,
            })

    return {
        "top_performers": top_performers[:5],
        "bottom_performers": bottom_performers[:5],
        "notes": user_notes,
    }


def build_evolution_prompt(
    tournament: Tournament,
    count: int = 5,
    instructions: Optional[str] = None,
) -> Tuple[str, str]:
    """Construct an evolution synthesis prompt from tournament preferences."""
    prefs = extract_tournament_preferences(tournament)

    prompt_lines = [
        "You are an expert generation engine optimizing candidate outputs based on human preference learning.",
        f"Goal / Task: {tournament.prompt or tournament.title}",
        f"Task Type: {tournament.task_type.value}",
        "",
        "### High-Performing Winners (The user prefers attributes found here):",
    ]

    if prefs["top_performers"]:
        for p in prefs["top_performers"]:
            note_str = f" [Note: {p['notes']}]" if p["notes"] else ""
            prompt_lines.append(f"- {p['label']} (Elo {p['elo']}): {p['content']}{note_str}")
    else:
        prompt_lines.append("- (No clear winners yet; explore diverse directions aligned with the goal)")

    prompt_lines.append("\n### Low-Performing / Rejected (The user disliked or eliminated these):")
    if prefs["bottom_performers"]:
        for p in prefs["bottom_performers"]:
            note_str = f" [Note: {p['notes']}]" if p["notes"] else ""
            prompt_lines.append(f"- {p['label']} (Elo {p['elo']}): {p['content']}{note_str}")
    else:
        prompt_lines.append("- (None recorded yet)")

    if prefs["notes"]:
        prompt_lines.append("\n### Explicit User Critiques & Feedback Notes:")
        for n in prefs["notes"]:
            prompt_lines.append(f"- {n}")

    if instructions and instructions.strip():
        prompt_lines.append(f"\n### Additional User Instructions:\n{instructions.strip()}")

    type_specific_guidance = ""
    if tournament.task_type == TaskType.SVG:
        type_specific_guidance = "\n5. For each candidate, 'content' MUST be a clean, valid, standalone inline SVG string starting with '<svg' and ending with '</svg>', containing viewBox, width, height, and well-styled SVG elements."
    elif tournament.task_type == TaskType.CODE:
        type_specific_guidance = "\n5. For each candidate, 'content' MUST be clean, executable code without outer markdown quotes."

    prompt_lines.append(f"""
### Generation Instructions:
Generate exactly {count} NEW distinct candidate variations that:
1. If high-performing winners exist, emphasize and refine their patterns. Otherwise, explore diverse creative variations.
2. Strictly avoid patterns, words, or styles seen in the low-performing / rejected candidates.
3. Explicitly honor the user's critiques, notes, and dislikes.
4. Keep the outputs punchy, relevant, and high caliber.{type_specific_guidance}

Output ONLY a JSON array of objects with 'label' and 'content' keys. Do not include markdown code block formatting or explanation. Example format:
[
  {{"label": "Candidate Name", "content": "..."}}
]
""")

    full_prompt = "\n".join(prompt_lines).strip()
    summary = f"Synthesized preferences from {len(prefs['top_performers'])} winners, {len(prefs['bottom_performers'])} rejected items, and {len(prefs['notes'])} user notes."
    return full_prompt, summary


def _parse_candidates_json(raw_text: str, next_gen: int, backend_name: str) -> List[Candidate]:
    """Parse JSON candidate list from model output, handling potential markdown wrappers."""
    # Attempt to locate JSON array in response
    text = raw_text.strip()
    # Strip CLI preambles (e.g. omp 'Working...')
    if "Working..." in text:
        text = text.split("Working...", 1)[-1].strip()
    # Strip markdown block wrappers if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("Model output was not a JSON array")

    new_cands = []
    for i, item in enumerate(items):
        cid = f"gen{next_gen}_{uuid.uuid4().hex[:6]}"
        label = item.get("label") or f"Gen {next_gen} #{i+1}"
        content = item.get("content", "").strip()
        if content:
            new_cands.append(
                Candidate(
                    id=cid,
                    label=label,
                    content=content,
                    generation=next_gen,
                    metadata={"lineage": "evolved", "backend": backend_name, "created_at": time.time()},
                )
            )

    return new_cands


def execute_evolution(
    tournament: Tournament,
    count: int = 5,
    instructions: Optional[str] = None,
    backend: Optional[str] = "auto",
) -> EvolveResponse:
    """Run candidate generation through available CLI or API backends."""
    full_prompt, summary = build_evolution_prompt(tournament, count, instructions)

    # Determine generation number
    existing_gens = [c.generation for c in tournament.candidates if hasattr(c, "generation")]
    next_gen = (max(existing_gens) + 1) if existing_gens else 2

    # Choose backend
    resolved_backend = backend or os.environ.get("SHOWDOWN_BACKEND") or "auto"
    if resolved_backend == "auto":
        if os.environ.get("SHOWDOWN_PREFER_HOMELAB") and shutil.which("omp"):
            resolved_backend = "omp"
        elif shutil.which("claude"):
            resolved_backend = "claude"
        elif shutil.which("omp"):
            resolved_backend = "omp"
        elif shutil.which("codex"):
            resolved_backend = "codex"
        elif os.environ.get("OPENAI_API_KEY"):
            resolved_backend = "openai"
        else:
            resolved_backend = "cli_fallback"

    raw_output = ""

    if resolved_backend == "claude" and shutil.which("claude"):
        res = subprocess.run(
            ["claude", "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {res.stderr.strip()}")
        raw_output = res.stdout.strip()

    elif resolved_backend in ("omp", "homelab", "homelab-default") and shutil.which("omp"):
        model_name = os.environ.get("SHOWDOWN_HOMELAB_MODEL", "homelab-default")
        res = subprocess.run(
            ["omp", "-p", f"--model={model_name}", "--no-session", "--no-tools", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"OMP Homelab CLI failed: {res.stderr.strip()}")
        raw_output = res.stdout.strip()

    elif resolved_backend == "codex" and shutil.which("codex"):
        res = subprocess.run(
            ["codex", "exec", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Codex CLI failed: {res.stderr.strip()}")
        raw_output = res.stdout.strip()

    elif resolved_backend == "openai":
        import urllib.request

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        payload = {
            "model": os.environ.get("SHOWDOWN_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": "You are a precise preference optimization assistant that outputs only valid JSON."},
                {"role": "user", "content": full_prompt},
            ],
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_output = data["choices"][0]["message"]["content"].strip()

    else:
        # Graceful fallback if no LLM executable is configured
        raise RuntimeError(
            f"No suitable generation backend found for '{resolved_backend}'. Ensure 'claude', 'codex', or OPENAI_API_KEY is available."
        )

    new_candidates = _parse_candidates_json(raw_output, next_gen, resolved_backend)
    return EvolveResponse(
        prompt_used=full_prompt,
        new_candidates=new_candidates,
        summary=f"{summary} Generated {len(new_candidates)} new candidates via {resolved_backend}.",
    )
