"""LLM-as-a-judge automated tournament runner with position-bias mitigation."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from showdown.engine import check_convergence, get_dynamic_k_factor, select_matchup, update_elo
from showdown.models import (
    Candidate,
    CandidateStats,
    JudgeMatchResult,
    JudgeResponse,
    Match,
    TaskType,
    Tournament,
)
from showdown.storage import Storage

DEFAULT_RUBRICS: Dict[str, str] = {
    "text": (
        "1. Relevance and accuracy to instructions/prompt\n"
        "2. Clarity, structure, flow, and tone\n"
        "3. Conciseness and absence of fluff"
    ),
    "markdown": (
        "1. Content accuracy and relevance\n"
        "2. Effective markdown structure (headers, lists, tables, code blocks)\n"
        "3. Readability and polished presentation"
    ),
    "code": (
        "1. Correctness, edge-case safety, and algorithmic efficiency\n"
        "2. Idiomatic syntax, readability, and modern conventions\n"
        "3. Minimal extraneous dependencies and robust error handling"
    ),
    "diff": (
        "1. Correctness and precision of changes\n"
        "2. Minimal collateral edits and clean patch formatting\n"
        "3. Semantic safety and backwards compatibility"
    ),
    "json": (
        "1. Strict JSON validity\n"
        "2. Complete schema coverage and accurate typing\n"
        "3. Clean structure and normalized data"
    ),
    "svg": (
        "1. Visual aesthetics, balance, and modern vector styling\n"
        "2. Valid SVG syntax and appropriate viewBox\n"
        "3. Responsive vector paths and scalable elements"
    ),
    "image": (
        "1. Visual relevance and faithfulness to prompt\n"
        "2. Aesthetic appeal, composition, and detail clarity"
    ),
}


def resolve_judge_backend(backend: Optional[str] = None) -> str:
    """Resolve the judge backend to an available executable or API."""
    resolved = backend or os.environ.get("SHOWDOWN_BACKEND") or "auto"
    if resolved == "auto":
        if os.environ.get("SHOWDOWN_PREFER_HOMELAB") and shutil.which("omp"):
            return "omp"
        elif shutil.which("claude"):
            return "claude"
        elif shutil.which("omp"):
            return "omp"
        elif shutil.which("codex"):
            return "codex"
        elif os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "omp" if shutil.which("omp") else "cli_fallback"
    return resolved


def build_judge_prompt(
    candidate_a: Candidate,
    candidate_b: Candidate,
    task_prompt: Optional[str] = None,
    rubric: Optional[str] = None,
    task_type: TaskType = TaskType.TEXT,
) -> str:
    """Build the head-to-head comparison prompt for the judge model."""
    type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
    resolved_rubric = rubric or DEFAULT_RUBRICS.get(type_str, DEFAULT_RUBRICS["text"])

    prompt_context = f"\n[Task Prompt / Instructions]:\n{task_prompt}\n" if task_prompt else ""

    return f"""You are an impartial, expert evaluator judging a blind head-to-head comparison between Candidate A and Candidate B.
{prompt_context}
[Evaluation Criteria / Rubric]:
{resolved_rubric}

Candidate A:
```
{candidate_a.content}
```

Candidate B:
```
{candidate_b.content}
```

Instructions:
1. Objectively critique Candidate A and Candidate B against the rubric.
2. Compare their specific strengths and weaknesses in 1-3 sentences.
3. Declare the decisive winner: 'A', 'B', or 'TIE' (only if genuinely indistinguishable).

You must format your final decision as valid JSON at the end of your response:
```json
{{
  "critique": "Brief 1-3 sentence comparison explaining why the winner prevailed.",
  "winner": "A"
}}
```
Valid winner values: "A", "B", or "TIE".
"""


def parse_judge_output(raw_output: str) -> Tuple[str, str]:
    """
    Parse the judge's critique and winner ('a', 'b', 'tie') from raw LLM output.
    Robustly handles markdown blocks, stray text, and partial JSON.
    """
    # 1. Search for fenced json block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    json_str = match.group(1) if match else None

    # 2. If no fence, search for outermost JSON object
    if not json_str:
        obj_match = re.search(r"\{[\s\S]*?\"winner\"[\s\S]*?\}", raw_output)
        if obj_match:
            json_str = obj_match.group(0)

    winner = None
    critique = raw_output.strip()

    if json_str:
        try:
            data = json.loads(json_str)
            raw_winner = str(data.get("winner", "")).strip().upper()
            if raw_winner == "A":
                winner = "a"
            elif raw_winner == "B":
                winner = "b"
            elif raw_winner in ("TIE", "EQUAL", "DRAW"):
                winner = "tie"
            if "critique" in data and data["critique"]:
                critique = str(data["critique"]).strip()
            if winner is not None:
                return winner, critique
        except Exception:
            pass

    # 3. Fallback regex for winner
    w_match = re.search(r"\"?winner\"?\s*[:=]\s*\"?(?:candidate\s+)?([AB]|TIE|EQUAL|DRAW)\b", raw_output, re.IGNORECASE)
    if w_match:
        val = w_match.group(1).upper()
        if val == "A":
            winner = "a"
        elif val == "B":
            winner = "b"
        elif val in ("TIE", "EQUAL", "DRAW"):
            winner = "tie"

    # Truncate critique if overly verbose
    if len(critique) > 500:
        critique = critique[:497] + "..."

    return winner, critique


def call_judge_backend(prompt: str, backend: str = "auto") -> str:
    """Execute evaluation prompt through the selected model backend."""
    resolved = resolve_judge_backend(backend)

    if resolved == "claude" and shutil.which("claude"):
        res = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Claude CLI judge failed: {res.stderr.strip()}")
        return res.stdout.strip()

    elif resolved in ("omp", "homelab", "homelab-default") and shutil.which("omp"):
        model_name = os.environ.get("SHOWDOWN_HOMELAB_MODEL", "homelab-default")
        res = subprocess.run(
            ["omp", "-p", f"--model={model_name}", "--no-session", "--no-tools", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"OMP Homelab CLI judge failed: {res.stderr.strip()}")
        return res.stdout.strip()

    elif resolved == "codex" and shutil.which("codex"):
        res = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Codex CLI judge failed: {res.stderr.strip()}")
        return res.stdout.strip()

    elif resolved == "openai":
        import urllib.request

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("SHOWDOWN_MODEL", "gpt-4o-mini")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise LLM evaluation judge that outputs JSON comparisons.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
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
            return data["choices"][0]["message"]["content"].strip()

    raise RuntimeError(
        f"No suitable execution backend found for judge '{resolved}'. Ensure 'claude', 'omp', 'codex', or OPENAI_API_KEY is available."
    )


def evaluate_pair(
    candidate_a: Candidate,
    candidate_b: Candidate,
    task_prompt: Optional[str] = None,
    rubric: Optional[str] = None,
    task_type: TaskType = TaskType.TEXT,
    backend: str = "auto",
    swap_positions: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a pair of candidates with position-bias mitigation (order swapping).
    Returns dict with final winner ('a', 'b', 'tie'), critique notes, and swap consistency.
    """
    # Presentation 1: A vs B
    prompt_1 = build_judge_prompt(candidate_a, candidate_b, task_prompt, rubric, task_type)
    raw_1 = call_judge_backend(prompt_1, backend=backend)
    w1, c1 = parse_judge_output(raw_1)

    if not w1:
        raise ValueError(f"Judge output in presentation 1 could not be parsed as a valid decision: {raw_1[:200]}")

    if not swap_positions:
        return {
            "winner": w1,
            "critique": c1,
            "swapped_consistent": True,
        }

    # Presentation 2: B vs A (mitigate positional bias)
    prompt_2 = build_judge_prompt(candidate_b, candidate_a, task_prompt, rubric, task_type)
    raw_2 = call_judge_backend(prompt_2, backend=backend)
    w2, c2 = parse_judge_output(raw_2)

    if not w2:
        raise ValueError(f"Judge output in presentation 2 could not be parsed as a valid decision: {raw_2[:200]}")

    # Map presentation 2 result back to candidate_a and candidate_b:
    # In presentation 2: Candidate A is candidate_b, Candidate B is candidate_a.
    # So if w2 == "a" -> candidate_b won; if w2 == "b" -> candidate_a won.
    if w2 == "a":
        p2_winner = "b"
    elif w2 == "b":
        p2_winner = "a"
    else:
        p2_winner = "tie"

    # Analyze cross-presentation concordance
    if w1 == p2_winner:
        # Perfectly consistent across presentations
        winner = w1
        consistent = True
        critique = f"[Consistent] {c1}"
    elif (w1 == "a" and w2 == "a") or (w1 == "b" and w2 == "b"):
        # Position bias detected: judge picked presentation slot 1 (or 2) both times
        winner = "abstain"
        consistent = False
        favored_slot = "slot 1 (first presented)" if w1 == "a" else "slot 2 (second presented)"
        critique = f"[Position-Bias Contradiction] Judge favored {favored_slot} in both presentations. Decision abstained without moving Elo. Critique 1: {c1} | Critique 2: {c2}"
    else:
        # Asymmetric tie or disagreement (judge nondeterminism)
        winner = "abstain"
        consistent = False
        critique = f"[Judge Nondeterminism] Disagreement across presentations ({w1} vs {p2_winner}). Decision abstained without moving Elo. {c1}"

    return {
        "winner": winner,
        "critique": critique,
        "swapped_consistent": consistent,
    }


def run_tournament_judge(
    tournament: Tournament,
    rounds: int = 5,
    backend: Optional[str] = "auto",
    rubric: Optional[str] = None,
    swap_positions: bool = True,
    voter: Optional[str] = None,
    mode: str = "active",
    stop_on_convergence: bool = False,
    storage: Optional[Storage] = None,
) -> JudgeResponse:
    """
    Run automated LLM-as-a-judge tournament rounds against the tournament.
    Applies dynamic Elo updates, position-bias mitigation, and convergence tracking.
    """
    if len(tournament.candidates) < 2:
        raise ValueError("Tournament must contain at least 2 candidates for judging.")

    resolved_backend = resolve_judge_backend(backend)
    if voter:
        judge_voter = voter if voter.startswith("judge:") else f"judge:{voter}"
    else:
        judge_voter = f"judge:{resolved_backend}"

    # Server-side rubric input is strictly string text (prevents arbitrary file read vulnerabilities)
    resolved_rubric = rubric

    results: List[JudgeMatchResult] = []
    consistent_count = 0
    contradictions = 0

    for _ in range(rounds):
        pair = select_matchup(tournament, mode=mode)
        if not pair:
            break

        cand_a, cand_b = pair
        eval_result = evaluate_pair(
            candidate_a=cand_a,
            candidate_b=cand_b,
            task_prompt=tournament.prompt,
            rubric=resolved_rubric,
            task_type=tournament.task_type,
            backend=resolved_backend,
            swap_positions=swap_positions,
        )

        winner = eval_result["winner"]
        critique = eval_result["critique"]
        consistent = eval_result["swapped_consistent"]

        if swap_positions:
            if consistent:
                consistent_count += 1
            else:
                contradictions += 1

        # Atomically update tournament state under lock to prevent overwriting concurrent human votes/triage
        if storage:
            with storage.lock_tournament(tournament.id):
                fresh = storage.load_tournament(tournament.id) or tournament
                stat_a = fresh.stats.get(cand_a.id, CandidateStats())
                stat_b = fresh.stats.get(cand_b.id, CandidateStats())
                elo_a_before = stat_a.elo
                elo_b_before = stat_b.elo

                if winner in ("a", "b", "tie"):
                    k = get_dynamic_k_factor(stat_a.matches, stat_b.matches)
                    if winner == "a":
                        score_a = 1.0
                        stat_a.wins += 1
                        stat_b.losses += 1
                    elif winner == "b":
                        score_a = 0.0
                        stat_b.wins += 1
                        stat_a.losses += 1
                    else:  # tie
                        score_a = 0.5
                        stat_a.ties += 1
                        stat_b.ties += 1

                    stat_a.matches += 1
                    stat_b.matches += 1
                    elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a, k_factor=k)
                    stat_a.elo = elo_a_after
                    stat_b.elo = elo_b_after
                else:  # abstain - position bias or nondeterminism does not move Elo
                    elo_a_after = elo_a_before
                    elo_b_after = elo_b_before

                match_rec = Match(
                    id_a=cand_a.id,
                    id_b=cand_b.id,
                    winner=winner,
                    elo_a_before=elo_a_before,
                    elo_b_before=elo_b_before,
                    elo_a_after=elo_a_after,
                    elo_b_after=elo_b_after,
                    voter=judge_voter,
                    notes=critique,
                    timestamp=time.time(),
                )
                fresh.matches.append(match_rec)
                fresh.updated_at = time.time()
                storage.save_tournament(fresh)

                # Sync local in-memory view for next matchup selection
                tournament.stats = fresh.stats
                tournament.matches = fresh.matches

        else:
            stat_a = tournament.stats.get(cand_a.id, CandidateStats())
            stat_b = tournament.stats.get(cand_b.id, CandidateStats())
            elo_a_before = stat_a.elo
            elo_b_before = stat_b.elo

            if winner in ("a", "b", "tie"):
                k = get_dynamic_k_factor(stat_a.matches, stat_b.matches)
                if winner == "a":
                    score_a = 1.0
                    stat_a.wins += 1
                    stat_b.losses += 1
                elif winner == "b":
                    score_a = 0.0
                    stat_b.wins += 1
                    stat_a.losses += 1
                else:
                    score_a = 0.5
                    stat_a.ties += 1
                    stat_b.ties += 1

                stat_a.matches += 1
                stat_b.matches += 1
                elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, score_a, k_factor=k)
                stat_a.elo = elo_a_after
                stat_b.elo = elo_b_after
            else:
                elo_a_after = elo_a_before
                elo_b_after = elo_b_before

            match_rec = Match(
                id_a=cand_a.id,
                id_b=cand_b.id,
                winner=winner,
                elo_a_before=elo_a_before,
                elo_b_before=elo_b_before,
                elo_a_after=elo_a_after,
                elo_b_after=elo_b_after,
                voter=judge_voter,
                notes=critique,
                timestamp=time.time(),
            )
            tournament.matches.append(match_rec)
            tournament.updated_at = time.time()

        results.append(
            JudgeMatchResult(
                id_a=cand_a.id,
                id_b=cand_b.id,
                winner=winner,
                critique=critique,
                swapped_consistent=consistent,
                elo_a_after=elo_a_after,
                elo_b_after=elo_b_after,
            )
        )

        converged, confidence = check_convergence(tournament.candidates, tournament.stats)
        if stop_on_convergence and converged:
            break

    converged, confidence = check_convergence(tournament.candidates, tournament.stats)

    return JudgeResponse(
        tournament_id=tournament.id,
        backend=resolved_backend,
        voter=judge_voter,
        rounds_requested=rounds,
        matches_evaluated=len(results),
        consistent_matches=consistent_count,
        contradictions=contradictions,
        converged=converged,
        confidence=confidence,
        results=results,
    )

