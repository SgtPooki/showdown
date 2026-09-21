"""Preference synthesis and candidate evolution engine for Showdown."""

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from showdown.models import (
    Candidate,
    CandidateStats,
    EvolveResponse,
    TaskType,
    TriageStatus,
    Tournament,
)
from showdown.providers import AgentProvider, registry
from showdown.storage import Storage


def resolve_upstream_context(
    tournament_or_parent_ids: Union[Tournament, List[str]],
    storage: Storage,
) -> Optional[Dict[str, Any]]:
    """
    Resolve upstream context from parent stage(s).
    Inherits ONLY officially accepted winners from parent tournaments.
    Per peer review recommendations, does NOT fall back to provisional leaders.
    """
    parent_ids = (
        tournament_or_parent_ids.parent_ids
        if isinstance(tournament_or_parent_ids, Tournament)
        else tournament_or_parent_ids
    )
    if not parent_ids:
        return None

    for p_id in parent_ids:
        parent = storage.load_tournament(p_id)
        if not parent or not parent.accepted_candidate_id:
            continue
        accepted = next((c for c in parent.candidates if c.id == parent.accepted_candidate_id), None)
        if accepted:
            return {
                "tournament_id": parent.id,
                "parent_title": parent.title,
                "parent_prompt": parent.prompt,
                "id": accepted.id,
                "label": accepted.label or accepted.id,
                "content": accepted.content,
            }

    return None


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
    mode: str = "refine",
    chain_mode: Optional[str] = None,
    upstream_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Construct an evolution synthesis prompt from tournament preferences.
    Supports continuity/growth refinement and structural axis divergence/novelty injection.
    """
    prefs = extract_tournament_preferences(tournament)
    is_diverge = mode == "diverge" or (chain_mode == "divergence" and mode != "refine")

    if is_diverge:
        prompt_lines = [
            "You are an expert generation engine exploring creative divergence and structural novelty based on human preference learning.",
            f"Goal / Task: {tournament.prompt or tournament.title}",
            f"Task Type: {tournament.task_type.value}",
            "",
        ]
    else:
        prompt_lines = [
            "You are an expert generation engine optimizing candidate outputs based on human preference learning.",
            f"Goal / Task: {tournament.prompt or tournament.title}",
            f"Task Type: {tournament.task_type.value}",
            "",
        ]

    # Include upstream baseline if available
    if upstream_context:
        prompt_lines.extend([
            f"### Upstream Stage Baseline (Accepted Winner from '{upstream_context.get('parent_title', 'Parent Stage')}'):",
            f"Winner Label: {upstream_context.get('label')}",
            "Winner Content:",
            f"{upstream_context.get('content')}",
            "",
        ])
    elif tournament.context and tournament.context.strip():
        prompt_lines.extend([
            "### Upstream Stage Context / Baseline:",
            tournament.context.strip(),
            "",
        ])

    # Top performers / established winners
    if is_diverge:
        prompt_lines.append("### Baseline / Established Contenders (The anchor direction to structurally diverge from):")
    else:
        prompt_lines.append("### High-Performing Winners (The user prefers attributes found here):")

    if prefs["top_performers"]:
        for p in prefs["top_performers"]:
            note_str = f" [Note: {p['notes']}]" if p["notes"] else ""
            prompt_lines.append(f"- {p['label']} (Elo {p['elo']}): {p['content']}{note_str}")
    elif not upstream_context and not tournament.context:
        prompt_lines.append("- (No clear winners yet; explore diverse directions aligned with the goal)")

    # Bottom performers / eliminated candidates (MUST preserve even in divergence mode for quality floor)
    prompt_lines.append("\n### Low-Performing / Rejected (The user disliked or eliminated these failure modes):")
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
        type_specific_guidance = "\n   For each candidate, 'content' MUST be a clean, valid, standalone inline SVG string starting with '<svg' and ending with '</svg>', containing viewBox, width, height, and well-styled SVG elements."
    elif tournament.task_type == TaskType.CODE:
        type_specific_guidance = "\n   For each candidate, 'content' MUST be clean, executable code without outer markdown quotes."

    if is_diverge:
        prompt_lines.append(f"""
### Divergence & Structural Novelty Instructions:
Generate exactly {count} NEW candidate variations that structurally diverge and explore uncharted creative territory.
CRITICAL REQUIREMENTS FOR DIVERGENCE:
1. Structural Axis Divergence: Identify 3–4 foundational structural axes of the baseline/top performers (e.g. paradigm/methodology, viewpoint/tone, pacing, architectural philosophy, aesthetic style).
2. Distinct Stances: Each candidate MUST take a distinct, intentional stance on at least 2 of these structural axes relative to the baseline.
3. No Negation Collapse: Do NOT create outputs that merely negate or reference the baseline (e.g., never say 'Unlike the previous solution...' or write meta commentary). The candidate content must be a self-contained, high-caliber artifact that stands entirely on its own.
4. Quality Floor & Negative Constraints: Strictly respect the task goal and all user critiques and notes. Strictly avoid all failure modes, bad patterns, or themes seen in the low-performing/rejected candidates.
5. Output Format:
   Output ONLY a JSON array of objects with 'label', 'content', and 'differs_by' keys. Do not include markdown code block formatting or explanation. Example format:
   [
     {{
       "label": "Candidate Name",
       "content": "...",{type_specific_guidance}
       "differs_by": "Explores an asynchronous event-driven paradigm with terse imperative tone rather than monolithic functional structure."
     }}
   ]
""")
        summary = f"Synthesized structural divergence prompt for {count} novelty candidates from {len(prefs['top_performers'])} baseline anchors, {len(prefs['bottom_performers'])} rejected items, and {len(prefs['notes'])} notes."
    else:
        prompt_lines.append(f"""
### Generation Instructions:
Generate exactly {count} NEW distinct candidate variations that:
1. If high-performing winners or upstream baselines exist, emphasize, deepen, and refine their patterns. Otherwise, explore diverse creative variations.
2. Strictly avoid patterns, words, or styles seen in the low-performing / rejected candidates.
3. Explicitly honor the user's critiques, notes, and dislikes.
4. Keep the outputs punchy, relevant, and high caliber.{type_specific_guidance}

Output ONLY a JSON array of objects with 'label', 'content', and optional 'differs_by' keys. Do not include markdown code block formatting or explanation. Example format:
[
  {{"label": "Candidate Name", "content": "...", "differs_by": "Refinement deepening top-performing attributes"}}
]
""")
        summary = f"Synthesized preference prompt for {count} candidates from {len(prefs['top_performers'])} winners, {len(prefs['bottom_performers'])} rejected items, and {len(prefs['notes'])} notes."

    full_prompt = "\n".join(prompt_lines).strip()
    return full_prompt, summary


def _parse_candidates_json(
    raw_text: str,
    next_gen: int,
    backend_name: str,
    is_divergence: bool = False,
    is_wildcard: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Candidate]:
    """Parse JSON candidate list from model output, handling potential markdown wrappers and metadata."""
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

    prov_id = provider or backend_name
    reg_p = registry.get(prov_id)
    prov_name = reg_p.display_name if reg_p else prov_id.capitalize()
    model_name = model or (reg_p.model if reg_p else None)

    new_cands = []
    for i, item in enumerate(items):
        cid = f"gen{next_gen}_{uuid.uuid4().hex[:6]}"
        label = item.get("label") or f"Gen {next_gen} #{i+1}"
        content = item.get("content", "").strip()
        differs_by = item.get("differs_by")
        if content:
            meta: Dict[str, Any] = {
                "lineage": "diverged" if is_divergence else "evolved",
                "backend": backend_name,
                "provider": prov_id,
                "provider_name": prov_name,
                "created_at": time.time(),
            }
            if model_name:
                meta["model"] = model_name
            if differs_by:
                meta["differs_by"] = differs_by
            if is_wildcard:
                meta["wildcard"] = True

            new_cands.append(
                Candidate(
                    id=cid,
                    label=label,
                    content=content,
                    generation=next_gen,
                    metadata=meta,
                )
            )

    return new_cands


def call_generation_backend(prompt: str, backend: str) -> str:
    """Execute LLM generation backend using provider registry and return raw text output."""
    provider = registry.resolve_provider(backend)
    return provider.call(prompt)


def execute_evolution(
    tournament: Tournament,
    count: int = 5,
    instructions: Optional[str] = None,
    backend: Optional[str] = "auto",
    providers: Optional[List[str]] = None,
    mode: str = "refine",
    wildcards: Optional[int] = None,
    chain_mode: Optional[str] = None,
    storage: Optional[Storage] = None,
) -> EvolveResponse:
    """Run candidate generation through available CLI or API backends with refine/diverge/hybrid and multi-agent fan-out support."""
    # Resolve upstream context if parent stages exist
    upstream_context = None
    if storage:
        upstream_context = resolve_upstream_context(tournament, storage)

    # Determine generation number
    existing_gens = [c.generation for c in tournament.candidates if hasattr(c, "generation")]
    next_gen = (max(existing_gens) + 1) if existing_gens else 2

    resolved_mode = mode.lower() if mode else "refine"
    if resolved_mode not in ("refine", "diverge", "hybrid"):
        resolved_mode = "refine"

    # MULTI-AGENT FAN-OUT PATH
    if providers and len(providers) > 1:
        resolved_providers: List[AgentProvider] = []
        for pid in providers:
            p = registry.get(pid)
            if p and p.is_available():
                resolved_providers.append(p)
            elif p:
                raise RuntimeError(f"Requested provider '{pid}' is not available on host.")
            else:
                raise ValueError(f"Unknown provider '{pid}'. Available: {[pr.id for pr in registry.list_available()]}")

        if not resolved_providers:
            raise RuntimeError(f"None of the requested providers {providers} are available.")

        # Partition target count across providers
        n_prov = len(resolved_providers)
        counts_per_prov = [count // n_prov + (1 if i < (count % n_prov) else 0) for i in range(n_prov)]

        # If hybrid, allocate total wildcards across providers
        total_wildcards = (
            min(max(1, wildcards), count)
            if wildcards is not None
            else (max(1, count // 3) if resolved_mode == "hybrid" else 0)
        )
        wildcards_remaining = total_wildcards if resolved_mode == "hybrid" else 0

        provider_tasks: List[Tuple[AgentProvider, int, int]] = []
        for i, prov in enumerate(resolved_providers):
            p_total = counts_per_prov[i]
            if p_total == 0:
                continue
            if resolved_mode == "diverge":
                provider_tasks.append((prov, 0, p_total))
            elif resolved_mode == "refine":
                provider_tasks.append((prov, p_total, 0))
            else:  # hybrid
                w_alloc = min(p_total, wildcards_remaining)
                r_alloc = p_total - w_alloc
                wildcards_remaining -= w_alloc
                provider_tasks.append((prov, r_alloc, w_alloc))

        def _run_single_provider(prov: AgentProvider, r_count: int, w_count: int):
            prov_cands: List[Candidate] = []
            prov_prompts: List[str] = []

            if r_count > 0:
                r_prompt, _ = build_evolution_prompt(
                    tournament=tournament,
                    count=r_count,
                    instructions=instructions,
                    mode="refine",
                    chain_mode=chain_mode,
                    upstream_context=upstream_context,
                )
                r_raw = call_generation_backend(r_prompt, prov.id)
                r_cands = _parse_candidates_json(
                    r_raw,
                    next_gen,
                    backend_name=prov.id,
                    is_divergence=False,
                    is_wildcard=False,
                    provider=prov.id,
                    model=prov.model,
                )
                prov_cands.extend(r_cands[:r_count])
                prov_prompts.append(r_prompt)

            if w_count > 0:
                w_prompt, _ = build_evolution_prompt(
                    tournament=tournament,
                    count=w_count,
                    instructions=instructions,
                    mode="diverge",
                    chain_mode=chain_mode,
                    upstream_context=upstream_context,
                )
                w_raw = call_generation_backend(w_prompt, prov.id)
                w_cands = _parse_candidates_json(
                    w_raw,
                    next_gen,
                    backend_name=prov.id,
                    is_divergence=True,
                    is_wildcard=True,
                    provider=prov.id,
                    model=prov.model,
                )
                prov_cands.extend(w_cands[:w_count])
                prov_prompts.append(w_prompt)

            return prov.id, prov_cands, prov_prompts

        combined_cands: List[Candidate] = []
        all_prompts: List[str] = []
        with ThreadPoolExecutor(max_workers=min(len(provider_tasks), 8)) as executor:
            futures = [
                executor.submit(_run_single_provider, prov, r_cnt, w_cnt)
                for prov, r_cnt, w_cnt in provider_tasks
            ]
            for fut in as_completed(futures):
                pid, cands, p_list = fut.result()
                combined_cands.extend(cands)
                all_prompts.extend(p_list)

        refine_actual = sum(1 for c in combined_cands if not c.metadata.get("wildcard"))
        wildcard_actual = sum(1 for c in combined_cands if c.metadata.get("wildcard"))
        prov_ids = [p.id for p in resolved_providers]
        summary = (
            f"Multi-agent fan-out across {len(prov_ids)} providers ({', '.join(prov_ids)}): "
            f"generated {len(combined_cands)} candidates ({refine_actual} refined, {wildcard_actual} wildcards)."
        )

        return EvolveResponse(
            prompt_used="\n\n--- [FAN-OUT MULTI-AGENT] ---\n\n".join(all_prompts),
            new_candidates=combined_cands,
            summary=summary,
            mode=resolved_mode,
            refine_count=refine_actual,
            wildcard_count=wildcard_actual,
            providers_used=prov_ids,
        )

    # SINGLE PROVIDER PATH
    target_backend = providers[0] if (providers and len(providers) == 1) else (backend or "auto")
    provider_inst = registry.resolve_provider(target_backend)
    resolved_backend = provider_inst.id

    if resolved_mode == "refine":
        full_prompt, summary = build_evolution_prompt(
            tournament=tournament,
            count=count,
            instructions=instructions,
            mode="refine",
            chain_mode=chain_mode,
            upstream_context=upstream_context,
        )
        raw_output = call_generation_backend(full_prompt, resolved_backend)
        new_cands = _parse_candidates_json(
            raw_output,
            next_gen,
            backend_name=resolved_backend,
            is_divergence=False,
            is_wildcard=False,
            provider=resolved_backend,
            model=provider_inst.model,
        )[:count]
        return EvolveResponse(
            prompt_used=full_prompt,
            new_candidates=new_cands,
            summary=f"{summary} Generated {len(new_cands)} refined candidates via {provider_inst.display_name}.",
            mode="refine",
            refine_count=len(new_cands),
            wildcard_count=0,
            providers_used=[resolved_backend],
        )

    elif resolved_mode == "diverge":
        full_prompt, summary = build_evolution_prompt(
            tournament=tournament,
            count=count,
            instructions=instructions,
            mode="diverge",
            chain_mode=chain_mode,
            upstream_context=upstream_context,
        )
        raw_output = call_generation_backend(full_prompt, resolved_backend)
        new_cands = _parse_candidates_json(
            raw_output,
            next_gen,
            backend_name=resolved_backend,
            is_divergence=True,
            is_wildcard=True,
            provider=resolved_backend,
            model=provider_inst.model,
        )[:count]
        return EvolveResponse(
            prompt_used=full_prompt,
            new_candidates=new_cands,
            summary=f"{summary} Generated {len(new_cands)} structural divergence wildcards via {provider_inst.display_name}.",
            mode="diverge",
            refine_count=0,
            wildcard_count=len(new_cands),
            providers_used=[resolved_backend],
        )

    else:  # hybrid mode
        if wildcards is not None:
            w_count = min(max(1, wildcards), count)
            r_count = count - w_count
        else:
            w_count = max(1, count // 3)
            r_count = count - w_count

        combined_cands: List[Candidate] = []
        prompts: List[str] = []
        summaries: List[str] = []
        refine_actual = 0
        wildcard_actual = 0

        if r_count > 0:
            r_prompt, r_sum = build_evolution_prompt(
                tournament=tournament,
                count=r_count,
                instructions=instructions,
                mode="refine",
                chain_mode=chain_mode,
                upstream_context=upstream_context,
            )
            prompts.append(f"[Refinement Prompt]:\n{r_prompt}")
            r_raw = call_generation_backend(r_prompt, resolved_backend)
            r_cands = _parse_candidates_json(
                r_raw,
                next_gen,
                backend_name=resolved_backend,
                is_divergence=False,
                is_wildcard=False,
                provider=resolved_backend,
                model=provider_inst.model,
            )[:r_count]
            combined_cands.extend(r_cands)
            refine_actual = len(r_cands)
            summaries.append(f"{refine_actual} refinements")

        if w_count > 0:
            w_prompt, w_sum = build_evolution_prompt(
                tournament=tournament,
                count=w_count,
                instructions=instructions,
                mode="diverge",
                chain_mode=chain_mode,
                upstream_context=upstream_context,
            )
            prompts.append(f"[Divergence Wildcards Prompt]:\n{w_prompt}")
            w_raw = call_generation_backend(w_prompt, resolved_backend)
            w_cands = _parse_candidates_json(
                w_raw,
                next_gen,
                backend_name=resolved_backend,
                is_divergence=True,
                is_wildcard=True,
                provider=resolved_backend,
                model=provider_inst.model,
            )[:w_count]
            combined_cands.extend(w_cands)
            wildcard_actual = len(w_cands)
            summaries.append(f"{wildcard_actual} wildcards")

        full_prompt = "\n\n" + ("=" * 40) + "\n\n".join(prompts)
        summary = f"Hybrid generation via {provider_inst.display_name}: {', '.join(summaries)}."
        return EvolveResponse(
            prompt_used=full_prompt,
            new_candidates=combined_cands,
            summary=summary,
            mode="hybrid",
            refine_count=refine_actual,
            wildcard_count=wildcard_actual,
            providers_used=[resolved_backend],
        )
