"""Multi-generational simulation harness, logo arena, and chained narrative evaluation."""

import json
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from showdown.engine import apply_bradley_terry_stats, fit_bradley_terry, replay_stats
from showdown.evolve import execute_evolution
from showdown.judge import evaluate_pair, run_tournament_judge
from showdown.models import (
    Candidate,
    CandidateStats,
    Match,
    TaskType,
    Tournament,
    TournamentStatus,
    TriageRecord,
    TriageStatus,
)
from showdown.storage import Storage


class SimulationHarness:
    """Automated simulation harness running multi-generational evolution, self-play, and convergence proofs."""

    def __init__(self, storage: Optional[Storage] = None):
        self.storage = storage or Storage()

    def run_generational_simulation(
        self,
        title: str = "Generational Simulation",
        prompt: str = "Generate punchy brand slogans for developer tools",
        task_type: TaskType = TaskType.TEXT,
        generations: int = 3,
        candidates_per_gen: int = 4,
        judge_rounds_per_gen: int = 6,
        judge_backend: str = "auto",
        generation_backend: str = "auto",
        generation_providers: Optional[List[str]] = None,
        evolution_mode: str = "hybrid",
        wildcards: Optional[int] = 1,
        initial_candidates: Optional[List[Candidate]] = None,
    ) -> Dict[str, Any]:
        """
        Run an N-generation self-play simulation measuring Elo progression,
        convergence confidence, and a final head-to-head validation battle (Gen N vs Gen 1).
        """
        tourney_id = f"sim_gen_{uuid.uuid4().hex[:8]}"

        # Step 1: Initialize Generation 1 pool
        if initial_candidates:
            cands = [c.model_copy() for c in initial_candidates]
            for c in cands:
                c.generation = 1
        else:
            # Cold start seed candidates
            cands = [
                Candidate(
                    id=f"gen1_{i+1}",
                    label=f"Gen 1 Baseline #{i+1}",
                    content=f"Baseline concept #{i+1} for {prompt}",
                    generation=1,
                    metadata={"provider": "baseline", "lineage": "seed"},
                )
                for i in range(candidates_per_gen)
            ]

        tournament = Tournament(
            id=tourney_id,
            title=title,
            prompt=prompt,
            task_type=task_type,
            candidates=cands,
            stats={c.id: CandidateStats() for c in cands},
        )
        self.storage.save_tournament(tournament)

        generation_history: List[Dict[str, Any]] = []

        # Step 2: Run generational loop (Gen 1 to Gen N)
        for gen_idx in range(1, generations + 1):
            gen_cands = [c for c in tournament.candidates if getattr(c, "generation", 1) == gen_idx]

            # Run judge rounds on current generation pool
            judge_res = run_tournament_judge(
                tournament=tournament,
                rounds=judge_rounds_per_gen,
                backend=judge_backend,
                swap_positions=True,
                mode="active",
                storage=self.storage,
            )

            # Recalculate stats & Bradley-Terry parameters
            apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)

            # Identify top performer of this generation
            gen_stats = [(c, tournament.stats.get(c.id, CandidateStats())) for c in gen_cands]
            gen_stats.sort(key=lambda item: item[1].elo, reverse=True)
            top_cand, top_stat = gen_stats[0] if gen_stats else (None, None)

            # Auto-triage: favorite the top performer, dislike lowest
            if top_cand:
                tournament.triage[top_cand.id] = TriageRecord(
                    status=TriageStatus.FAVORITE,
                    notes=f"Top performer of Gen {gen_idx} (Elo: {top_stat.elo:.1f})",
                )
            if len(gen_stats) > 1:
                bottom_cand = gen_stats[-1][0]
                tournament.triage[bottom_cand.id] = TriageRecord(
                    status=TriageStatus.DISLIKED,
                    notes=f"Lowest performer of Gen {gen_idx}",
                )

            self.storage.save_tournament(tournament)

            gen_avg_elo = (
                sum(s.elo for _, s in gen_stats) / len(gen_stats) if gen_stats else 1200.0
            )

            generation_history.append({
                "generation": gen_idx,
                "candidate_count": len(gen_cands),
                "top_candidate": {
                    "id": top_cand.id if top_cand else None,
                    "label": top_cand.label if top_cand else None,
                    "content": top_cand.content if top_cand else None,
                    "elo": round(top_stat.elo, 1) if top_stat else 1200.0,
                    "provider": top_cand.metadata.get("provider") if top_cand else None,
                },
                "average_elo": round(gen_avg_elo, 1),
                "matches_evaluated": getattr(judge_res, "matches_evaluated", 0),
                "converged": getattr(judge_res, "converged", False),
            })

            # Synthesize Generation N+1 if not on final generation
            if gen_idx < generations:
                evolve_res = execute_evolution(
                    tournament=tournament,
                    count=candidates_per_gen,
                    backend=generation_backend,
                    providers=generation_providers,
                    mode=evolution_mode,
                    wildcards=wildcards,
                    storage=self.storage,
                )
                for new_c in evolve_res.new_candidates:
                    new_c.generation = gen_idx + 1
                    tournament.candidates.append(new_c)
                    tournament.stats[new_c.id] = CandidateStats()

                self.storage.save_tournament(tournament)

        # Step 3: Final Validation Battle (Gen N vs Gen 1 Head-to-Head Showdown)
        gen1_cands = [c for c in tournament.candidates if getattr(c, "generation", 1) == 1]
        final_gen_cands = [c for c in tournament.candidates if getattr(c, "generation", 1) == generations]

        validation_matches: List[Dict[str, Any]] = []
        gen_n_wins = 0
        gen_1_wins = 0
        ties = 0

        # Run pairwise battles between final generation contenders and gen 1 baseline
        for fn_c in final_gen_cands[:3]:
            for g1_c in gen1_cands[:3]:
                eval_res = evaluate_pair(
                    candidate_a=fn_c,
                    candidate_b=g1_c,
                    task_prompt=tournament.prompt or tournament.title,
                    backend=judge_backend,
                    swap_positions=True,
                )
                winner_slot = eval_res.get("winner")
                critique = eval_res.get("critique", "")

                if winner_slot == "a":
                    gen_n_wins += 1
                    battle_winner = "final_gen"
                elif winner_slot == "b":
                    gen_1_wins += 1
                    battle_winner = "gen_1"
                else:
                    ties += 1
                    battle_winner = "tie"

                validation_matches.append({
                    "final_gen_candidate": fn_c.id,
                    "gen_1_candidate": g1_c.id,
                    "winner": battle_winner,
                    "critique": critique,
                })

        total_validation = len(validation_matches)
        gen_n_win_rate = (
            round((gen_n_wins / total_validation) * 100.0, 1) if total_validation > 0 else 0.0
        )
        objective_convergence_proven = gen_n_wins > gen_1_wins

        # Overall accepted winner is the top contender of the final generation
        best_overall = max(tournament.candidates, key=lambda c: tournament.stats.get(c.id, CandidateStats()).elo)
        tournament.accepted_candidate_id = best_overall.id
        tournament.status = TournamentStatus.COMPLETED
        self.storage.save_tournament(tournament)

        return {
            "tournament_id": tournament.id,
            "title": tournament.title,
            "generations_run": generations,
            "total_candidates": len(tournament.candidates),
            "generation_history": generation_history,
            "validation_battle": {
                "matches_played": total_validation,
                "gen_n_wins": gen_n_wins,
                "gen_1_wins": gen_1_wins,
                "ties": ties,
                "gen_n_win_rate_pct": gen_n_win_rate,
                "objective_convergence_proven": objective_convergence_proven,
                "matches": validation_matches,
            },
            "winning_candidate": {
                "id": best_overall.id,
                "label": best_overall.label,
                "content": best_overall.content,
                "generation": getattr(best_overall, "generation", generations),
                "elo": round(tournament.stats.get(best_overall.id, CandidateStats()).elo, 1),
            },
        }

    def run_logo_arena_simulation(
        self,
        company_name: str = "Apex Systems",
        concept: str = "Modern, high-performance distributed systems engineering firm",
        rounds: int = 5,
        judge_backend: str = "auto",
    ) -> Dict[str, Any]:
        """
        Scenario 2: Multimodal & Visual Vector SVG Logo Arena.
        Evaluates geometric and color balance across vector candidates.
        """
        tourney_id = f"sim_logo_{uuid.uuid4().hex[:8]}"

        # Seed initial SVG candidate geometries
        logo_seeds = [
            (
                "Geometric Monogram",
                f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><polygon points="100,20 180,180 20,180" fill="#2563eb"/><text x="100" y="140" font-family="sans-serif" font-size="28" fill="#ffffff" text-anchor="middle">{company_name[:4]}</text></svg>',
            ),
            (
                "Abstract Circuit Hexagon",
                f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><polygon points="100,20 170,60 170,140 100,180 30,140 30,60" fill="none" stroke="#10b981" stroke-width="8"/><circle cx="100" cy="100" r="30" fill="#10b981"/></svg>',
            ),
            (
                "Minimalist Wordmark",
                f'<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><rect x="20" y="20" width="160" height="160" rx="20" fill="#0f172a"/><text x="100" y="115" font-family="monospace" font-size="32" font-weight="bold" fill="#38bdf8" text-anchor="middle">{company_name[:1]}</text></svg>',
            ),
        ]

        candidates = [
            Candidate(
                id=f"logo_{i+1}",
                label=name,
                content=svg,
                generation=1,
                metadata={"task": "logo_design", "company": company_name},
            )
            for i, (name, svg) in enumerate(logo_seeds)
        ]

        prompt = f"Design a memorable, minimalist vector SVG logo for '{company_name}': {concept}."
        tournament = Tournament(
            id=tourney_id,
            title=f"Logo Arena: {company_name}",
            prompt=prompt,
            task_type=TaskType.SVG,
            candidates=candidates,
            stats={c.id: CandidateStats() for c in candidates},
        )
        self.storage.save_tournament(tournament)

        rubric = "Evaluate vector visual balance, brand scalability, aesthetic minimalism, and clarity of geometry."
        judge_res = run_tournament_judge(
            tournament=tournament,
            rounds=rounds,
            backend=judge_backend,
            rubric=rubric,
            swap_positions=True,
            mode="active",
            storage=self.storage,
        )

        apply_bradley_terry_stats(tournament.candidates, tournament.matches, tournament.stats)
        winner = max(tournament.candidates, key=lambda c: tournament.stats[c.id].elo)
        tournament.accepted_candidate_id = winner.id
        tournament.status = TournamentStatus.COMPLETED
        self.storage.save_tournament(tournament)

        return {
            "tournament_id": tournament.id,
            "company_name": company_name,
            "candidates_count": len(tournament.candidates),
            "matches_evaluated": getattr(judge_res, "matches_evaluated", 0),
            "winning_logo": {
                "id": winner.id,
                "label": winner.label,
                "svg_content": winner.content,
                "elo": round(tournament.stats[winner.id].elo, 1),
            },
        }

    def run_chained_narrative_simulation(
        self,
        project_title: str = "Distributed Runtime Spec",
        stages_config: Optional[List[Dict[str, Any]]] = None,
        judge_backend: str = "auto",
        generation_backend: str = "auto",
    ) -> Dict[str, Any]:
        """
        Scenario 3: Chained Sequential Paragraph-by-Paragraph / Stage Pipeline.
        Each winning selection conditions the context and generation pool of the child stage.
        """
        default_stages = [
            {"title": "Architecture Overview", "prompt": "Define the core distributed consensus protocol and topology."},
            {"title": "Failover & Partition Strategy", "prompt": "Specify Byzantine fault tolerance and dynamic network recovery."},
            {"title": "Client SDK & API Surface", "prompt": "Outline the developer experience, wire protocol, and ergonomics."},
        ]
        stages = stages_config or default_stages

        parent_id = None
        assembled_document: List[Dict[str, str]] = []
        tournament_ids: List[str] = []

        for idx, stage in enumerate(stages):
            stage_title = f"{project_title} - Stage {idx+1}: {stage['title']}"
            t_id = f"sim_chain_{uuid.uuid4().hex[:8]}"

            # Generate candidate variations for this stage
            cands = [
                Candidate(
                    id=f"stage{idx+1}_cand{i+1}",
                    label=f"Approach #{i+1}",
                    content=f"[{stage['title']} Approach #{i+1}]: Detailed technical specification for {stage['prompt']}",
                    generation=1,
                    metadata={"stage": idx + 1, "topic": stage["title"]},
                )
                for i in range(3)
            ]

            upstream_context = None
            if assembled_document:
                upstream_context = "\n\n".join(
                    f"### {doc['stage']}:\n{doc['content']}" for doc in assembled_document
                )

            t = Tournament(
                id=t_id,
                title=stage_title,
                prompt=stage["prompt"],
                task_type=TaskType.TEXT,
                candidates=cands,
                parent_ids=[parent_id] if parent_id else [],
                context=upstream_context,
                chain_mode="growth",
                stats={c.id: CandidateStats() for c in cands},
            )
            self.storage.save_tournament(t)
            tournament_ids.append(t_id)

            # Evaluate stage candidates with judge
            run_tournament_judge(
                tournament=t,
                rounds=3,
                backend=judge_backend,
                swap_positions=True,
                storage=self.storage,
            )

            apply_bradley_terry_stats(t.candidates, t.matches, t.stats)
            stage_winner = max(t.candidates, key=lambda c: t.stats[c.id].elo)
            t.accepted_candidate_id = stage_winner.id
            t.status = TournamentStatus.COMPLETED
            self.storage.save_tournament(t)

            assembled_document.append({
                "stage": stage["title"],
                "winner_id": stage_winner.id,
                "content": stage_winner.content,
            })
            parent_id = t_id

        full_compiled_text = "\n\n---\n\n".join(
            f"## {doc['stage']}\n{doc['content']}" for doc in assembled_document
        )

        return {
            "project_title": project_title,
            "stages_completed": len(stages),
            "tournament_ids": tournament_ids,
            "assembled_document": assembled_document,
            "full_text": full_compiled_text,
        }
