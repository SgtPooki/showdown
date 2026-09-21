"""Elo rating calculation and active tournament pairing engine."""

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from showdown.models import Candidate, CandidateStats, Match, Tournament

DEFAULT_ELO = 1200.0
K_FACTOR = 32.0


def get_dynamic_k_factor(matches_a: int = 0, matches_b: int = 0) -> float:
    """
    Calculate dynamic K-factor:
    - High K (48.0) for early matches (< 5) to accelerate rating convergence.
    - Medium K (32.0) for intermediate matches (5-15).
    - Stable K (24.0) for well-evaluated candidates (>= 15).
    """
    avg_matches = (matches_a + matches_b) / 2.0
    if avg_matches < 5:
        return 48.0
    elif avg_matches < 15:
        return 32.0
    else:
        return 24.0


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate the expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    actual_score_a: float,
    k_factor: float = K_FACTOR
) -> Tuple[float, float]:
    """
    Update Elo ratings given actual score for A (1.0=win, 0.5=tie, 0.0=loss).
    Returns (new_rating_a, new_rating_b).
    """
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    actual_score_b = 1.0 - actual_score_a

    new_a = rating_a + k_factor * (actual_score_a - expected_a)
    new_b = rating_b + k_factor * (actual_score_b - expected_b)

    return round(new_a, 2), round(new_b, 2)


def select_matchup(tournament: Tournament) -> Optional[Tuple[Candidate, Candidate]]:
    """
    Select an optimal next pair of candidates for comparison.
    Strategy:
    1. Prioritize candidates with the fewest matches played (exploration).
    2. Pair candidates with similar Elo ratings (exploitation / highest information gain).
    3. Avoid recently repeated matchups.
    """
    candidates = tournament.candidates
    if len(candidates) < 2:
        return None

    # Track match history counts between pairs
    pair_counts: Dict[Tuple[str, str], int] = {}
    for m in tournament.matches:
        pair = tuple(sorted([m.id_a, m.id_b]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Sort candidates by number of matches played
    def get_matches(c: Candidate) -> int:
        stat = tournament.stats.get(c.id)
        return stat.matches if stat else 0

    sorted_by_matches = sorted(candidates, key=get_matches)

    # For small pools (<= 6), pick strictly the least-evaluated candidate.
    # For larger pools, choose among top 3 least-evaluated to add mild exploration.
    if len(sorted_by_matches) <= 6:
        candidate_a = sorted_by_matches[0]
    else:
        pool_size = min(3, len(sorted_by_matches))
        candidate_a = random.choice(sorted_by_matches[:pool_size])

    # Find candidate B: prefer similar Elo, fewer mutual matchups
    stat_a = tournament.stats.get(candidate_a.id, CandidateStats())
    elo_a = stat_a.elo

    candidates_b = [c for c in candidates if c.id != candidate_a.id]

    def score_opponent(c: Candidate) -> float:
        stat_b = tournament.stats.get(c.id, CandidateStats())
        elo_diff = abs(elo_a - stat_b.elo)
        pair = tuple(sorted([candidate_a.id, c.id]))
        prior_matches = pair_counts.get(pair, 0)
        # Lower score is better: heavily penalize repeated matchups, lightly penalize large Elo gaps
        return (prior_matches * 300.0) + elo_diff + random.uniform(0, 20.0)

    candidates_b.sort(key=score_opponent)
    candidate_b = candidates_b[0]

    # Randomize A/B side to eliminate left/right presentation bias
    if random.random() > 0.5:
        return candidate_a, candidate_b
    return candidate_b, candidate_a


def replay_stats(
    candidates: List[Candidate],
    matches: List[Match],
    voter: Optional[str] = None,
) -> Dict[str, CandidateStats]:
    """
    Replay candidate stats from match history.
    If voter is specified (and not 'all' or 'pooled'), only matches cast by that voter are replayed.
    Dynamic K-factors are calculated using that evaluator's match progression.
    """
    stats: Dict[str, CandidateStats] = {c.id: CandidateStats() for c in candidates}

    filtered_matches = matches
    if voter and voter.strip().lower() not in ("all", "pooled", "*"):
        filtered_matches = [m for m in matches if m.voter == voter]

    for m in filtered_matches:
        if m.id_a not in stats or m.id_b not in stats:
            continue
        stat_a = stats[m.id_a]
        stat_b = stats[m.id_b]

        elo_a_before = stat_a.elo
        elo_b_before = stat_b.elo

        k = get_dynamic_k_factor(stat_a.matches, stat_b.matches)

        if m.winner == "a":
            stat_a.wins += 1
            stat_b.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, 1.0, k_factor=k)
        elif m.winner == "b":
            stat_b.wins += 1
            stat_a.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, 0.0, k_factor=k)
        elif m.winner == "tie":
            stat_a.ties += 1
            stat_b.ties += 1
            stat_a.matches += 1
            stat_b.matches += 1
            elo_a_after, elo_b_after = update_elo(elo_a_before, elo_b_before, 0.5, k_factor=k)
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

    return stats


def check_convergence(
    candidates: List[Candidate],
    stats: Dict[str, CandidateStats],
    min_matches_per_cand: int = 2,
    lead_margin: float = 35.0,
) -> Tuple[bool, float]:
    """
    Check if a tournament's top candidate is statistically separated.
    Returns (is_converged, confidence_score_0_to_1).
    """
    if len(candidates) < 2:
        return True, 1.0

    sorted_stats = sorted([stats.get(c.id, CandidateStats()) for c in candidates], key=lambda s: s.elo, reverse=True)
    min_played = min(s.matches for s in sorted_stats)
    if min_played < min_matches_per_cand:
        progress = min(1.0, sum(s.matches for s in sorted_stats) / max(1, len(candidates) * min_matches_per_cand))
        return False, round(progress * 0.5, 2)

    top_elo = sorted_stats[0].elo
    second_elo = sorted_stats[1].elo
    margin = top_elo - second_elo

    confidence = min(1.0, max(0.0, margin / lead_margin))
    is_converged = margin >= lead_margin and min_played >= min_matches_per_cand
    return is_converged, round(confidence, 2)


def compute_inter_annotator_agreement(
    matches: List[Match],
) -> Dict[str, Any]:
    """
    Calculate inter-annotator agreement metrics across all distinct pairs of evaluators.
    Canonicalizes pairs (min(id_a, id_b), max(id_a, id_b)) to detect directional alignment or reversals.
    """
    voters = sorted(list({m.voter for m in matches if m.voter}))
    if len(voters) < 2:
        return {
            "voters": voters,
            "total_matches": len(matches),
            "evaluator_pairs": [],
            "overall_agreement_rate": None,
            "message": "At least 2 distinct evaluators required to compute inter-annotator agreement.",
        }

    # Map: voter -> { (c1, c2): direction } where c1 < c2
    # direction: +1 if c1 preferred, -1 if c2 preferred, 0 if tie/both_bad
    voter_decisions: Dict[str, Dict[Tuple[str, str], int]] = {v: {} for v in voters}

    for m in matches:
        if not m.voter:
            continue
        c1, c2 = (m.id_a, m.id_b) if m.id_a < m.id_b else (m.id_b, m.id_a)
        pair = (c1, c2)

        if m.winner == "a":
            direction = 1 if m.id_a == c1 else -1
        elif m.winner == "b":
            direction = -1 if m.id_a == c1 else 1
        else:
            direction = 0

        voter_decisions[m.voter][pair] = direction

    evaluator_pairs = []
    total_agreements = 0
    total_decisive = 0

    for i in range(len(voters)):
        for j in range(i + 1, len(voters)):
            v1 = voters[i]
            v2 = voters[j]
            decisions_1 = voter_decisions[v1]
            decisions_2 = voter_decisions[v2]

            shared_pairs = set(decisions_1.keys()) & set(decisions_2.keys())
            if not shared_pairs:
                continue

            agreed = 0
            reversals = 0
            ties = 0

            for pair in shared_pairs:
                d1 = decisions_1[pair]
                d2 = decisions_2[pair]
                if d1 == 0 or d2 == 0:
                    ties += 1
                elif d1 == d2:
                    agreed += 1
                else:
                    reversals += 1

            decisive = agreed + reversals
            agreement_rate = round(agreed / decisive, 3) if decisive > 0 else 1.0
            reversal_rate = round(reversals / decisive, 3) if decisive > 0 else 0.0

            evaluator_pairs.append({
                "evaluator_a": v1,
                "evaluator_b": v2,
                "shared_pairs_count": len(shared_pairs),
                "decisive_pairs_count": decisive,
                "agreements": agreed,
                "reversals": reversals,
                "ties": ties,
                "agreement_rate": agreement_rate,
                "reversal_rate": reversal_rate,
            })

            total_agreements += agreed
            total_decisive += decisive

    overall_rate = round(total_agreements / total_decisive, 3) if total_decisive > 0 else None

    return {
        "voters": voters,
        "total_matches": len(matches),
        "shared_pairs_evaluated": total_decisive,
        "overall_agreement_rate": overall_rate,
        "evaluator_pairs": evaluator_pairs,
    }


def compute_voter_consistency(
    matches: List[Match],
    voter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure consistency across repeated evaluations of identical candidate pairs by the same voter.
    Returns consistent pairs count, self-reversals, and consistency rate.
    """
    filtered_matches = matches
    if voter and voter.strip().lower() not in ("all", "pooled", "*"):
        filtered_matches = [m for m in matches if m.voter == voter]

    pair_history: Dict[Tuple[str, Tuple[str, str]], List[int]] = {}

    for m in filtered_matches:
        v = m.voter or "anonymous"
        c1, c2 = (m.id_a, m.id_b) if m.id_a < m.id_b else (m.id_b, m.id_a)
        key = (v, (c1, c2))

        if m.winner == "a":
            d = 1 if m.id_a == c1 else -1
        elif m.winner == "b":
            d = -1 if m.id_a == c1 else 1
        else:
            d = 0

        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append(d)

    repeated_pairs_count = 0
    consistent_pairs_count = 0
    inconsistent_pairs_count = 0

    for (v, pair), directions in pair_history.items():
        decisive_directions = [d for d in directions if d != 0]
        if len(decisive_directions) >= 2:
            repeated_pairs_count += 1
            if all(d == decisive_directions[0] for d in decisive_directions):
                consistent_pairs_count += 1
            else:
                inconsistent_pairs_count += 1

    total_evaluated = consistent_pairs_count + inconsistent_pairs_count
    rate = round(consistent_pairs_count / total_evaluated, 3) if total_evaluated > 0 else None

    return {
        "voter": voter or "all",
        "repeated_pairs_count": repeated_pairs_count,
        "consistent_pairs_count": consistent_pairs_count,
        "self_reversals_count": inconsistent_pairs_count,
        "consistency_rate": rate,
    }

