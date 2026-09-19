"""Elo rating calculation and active tournament pairing engine."""

import math
import random
from typing import Dict, List, Optional, Tuple
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
