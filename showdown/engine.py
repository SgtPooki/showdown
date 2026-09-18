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

    # Pick first candidate from the least-evaluated group (with some randomness)
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
