"""Elo rating calculation and active tournament pairing engine."""

import math
import random
from collections import defaultdict
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


def select_matchup(
    tournament: Tournament,
    mode: str = "active",
) -> Optional[Tuple[Candidate, Candidate]]:
    """
    Select an optimal next pair of candidates for comparison.
    Strategies:
    - 'active' (default): exploration + exploitation (fewest matches + similar Elo).
    - 'controversial': pairs with position-bias contradictions or evaluator disagreements.
    - 'close': candidates with the closest Elo ratings for decisive triage.
    """
    candidates = tournament.candidates
    if len(candidates) < 2:
        return None

    cand_map = {c.id: c for c in candidates}

    # Track match history counts between pairs
    pair_counts: Dict[Tuple[str, str], int] = {}
    controversial_pairs: List[Tuple[str, str]] = []

    for m in tournament.matches:
        pair = tuple(sorted([m.id_a, m.id_b]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if m.notes and ("contradiction" in m.notes.lower() or "position-bias" in m.notes.lower()):
            if pair not in controversial_pairs:
                controversial_pairs.append(pair)

    if mode == "controversial" and controversial_pairs:
        # Pick the most controversial pair with fewest total evaluations
        controversial_pairs.sort(key=lambda p: pair_counts.get(p, 0))
        c_a_id, c_b_id = controversial_pairs[0]
        if c_a_id in cand_map and c_b_id in cand_map:
            cand_a, cand_b = cand_map[c_a_id], cand_map[c_b_id]
            if random.random() > 0.5:
                return cand_a, cand_b
            return cand_b, cand_a

    if mode in ("close", "controversial"):
        # Find closest Elo pair among candidates
        all_pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1 = candidates[i]
                c2 = candidates[j]
                s1 = tournament.stats.get(c1.id, CandidateStats())
                s2 = tournament.stats.get(c2.id, CandidateStats())
                elo_diff = abs(s1.elo - s2.elo)
                p_key = tuple(sorted([c1.id, c2.id]))
                p_count = pair_counts.get(p_key, 0)
                all_pairs.append((elo_diff + (p_count * 50.0), c1, c2))

        all_pairs.sort(key=lambda item: item[0])
        if all_pairs:
            _, c_a, c_b = all_pairs[0]
            if random.random() > 0.5:
                return c_a, c_b
            return c_b, c_a

    if mode == "info_gain":
        # Bradley-Terry active information-gain matchmaking.
        # Selects candidate pairs that maximize expected variance reduction,
        # weighted toward top-performing candidates to resolve the winner faster.
        best_gain = -1e9
        best_pair = None

        mean_elo = sum(tournament.stats.get(c.id, CandidateStats()).elo for c in candidates) / len(candidates)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1 = candidates[i]
                c2 = candidates[j]
                s1 = tournament.stats.get(c1.id, CandidateStats())
                s2 = tournament.stats.get(c2.id, CandidateStats())

                prior_se = (400.0 / math.log(10.0)) * 2.0
                u1 = s1.bt_uncertainty if s1.bt_uncertainty is not None else prior_se / math.sqrt(max(1, s1.matches + 1))
                u2 = s2.bt_uncertainty if s2.bt_uncertainty is not None else prior_se / math.sqrt(max(1, s2.matches + 1))
                joint_uncertainty = math.sqrt(u1 * u1 + u2 * u2)

                r1 = s1.bt_elo if s1.bt_elo is not None else s1.elo
                r2 = s2.bt_elo if s2.bt_elo is not None else s2.elo
                diff = abs(r1 - r2)
                closeness_factor = 1.0 / (1.0 + (diff / 200.0))

                avg_rating = (r1 + r2) / 2.0
                relevance = 1.0 + max(0.0, (avg_rating - mean_elo) / 200.0)

                pair_key = tuple(sorted([c1.id, c2.id]))
                prior = pair_counts.get(pair_key, 0)
                repetition_penalty = prior * 30.0

                gain = (joint_uncertainty * closeness_factor * relevance) - repetition_penalty
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (c1, c2)

        if best_pair:
            c_a, c_b = best_pair
            if random.random() > 0.5:
                return c_a, c_b
            return c_b, c_a

    # Standard active selection
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


def _invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Invert an n x n symmetric positive-definite matrix via Gauss-Jordan elimination with partial pivoting."""
    n = len(matrix)
    if n == 0:
        return []
    if n == 1:
        val = matrix[0][0]
        return [[1.0 / max(1e-9, val)]]

    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for i in range(n):
        pivot_row = i
        max_val = abs(aug[i][i])
        for r in range(i + 1, n):
            if abs(aug[r][i]) > max_val:
                max_val = abs(aug[r][i])
                pivot_row = r

        if pivot_row != i:
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        pivot = aug[i][i]
        if abs(pivot) < 1e-9:
            pivot = 1e-9 if pivot >= 0 else -1e-9

        for c in range(2 * n):
            aug[i][c] /= pivot

        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for c in range(2 * n):
                    aug[r][c] -= factor * aug[i][c]

    return [row[n:] for row in aug]


def fit_bradley_terry(
    candidates: List[Candidate],
    matches: List[Match],
    prior_weight: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    """
    Fit Bradley-Terry latent strength parameters via Minorization-Maximization (MM)
    with Laplace/Bayesian regularization to eliminate presentation order bias
    and produce calibrated standard errors (confidence bands).
    """
    n = len(candidates)
    if n == 0:
        return {}
    if n == 1:
        return {
            candidates[0].id: {
                "bt_elo": DEFAULT_ELO,
                "bt_uncertainty": 0.0,
                "bt_ci_lower": DEFAULT_ELO,
                "bt_ci_upper": DEFAULT_ELO,
            }
        }

    cand_indices = {c.id: i for i, c in enumerate(candidates)}

    # Accumulate wins W_i and match counts N_ij
    w = [0.0] * n
    n_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for m in matches:
        if m.id_a not in cand_indices or m.id_b not in cand_indices:
            continue
        i = cand_indices[m.id_a]
        j = cand_indices[m.id_b]
        if i == j:
            continue

        if m.winner == "a":
            w[i] += 1.0
            n_matrix[i][j] += 1
            n_matrix[j][i] += 1
        elif m.winner == "b":
            w[j] += 1.0
            n_matrix[i][j] += 1
            n_matrix[j][i] += 1
        elif m.winner == "tie":
            w[i] += 0.5
            w[j] += 0.5
            n_matrix[i][j] += 1
            n_matrix[j][i] += 1

    # Minorization-Maximization (MM) algorithm for Bradley-Terry
    p = [1.0] * n

    for _ in range(max_iter):
        p_next = [0.0] * n
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and n_matrix[i][j] > 0:
                    denom += n_matrix[i][j] / (p[i] + p[j])
            denom += prior_weight / (p[i] + 1.0)
            p_next[i] = (w[i] + (0.5 * prior_weight)) / denom

        max_diff = max(abs(p_next[i] - p[i]) for i in range(n))
        p = p_next
        if max_diff < tol:
            break

    scale = 400.0 / math.log(10.0)
    beta = [math.log(max(1e-12, val)) for val in p]

    # Compute Fisher Information Matrix for standard error estimation
    info_matrix = [[0.0] * n for _ in range(n)]
    ridge_lambda = 1e-4

    for i in range(n):
        diag = ridge_lambda + (prior_weight * p[i] / math.pow(p[i] + 1.0, 2))
        for j in range(n):
            if i != j:
                if n_matrix[i][j] > 0:
                    val = n_matrix[i][j] * (p[i] * p[j]) / math.pow(p[i] + p[j], 2)
                    diag += val
                    info_matrix[i][j] = -val
                else:
                    info_matrix[i][j] = 0.0
        info_matrix[i][i] = diag

    cov_matrix = _invert_matrix(info_matrix)

    out: Dict[str, Dict[str, float]] = {}
    for c in candidates:
        idx = cand_indices[c.id]
        bt_elo = round(DEFAULT_ELO + (beta[idx] * scale), 2)
        var_beta = max(1e-4, cov_matrix[idx][idx])
        se_elo = round(math.sqrt(var_beta) * scale, 2)
        ci_lower = round(bt_elo - (1.96 * se_elo), 2)
        ci_upper = round(bt_elo + (1.96 * se_elo), 2)

        out[c.id] = {
            "bt_elo": bt_elo,
            "bt_uncertainty": se_elo,
            "bt_ci_lower": ci_lower,
            "bt_ci_upper": ci_upper,
        }

    return out


def apply_bradley_terry_stats(
    candidates: List[Candidate],
    matches: List[Match],
    stats: Dict[str, CandidateStats],
) -> None:
    """Populate Bradley-Terry MLE parameters and calibrated uncertainty intervals into stats."""
    if matches:
        bt_results = fit_bradley_terry(candidates, matches)
        for cid, b_data in bt_results.items():
            if cid in stats:
                stats[cid].bt_elo = b_data["bt_elo"]
                stats[cid].bt_uncertainty = b_data["bt_uncertainty"]
                stats[cid].bt_ci_lower = b_data["bt_ci_lower"]
                stats[cid].bt_ci_upper = b_data["bt_ci_upper"]
    else:
        scale = 400.0 / math.log(10.0)
        prior_se = round(scale * 2.0, 2)
        for cid in stats:
            stats[cid].bt_elo = DEFAULT_ELO
            stats[cid].bt_uncertainty = prior_se
            stats[cid].bt_ci_lower = round(DEFAULT_ELO - (1.96 * prior_se), 2)
            stats[cid].bt_ci_upper = round(DEFAULT_ELO + (1.96 * prior_se), 2)


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
        elif m.winner == "both_bad":
            stat_a.losses += 1
            stat_b.losses += 1
            stat_a.matches += 1
            stat_b.matches += 1
            penalty = round(k / 2.0, 2)
            elo_a_after = max(100.0, round(elo_a_before - penalty, 2))
            elo_b_after = max(100.0, round(elo_b_before - penalty, 2))
        else:
            # e.g. 'abstain' or unrecognized outcome — do not shift ratings
            continue

        stat_a.elo = elo_a_after
        stat_b.elo = elo_b_after

    # Compute Bradley-Terry MLE latent parameters & uncertainty intervals
    apply_bradley_terry_stats(candidates, filtered_matches, stats)

    return stats


def check_convergence(
    candidates: List[Candidate],
    stats: Dict[str, CandidateStats],
    min_matches_per_cand: int = 2,
    lead_margin: float = 35.0,
    z_threshold: float = 1.645,
) -> Tuple[bool, float]:
    """
    Check if a tournament's top candidate is statistically separated.
    Uses Bayesian Bradley-Terry confidence intervals (z-score >= 1.645 for 95% one-sided confidence)
    or standard Elo point margin separation.
    Returns (is_converged, confidence_score_0_to_1).
    """
    if len(candidates) < 2:
        return True, 1.0

    has_bt = any(stats.get(c.id, CandidateStats()).bt_elo is not None for c in candidates)
    if has_bt:
        sorted_stats = sorted(
            [stats.get(c.id, CandidateStats()) for c in candidates],
            key=lambda s: s.bt_elo if s.bt_elo is not None else s.elo,
            reverse=True,
        )
    else:
        sorted_stats = sorted([stats.get(c.id, CandidateStats()) for c in candidates], key=lambda s: s.elo, reverse=True)

    min_played = min(s.matches for s in sorted_stats)
    if min_played < min_matches_per_cand:
        progress = min(1.0, sum(s.matches for s in sorted_stats) / max(1, len(candidates) * min_matches_per_cand))
        return False, round(progress * 0.5, 2)

    top = sorted_stats[0]
    second = sorted_stats[1]

    # If Bradley-Terry ratings and uncertainties are available, calculate exact z-score separation
    if (
        top.bt_elo is not None
        and second.bt_elo is not None
        and top.bt_uncertainty is not None
        and second.bt_uncertainty is not None
    ):
        diff = top.bt_elo - second.bt_elo
        pooled_se = math.sqrt(math.pow(top.bt_uncertainty, 2) + math.pow(second.bt_uncertainty, 2))
        z = diff / max(1e-4, pooled_se)
        # Normal CDF: 0.5 * (1 + erf(z / sqrt(2)))
        confidence = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        confidence = min(1.0, max(0.0, confidence))
        is_converged = z >= z_threshold and min_played >= min_matches_per_cand
        return is_converged, round(confidence, 2)

    top_elo = top.elo
    second_elo = second.elo
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

    # Map: voter -> { (c1, c2): [directions] } where c1 < c2
    voter_pair_history: Dict[str, Dict[Tuple[str, str], List[int]]] = {v: defaultdict(list) for v in voters}

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

        voter_pair_history[m.voter][pair].append(direction)

    voter_decisions: Dict[str, Dict[Tuple[str, str], int]] = {v: {} for v in voters}
    for v in voters:
        for pair, dirs in voter_pair_history[v].items():
            s = sum(dirs)
            voter_decisions[v][pair] = 1 if s > 0 else (-1 if s < 0 else 0)

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
            agreement_rate = round(agreed / decisive, 3) if decisive > 0 else None
            reversal_rate = round(reversals / decisive, 3) if decisive > 0 else None

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

    # Calculate concordance specifically between human voters and synthetic LLM judges
    human_judge_agreements = 0
    human_judge_decisive = 0
    for p in evaluator_pairs:
        v1_is_judge = str(p["evaluator_a"]).startswith("judge:")
        v2_is_judge = str(p["evaluator_b"]).startswith("judge:")
        if v1_is_judge != v2_is_judge:
            human_judge_agreements += p["agreements"]
            human_judge_decisive += p["decisive_pairs_count"]

    human_vs_judge = None
    if human_judge_decisive > 0:
        human_vs_judge = {
            "shared_pairs_evaluated": human_judge_decisive,
            "agreements": human_judge_agreements,
            "reversals": human_judge_decisive - human_judge_agreements,
            "agreement_rate": round(human_judge_agreements / human_judge_decisive, 3),
        }

    return {
        "voters": voters,
        "total_matches": len(matches),
        "shared_pairs_evaluated": total_decisive,
        "overall_agreement_rate": overall_rate,
        "human_vs_judge": human_vs_judge,
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

