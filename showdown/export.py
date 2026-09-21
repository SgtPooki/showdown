"""Dataset export adapters, multi-tournament aggregation, and train/val splitting."""

import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from showdown.models import Candidate, CandidateStats, Match, Tournament, TriageStatus


def find_tournament_lineages(tournaments: List[Tournament]) -> Dict[str, str]:
    """
    Partition tournaments into connected lineage clusters based on parent_ids.
    Returns a mapping from tournament_id to cluster_root_id.
    """
    # Build undirected adjacency graph between tournaments and their parents
    adj: Dict[str, Set[str]] = defaultdict(set)
    all_ids = {t.id for t in tournaments}

    for t in tournaments:
        for p in t.parent_ids:
            if p in all_ids:
                adj[t.id].add(p)
                adj[p].add(t.id)

    lineage_map: Dict[str, str] = {}
    visited: Set[str] = set()

    for t in tournaments:
        if t.id in visited:
            continue
        # BFS/DFS to discover full connected lineage
        cluster = []
        queue = [t.id]
        visited.add(t.id)
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        root_id = sorted(cluster)[0]
        for tid in cluster:
            lineage_map[tid] = root_id

    return lineage_map


def export_dataset(
    tournaments: List[Tournament],
    format: str = "dpo",
    task_type: Optional[str] = None,
    voter: Optional[str] = None,
    consensus: Optional[str] = None,
    min_agreement: Optional[float] = None,
    dedup: bool = True,
    include_critique: bool = True,
    split: Optional[float] = None,
    split_by: str = "lineage",
    seed: int = 42,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Aggregate preference dataset across tournaments for DPO, KTO, or pairwise reward modeling.

    Args:
        tournaments: List of tournaments to aggregate from.
        format: Export format: 'dpo', 'kto', or 'pairwise_margins'.
        task_type: Filter by task_type (e.g. 'code', 'text', 'svg', 'markdown').
        voter: Filter matches by evaluator tag.
        consensus: Multi-annotator consensus requirement ('strict' or 'majority').
        min_agreement: Minimum inter-rater agreement threshold (0.5 to 1.0).
        dedup: Whether to deduplicate identical prompt/chosen/rejected pairs.
        include_critique: Whether to include evaluator notes/critiques in records.
        split: Train/validation split ratio (e.g. 0.8 for 80% train / 20% val).
        split_by: Split strategy: 'lineage' (cluster by prompt/tournament lineage) or 'random'.
        seed: Random seed for deterministic train/val partition.

    Returns:
        List of records if split is None, or dict with 'train', 'val', and 'split_stats' if split is set.
    """
    if consensus is not None and consensus not in ("strict", "majority"):
        raise ValueError(f"Invalid consensus mode '{consensus}'. Must be 'strict' or 'majority'.")
    if min_agreement is not None and not (0.0 < min_agreement <= 1.0):
        raise ValueError("min_agreement must be between 0.0 and 1.0.")
    if (consensus or min_agreement is not None) and voter and voter.strip().lower() not in ("all", "pooled", "*"):
        raise ValueError("Cannot combine 'voter' filter with multi-annotator 'consensus' or 'min_agreement'.")
    if format not in ("dpo", "kto", "pairwise_margins", "raw", "spo", "prm"):
        raise ValueError(f"Invalid export format '{format}'. Must be 'dpo', 'kto', 'pairwise_margins', 'raw', 'spo', or 'prm'.")

    # Filter tournaments by task_type if requested
    if task_type:
        t_filter = task_type.strip().lower()
        tournaments = [t for t in tournaments if t.task_type.value.lower() == t_filter]

    records_by_tournament: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for t in tournaments:
        cand_map = {c.id: c for c in t.candidates}
        prompt = t.prompt or t.title

        if format in ("dpo", "pairwise_margins"):
            if consensus or min_agreement is not None:
                # Group matches by canonical candidate pair (c1, c2) where c1 < c2
                canonical_groups: Dict[Tuple[str, str], List[Match]] = defaultdict(list)
                for m in t.matches:
                    if m.winner in ("a", "b"):
                        c1, c2 = (m.id_a, m.id_b) if m.id_a < m.id_b else (m.id_b, m.id_a)
                        canonical_groups[(c1, c2)].append(m)

                for (c1, c2), pair_matches in canonical_groups.items():
                    votes_c1 = sum(1 for m in pair_matches if (m.winner == "a" and m.id_a == c1) or (m.winner == "b" and m.id_b == c1))
                    votes_c2 = sum(1 for m in pair_matches if (m.winner == "a" and m.id_a == c2) or (m.winner == "b" and m.id_b == c2))
                    total_decisive = votes_c1 + votes_c2
                    if total_decisive < 2 or votes_c1 == votes_c2:
                        continue

                    voters_for_pair = sorted(list({m.voter for m in pair_matches if m.voter}))
                    if consensus == "strict" and (len(voters_for_pair) < 2 or (votes_c1 > 0 and votes_c2 > 0)):
                        continue

                    agreement_rate = max(votes_c1, votes_c2) / total_decisive
                    if min_agreement is not None and agreement_rate < min_agreement:
                        continue
                    if consensus == "majority" and agreement_rate <= 0.5:
                        continue

                    chosen_id = c1 if votes_c1 > votes_c2 else c2
                    rejected_id = c2 if votes_c1 > votes_c2 else c1
                    chosen_cand = cand_map.get(chosen_id)
                    rejected_cand = cand_map.get(rejected_id)

                    if chosen_cand and rejected_cand:
                        critiques = [m.notes for m in pair_matches if m.notes]
                        rec: Dict[str, Any] = {
                            "prompt": prompt,
                            "chosen": chosen_cand.content,
                            "rejected": rejected_cand.content,
                            "chosen_id": chosen_id,
                            "rejected_id": rejected_id,
                            "agreement_rate": round(agreement_rate, 3),
                            "votes_chosen": max(votes_c1, votes_c2),
                            "votes_rejected": min(votes_c1, votes_c2),
                            "evaluators": voters_for_pair,
                            "tournament_id": t.id,
                            "task_type": t.task_type.value,
                        }
                        if include_critique and critiques:
                            rec["critique"] = critiques[0] if len(critiques) == 1 else " | ".join(critiques)

                        if format == "pairwise_margins":
                            s_chosen = t.stats.get(chosen_id, CandidateStats())
                            s_rejected = t.stats.get(rejected_id, CandidateStats())
                            rating_chosen = s_chosen.bt_elo if s_chosen.bt_elo is not None else s_chosen.elo
                            rating_rejected = s_rejected.bt_elo if s_rejected.bt_elo is not None else s_rejected.elo
                            rec["margin"] = round(rating_chosen - rating_rejected, 2)

                        records_by_tournament[t.id].append(rec)

            else:
                filtered_matches = t.matches
                if voter and voter.strip().lower() not in ("all", "pooled", "*"):
                    filtered_matches = [m for m in t.matches if m.voter == voter]

                for m in filtered_matches:
                    if m.winner in ("a", "b"):
                        chosen_id = m.id_a if m.winner == "a" else m.id_b
                        rejected_id = m.id_b if m.winner == "a" else m.id_a
                        chosen_cand = cand_map.get(chosen_id)
                        rejected_cand = cand_map.get(rejected_id)

                        if chosen_cand and rejected_cand:
                            rec = {
                                "prompt": prompt,
                                "chosen": chosen_cand.content,
                                "rejected": rejected_cand.content,
                                "chosen_id": chosen_id,
                                "rejected_id": rejected_id,
                                "voter": m.voter,
                                "timestamp": m.timestamp,
                                "tournament_id": t.id,
                                "task_type": t.task_type.value,
                            }
                            if include_critique and m.notes:
                                rec["critique"] = m.notes

                            if format == "pairwise_margins":
                                s_chosen = t.stats.get(chosen_id, CandidateStats())
                                s_rejected = t.stats.get(rejected_id, CandidateStats())
                                rating_chosen = s_chosen.bt_elo if s_chosen.bt_elo is not None else s_chosen.elo
                                rating_rejected = s_rejected.bt_elo if s_rejected.bt_elo is not None else s_rejected.elo
                                rec["margin"] = round(rating_chosen - rating_rejected, 2)

                            records_by_tournament[t.id].append(rec)

        elif format == "kto":
            for cid, t_rec in t.triage.items():
                cand = cand_map.get(cid)
                if not cand:
                    continue
                if t_rec.status in (TriageStatus.LIKED, TriageStatus.FAVORITE):
                    rec = {
                        "prompt": prompt,
                        "completion": cand.content,
                        "label": True,
                        "candidate_id": cid,
                        "status": t_rec.status.value,
                        "tournament_id": t.id,
                        "task_type": t.task_type.value,
                    }
                    if include_critique and t_rec.notes:
                        rec["critique"] = t_rec.notes
                    records_by_tournament[t.id].append(rec)
                elif t_rec.status == TriageStatus.DISLIKED:
                    rec = {
                        "prompt": prompt,
                        "completion": cand.content,
                        "label": False,
                        "candidate_id": cid,
                        "status": t_rec.status.value,
                        "tournament_id": t.id,
                        "task_type": t.task_type.value,
                    }
                    if include_critique and t_rec.notes:
                        rec["critique"] = t_rec.notes
                    records_by_tournament[t.id].append(rec)

        elif format in ("spo", "prm"):
            from showdown.trajectories import extract_candidate_steps
            for cand in t.candidates:
                steps = extract_candidate_steps(cand)
                for step in steps:
                    for ann in step.annotations:
                        if voter and voter.strip().lower() not in ("all", "pooled", "*"):
                            if ann.voter != voter:
                                continue
                        tag_val = ann.tag.value if hasattr(ann.tag, "value") else str(ann.tag)
                        score = 1.0 if tag_val == "exemplary" else (0.0 if tag_val == "incorrect" else 0.5)
                        rec = {
                            "prompt": prompt,
                            "candidate_id": cand.id,
                            "step_index": step.step_index,
                            "thought": step.thought,
                            "tool_name": step.tool_name,
                            "tool_args": step.tool_args,
                            "tool_output": step.tool_output,
                            "duration_seconds": step.duration_seconds,
                            "tag": tag_val,
                            "label": score,
                            "notes": ann.notes,
                            "voter": ann.voter,
                            "tournament_id": t.id,
                            "task_type": t.task_type.value,
                        }
                        records_by_tournament[t.id].append(rec)

        elif format == "raw":
            for m in t.matches:
                records_by_tournament[t.id].append(m.model_dump())

    # Deduplication pass if requested
    if dedup:
        for tid in records_by_tournament:
            seen_keys: Set[Any] = set()
            deduped = []
            for rec in records_by_tournament[tid]:
                if format in ("dpo", "pairwise_margins"):
                    key = (rec["prompt"].strip(), rec["chosen"].strip(), rec["rejected"].strip())
                elif format == "kto":
                    key = (rec["prompt"].strip(), rec["completion"].strip(), rec["label"])
                elif format in ("spo", "prm"):
                    key = (rec["prompt"].strip(), rec["candidate_id"], rec["step_index"], rec.get("voter"), rec["tag"])
                else:
                    key = (rec.get("id_a"), rec.get("id_b"), rec.get("timestamp"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(rec)
            records_by_tournament[tid] = deduped

    # If no split requested, flatten all records
    if split is None:
        flat_records = []
        for tid in sorted(records_by_tournament.keys()):
            flat_records.extend(records_by_tournament[tid])
        return flat_records

    # Automated Train/Validation Split
    if not (0.0 < split < 1.0):
        raise ValueError(f"split ratio must be between 0.0 and 1.0, got {split}")

    rng = random.Random(seed)

    if split_by == "lineage":
        # Group tournaments by connected parent/child lineage
        lineage_map = find_tournament_lineages(tournaments)
        lineage_clusters: Dict[str, List[str]] = defaultdict(list)
        for t in tournaments:
            root = lineage_map.get(t.id, t.id)
            lineage_clusters[root].append(t.id)

        cluster_roots = sorted(list(lineage_clusters.keys()))
        rng.shuffle(cluster_roots)

        # Count total records
        total_records = sum(len(records_by_tournament[tid]) for tid in records_by_tournament)
        target_train = int(round(total_records * split))

        train_records = []
        val_records = []
        train_lineages = 0
        val_lineages = 0
        # Collect non-empty clusters
        clusters_with_data = []
        for root in cluster_roots:
            tids = lineage_clusters[root]
            cluster_recs = []
            for tid in tids:
                cluster_recs.extend(records_by_tournament[tid])
            if cluster_recs:
                clusters_with_data.append((root, cluster_recs))

        curr_train_count = 0
        for i, (root, cluster_recs) in enumerate(clusters_with_data):
            remaining_clusters = len(clusters_with_data) - i
            # If multiple clusters exist, reserve at least one non-empty cluster for val
            if remaining_clusters == 1 and not val_records and train_records:
                val_records.extend(cluster_recs)
                val_lineages += 1
            elif curr_train_count < target_train or not train_records:
                train_records.extend(cluster_recs)
                curr_train_count += len(cluster_recs)
                train_lineages += 1
            else:
                val_records.extend(cluster_recs)
                val_lineages += 1

        return {
            "train": train_records,
            "val": val_records,
            "split_stats": {
                "split_by": "lineage",
                "train_count": len(train_records),
                "val_count": len(val_records),
                "train_ratio": round(len(train_records) / max(1, len(train_records) + len(val_records)), 3),
                "train_lineages": train_lineages,
                "val_lineages": val_lineages,
            },
        }

    else:  # random split
        flat_records = []
        for tid in sorted(records_by_tournament.keys()):
            flat_records.extend(records_by_tournament[tid])
        rng.shuffle(flat_records)

        cutoff = int(round(len(flat_records) * split))
        train_records = flat_records[:cutoff]
        val_records = flat_records[cutoff:]

        return {
            "train": train_records,
            "val": val_records,
            "split_stats": {
                "split_by": "random",
                "train_count": len(train_records),
                "val_count": len(val_records),
                "train_ratio": round(len(train_records) / max(1, len(flat_records)), 3),
            },
        }


def format_jsonl(records: List[Dict[str, Any]]) -> str:
    """Serialize a list of dictionaries into a newline-delimited JSON string."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
