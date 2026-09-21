"""Agent trajectory parsing, validation, execution summaries, and step-level commentary."""

import json
from typing import Any, Dict, List, Optional, Union
from showdown.models import Candidate, StepAnnotation, StepAnnotationTag, TrajectoryStep, TrajectorySummary


def _first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_trajectory(content: Union[str, Dict[str, Any], List[Any]]) -> Optional[List[TrajectoryStep]]:
    """
    Parse content into a structured list of TrajectorySteps.
    Supports raw JSON strings, Python dicts with 'steps'/'trajectory', or lists of step objects.
    """
    if isinstance(content, str):
        content = content.strip()
        if not (content.startswith("{") or content.startswith("[")):
            return None
        try:
            data = json.loads(content)
        except Exception:
            return None
    else:
        data = content

    raw_steps = None
    if isinstance(data, list):
        raw_steps = data
    elif isinstance(data, dict):
        if "steps" in data and isinstance(data["steps"], list):
            raw_steps = data["steps"]
        elif "trajectory" in data and isinstance(data["trajectory"], list):
            raw_steps = data["trajectory"]
        elif "transcript" in data and isinstance(data["transcript"], list):
            raw_steps = data["transcript"]

    if raw_steps is None:
        return None

    steps: List[TrajectoryStep] = []
    for idx, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue

        raw_step_idx = _first_present(s, ["step_index", "index", "id"])
        try:
            step_idx = int(raw_step_idx) if raw_step_idx is not None else idx
        except (ValueError, TypeError):
            step_idx = idx

        thought = _first_present(s, ["thought", "reasoning", "thinking"])
        tool_name = _first_present(s, ["tool_name", "tool", "action"])
        tool_args = _first_present(s, ["tool_args", "args", "parameters", "input"])
        tool_output = _first_present(s, ["tool_output", "output", "result", "observation"])
        duration = _first_present(s, ["duration_seconds", "duration", "elapsed"])
        try:
            duration = float(duration) if duration is not None else None
        except (ValueError, TypeError):
            duration = None

        tokens = _first_present(s, ["tokens", "token_usage", "usage"])
        if tokens is not None and not isinstance(tokens, dict):
            try:
                tokens = {"total": int(tokens)}
            except (ValueError, TypeError):
                tokens = None

        raw_status = _first_present(s, ["status"])
        has_error_flag = bool(s.get("error"))
        if has_error_flag or (isinstance(raw_status, str) and raw_status.lower() in ("error", "failed", "failure")):
            status = "error"
        elif isinstance(raw_status, str) and raw_status.lower() in ("retry", "retrying"):
            status = "retry"
        else:
            status = "success"

        # Existing annotations
        annotations = []
        raw_ann = s.get("annotations", [])
        if isinstance(raw_ann, list):
            for a in raw_ann:
                if isinstance(a, dict) and "tag" in a:
                    try:
                        annotations.append(StepAnnotation(**a))
                    except Exception:
                        pass

        step = TrajectoryStep(
            step_index=step_idx,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=str(tool_output) if tool_output is not None else None,
            duration_seconds=duration,
            tokens=tokens,
            status=status,
            annotations=annotations,
        )
        steps.append(step)

    return steps


def compute_trajectory_summary(steps: List[TrajectoryStep]) -> TrajectorySummary:
    """Calculate summary execution statistics across all trajectory steps."""
    total_steps = len(steps)
    total_duration = 0.0
    total_tokens = 0
    total_tool_calls = 0
    error_count = 0

    for s in steps:
        if s.duration_seconds:
            total_duration += s.duration_seconds
        if s.tool_name:
            total_tool_calls += 1
        if s.status == "error":
            error_count += 1
        if s.tokens and isinstance(s.tokens, dict):
            if "total" in s.tokens and isinstance(s.tokens["total"], (int, float)):
                total_tokens += int(s.tokens["total"])
            else:
                prompt_tok = s.tokens.get("prompt", 0) or s.tokens.get("prompt_tokens", 0) or 0
                comp_tok = s.tokens.get("completion", 0) or s.tokens.get("completion_tokens", 0) or 0
                total_tokens += int(prompt_tok + comp_tok)

    return TrajectorySummary(
        total_steps=total_steps,
        total_duration_seconds=round(total_duration, 3),
        total_tokens=total_tokens,
        total_tool_calls=total_tool_calls,
        error_count=error_count,
    )


def extract_candidate_steps(candidate: Candidate) -> List[TrajectoryStep]:
    """Retrieve trajectory steps from candidate, syncing with any metadata annotations by voter."""
    steps = parse_trajectory(candidate.content) or []
    meta_ann = candidate.metadata.get("step_annotations", {})
    if meta_ann:
        for s in steps:
            key = str(s.step_index)
            if key in meta_ann:
                meta_voters: Dict[str, StepAnnotation] = {}
                for ann_dict in meta_ann[key]:
                    if isinstance(ann_dict, dict) and "tag" in ann_dict:
                        try:
                            ann_obj = StepAnnotation(**ann_dict)
                            meta_voters[ann_obj.voter or "human"] = ann_obj
                        except Exception:
                            pass
                filtered_embedded = [a for a in s.annotations if (a.voter or "human") not in meta_voters]
                s.annotations = filtered_embedded + list(meta_voters.values())
    return steps
