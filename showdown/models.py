"""Data models for Showdown tournaments, candidates, and matches."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class TaskType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    DIFF = "diff"
    IMAGE = "image"
    JSON = "json"
    SVG = "svg"
    TRAJECTORY = "trajectory"


class StepAnnotationTag(str, Enum):
    EXEMPLARY = "exemplary"
    INEFFICIENT = "inefficient"
    INCORRECT = "incorrect"
    NEUTRAL = "neutral"


class StepAnnotation(BaseModel):
    step_index: int
    tag: StepAnnotationTag
    notes: Optional[str] = None
    voter: Optional[str] = "human"
    timestamp: float = Field(default_factory=time.time)


class TrajectoryStep(BaseModel):
    step_index: int
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_output: Optional[str] = None
    duration_seconds: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    status: Optional[str] = "success"  # "success", "error", "retry"
    annotations: List[StepAnnotation] = Field(default_factory=list)


class TrajectorySummary(BaseModel):
    total_steps: int = 0
    total_duration_seconds: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    error_count: int = 0


class StepAnnotationRequest(BaseModel):
    tag: StepAnnotationTag
    notes: Optional[str] = None
    voter: Optional[str] = "human"


class TournamentStatus(str, Enum):
    ACTIVE = "active"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"


class Candidate(BaseModel):
    id: str
    label: Optional[str] = None
    content: str  # text, markdown, code snippet, image URL / base64 / path, or JSON string
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generation: int = 1


class Match(BaseModel):
    id_a: str
    id_b: str
    winner: str  # 'a', 'b', 'tie', 'both_bad'
    timestamp: float = Field(default_factory=time.time)
    elo_a_before: float = 1200.0
    elo_b_before: float = 1200.0
    elo_a_after: float = 1200.0
    elo_b_after: float = 1200.0
    voter: Optional[str] = "human"
    notes: Optional[str] = None


class TriageStatus(str, Enum):
    LIKED = "liked"
    DISLIKED = "disliked"
    FAVORITE = "favorite"
    NEUTRAL = "neutral"


class TriageRecord(BaseModel):
    status: TriageStatus
    notes: Optional[str] = None
    updated_at: float = Field(default_factory=time.time)


class CandidateStats(BaseModel):
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    matches: int = 0
    bt_elo: Optional[float] = None
    bt_uncertainty: Optional[float] = None
    bt_ci_lower: Optional[float] = None
    bt_ci_upper: Optional[float] = None


class Tournament(BaseModel):
    id: str
    title: str
    prompt: Optional[str] = None
    task_type: TaskType = TaskType.TEXT
    candidates: List[Candidate] = Field(default_factory=list)
    stats: Dict[str, CandidateStats] = Field(default_factory=dict)
    triage: Dict[str, TriageRecord] = Field(default_factory=dict)
    matches: List[Match] = Field(default_factory=list)
    status: TournamentStatus = TournamentStatus.ACTIVE
    accepted_candidate_id: Optional[str] = None
    parent_ids: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    chain_mode: Optional[str] = "growth"  # "growth", "divergence"
    voters: List[str] = Field(default_factory=list)
    blinded: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class CreateTournamentRequest(BaseModel):
    id: Optional[str] = None
    title: str
    prompt: Optional[str] = None
    task_type: TaskType = TaskType.TEXT
    candidates: List[Candidate]
    parent_ids: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    chain_mode: Optional[str] = "growth"
    blinded: bool = False
    overwrite: bool = False


class VoteRequest(BaseModel):
    id_a: str
    id_b: str
    winner: str  # 'a', 'b', 'tie', 'both_bad'
    notes: Optional[str] = None
    voter: Optional[str] = "human"


class TriageRequest(BaseModel):
    candidate_id: str
    status: TriageStatus
    notes: Optional[str] = None


class AddCandidatesRequest(BaseModel):
    candidates: List[Candidate]


class EvolveRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=20)
    instructions: Optional[str] = None
    backend: Optional[str] = "auto"  # "auto", "claude", "codex", "cursor", "omp", "openai", "vllm"
    providers: Optional[List[str]] = None  # Multi-agent fan-out list of provider IDs
    mode: str = "refine"  # "refine", "diverge", "hybrid"
    wildcards: Optional[int] = None  # Number of exploration wildcards if mode is hybrid
    chain_mode: Optional[str] = None  # Optional override ("growth", "divergence")


class EvolveResponse(BaseModel):
    prompt_used: str
    new_candidates: List[Candidate]
    summary: str
    mode: str = "refine"
    refine_count: int = 0
    wildcard_count: int = 0
    providers_used: List[str] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    id: str
    display_name: str
    provider_type: str
    model: Optional[str] = None
    available: bool


class ProviderLeaderboardEntry(BaseModel):
    provider_id: str
    display_name: str
    elo: float
    wins: int
    losses: int
    ties: int
    matches: int
    win_rate: float
    accepted_winners: int
    candidates_count: int


class SimulationRequest(BaseModel):
    scenario: str = "generational"  # "generational", "logo", "chained"
    generations: int = Field(default=3, ge=1, le=10)
    candidates_per_gen: int = Field(default=4, ge=2, le=20)
    judge_backend: str = "auto"
    generation_backend: str = "auto"
    generation_providers: Optional[List[str]] = None
    evolution_mode: str = "hybrid"
    wildcards: Optional[int] = 1
    prompt: Optional[str] = None
    company: Optional[str] = "Apex Systems"


class AcceptCandidateRequest(BaseModel):
    candidate_id: str
    notes: Optional[str] = None


class UpdateTournamentRequest(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None
    blinded: Optional[bool] = None
    context: Optional[str] = None


class JudgeRequest(BaseModel):
    backend: Optional[str] = "auto"
    rounds: int = Field(default=5, ge=1, le=20)
    rubric: Optional[str] = None
    swap_positions: bool = True
    voter: Optional[str] = None
    mode: str = "active"  # "active", "controversial", "close"
    stop_on_convergence: bool = False


class JudgeMatchResult(BaseModel):
    id_a: str
    id_b: str
    winner: str
    critique: Optional[str] = None
    swapped_consistent: bool = True
    elo_a_after: float
    elo_b_after: float


class JudgeResponse(BaseModel):
    tournament_id: str
    backend: str
    voter: str
    rounds_requested: int
    matches_evaluated: int
    consistent_matches: int
    contradictions: int
    converged: bool
    confidence: float
    results: List[JudgeMatchResult] = Field(default_factory=list)

