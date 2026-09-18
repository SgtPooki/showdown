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
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float
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
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class CreateTournamentRequest(BaseModel):
    id: Optional[str] = None
    title: str
    prompt: Optional[str] = None
    task_type: TaskType = TaskType.TEXT
    candidates: List[Candidate]
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
    backend: Optional[str] = "auto"  # "auto", "claude", "codex", "openai"


class EvolveResponse(BaseModel):
    prompt_used: str
    new_candidates: List[Candidate]
    summary: str


class AcceptCandidateRequest(BaseModel):
    candidate_id: str
    notes: Optional[str] = None

