"""
schemas.py — Pydantic request/response models for the BharatRank API
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class RankRequest(BaseModel):
    jd_id: str = Field(..., description="Job Description ID, e.g. 'JD001'")
    top_k: int = Field(10, ge=1, le=50, description="Number of candidates to return")


class SignalBreakdown(BaseModel):
    semantic_fit:    float
    career_velocity: float
    skill_recency:   float
    signal_strength: float
    role_alignment:  float


class CandidateResult(BaseModel):
    rank:             int
    candidate_id:     str
    candidate_name:   str
    location:         str
    institution:      str
    years_experience: float
    current_role:     str
    bharatscore:      float = Field(..., alias="adjusted_score")
    signals:          SignalBreakdown
    top_skills:       List[str]
    certifications:   List[str]
    hackathon_wins:   int
    is_hidden_gem:    bool
    hidden_gem_reason: Optional[str]
    fairness_boost:   float
    match_report:     str

    model_config = {"populate_by_name": True}


class RankResponse(BaseModel):
    jd_id:        str
    jd_title:     str
    total_ranked: int
    ranked_at:    str
    results:      List[Dict[str, Any]]   # flexible for raw pipeline output


class HealthResponse(BaseModel):
    status:        str
    version:       str
    model:         str
    candidates_loaded: int
    jds_loaded:    int
    index_ready:   bool


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
