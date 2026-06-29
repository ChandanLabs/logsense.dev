"""
tests/test_scorer.py — Unit tests for MultiSignalScorer
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.scorer import MultiSignalScorer

scorer = MultiSignalScorer()

# ─── Fixtures ────────────────────────────────────────────────────────────────

FRESHER = {
    "id": "T001",
    "years_total_experience": 0,
    "experience": [],
    "skills": [{"name": "Python", "years": 1, "acquired_year": 2025}],
    "certifications": [{"name": "Kaggle ML", "year": 2025}],
    "github_commits_last_year": 150,
    "open_source_contributions": 5,
    "hackathon_wins": 1,
}

SENIOR = {
    "id": "T002",
    "years_total_experience": 5,
    "experience": [
        {"company": "A", "role": "ML Engineer", "is_current": True},
        {"company": "B", "role": "Data Scientist", "is_current": False},
        {"company": "C", "role": "Analyst", "is_current": False},
    ],
    "skills": [
        {"name": "Python",    "years": 5, "acquired_year": 2020},
        {"name": "PyTorch",   "years": 3, "acquired_year": 2022},
        {"name": "LangChain", "years": 1, "acquired_year": 2025},
    ],
    "certifications": [
        {"name": "AWS ML Specialty", "year": 2022},
        {"name": "Databricks Associate", "year": 2024},
    ],
    "github_commits_last_year": 350,
    "open_source_contributions": 12,
    "hackathon_wins": 2,
}

JD = {
    "id": "JD_TEST",
    "role_title": "ML Engineer",
    "required_skills": ["Python", "PyTorch", "FastAPI"],
    "nice_to_have_skills": ["LangChain", "Docker"],
}


# ─── career_velocity ─────────────────────────────────────────────────────────

def test_career_velocity_fresher():
    v = scorer.career_velocity(FRESHER)
    assert v == 0.50, "Fresher should return neutral 0.50"

def test_career_velocity_senior():
    v = scorer.career_velocity(SENIOR)
    assert 0.0 < v <= 1.0
    assert v > 0.50, "Senior with 3 roles in 5 years should be above neutral"

def test_career_velocity_bounds():
    extreme = {**SENIOR, "years_total_experience": 1, "experience": [{"role": f"R{i}"} for i in range(10)]}
    v = scorer.career_velocity(extreme)
    assert v <= 1.0, "Should be capped at 1.0"


# ─── skill_recency ────────────────────────────────────────────────────────────

def test_skill_recency_recent():
    v = scorer.skill_recency(FRESHER)
    assert 0.80 <= v <= 1.0, f"Expected high recency, got {v}"

def test_skill_recency_old_skills():
    old = {
        "skills": [{"name": "COBOL", "years": 10, "acquired_year": 2010}],
        "certifications": [],
    }
    v = scorer.skill_recency(old)
    assert v <= 0.42, f"Old skill should have low recency, got {v}"

def test_skill_recency_no_skills():
    v = scorer.skill_recency({"skills": [], "certifications": []})
    assert v == 0.40


# ─── signal_strength ──────────────────────────────────────────────────────────

def test_signal_strength_high():
    v = scorer.signal_strength(SENIOR)
    assert v >= 0.60, f"Expected high signal strength, got {v}"

def test_signal_strength_zero():
    v = scorer.signal_strength({
        "github_commits_last_year": 0,
        "open_source_contributions": 0,
        "certifications": [],
        "hackathon_wins": 0,
    })
    assert v == 0.0

def test_signal_strength_bounds():
    maxed = {
        "github_commits_last_year": 10000,
        "open_source_contributions": 1000,
        "certifications": [{"name": f"C{i}"} for i in range(20)],
        "hackathon_wins": 99,
    }
    v = scorer.signal_strength(maxed)
    assert v == 1.0, "Maxed signals should hit 1.0 ceiling"


# ─── role_alignment ───────────────────────────────────────────────────────────

def test_role_alignment_perfect():
    candidate = {
        "experience": [{"role": "ML Engineer", "company": "X", "is_current": True}],
        "skills": [{"name": "Python"}, {"name": "PyTorch"}, {"name": "FastAPI"}],
    }
    v = scorer.role_alignment(candidate, JD)
    assert v >= 0.70, f"Expected high alignment, got {v}"

def test_role_alignment_zero():
    candidate = {
        "experience": [{"role": "Product Designer", "company": "X"}],
        "skills": [{"name": "Figma"}, {"name": "Sketch"}],
    }
    v = scorer.role_alignment(candidate, JD)
    assert v < 0.30, f"Expected low alignment, got {v}"

def test_role_alignment_bounds():
    v = scorer.role_alignment(SENIOR, JD)
    assert 0.0 <= v <= 1.0


# ─── compute (full BharatScore) ───────────────────────────────────────────────

def test_compute_returns_all_fields():
    result = scorer.compute(SENIOR, semantic_score=0.85, jd=JD)
    assert "candidate_id"  in result
    assert "final_score"   in result
    assert "signals"       in result
    assert "weights"       in result
    assert set(result["signals"].keys()) == {
        "semantic_fit", "career_velocity", "skill_recency",
        "signal_strength", "role_alignment"
    }

def test_compute_score_in_range():
    result = scorer.compute(SENIOR, semantic_score=0.80, jd=JD)
    assert 0.0 <= result["final_score"] <= 1.0

def test_compute_weights_sum():
    weights = scorer.compute(SENIOR, 0.7, JD)["weights"]
    total = round(sum(weights.values()), 6)
    assert total == 1.0, f"Weights should sum to 1.0, got {total}"

def test_compute_semantic_dominance():
    """Higher semantic score should produce higher final score (all else equal)."""
    low  = scorer.compute(SENIOR, semantic_score=0.40, jd=JD)["final_score"]
    high = scorer.compute(SENIOR, semantic_score=0.95, jd=JD)["final_score"]
    assert high > low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
