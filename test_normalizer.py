"""
tests/test_normalizer.py — Unit tests for India-first normalizer
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import normalizer

# ─── Fixtures ────────────────────────────────────────────────────────────────

IIT_CANDIDATE = {
    "id": "T_IIT",
    "name": "IIT Candidate",
    "education": [{"institution": "IIT Delhi", "cgpa": 9.0, "graduation_year": 2022}],
    "github_commits_last_year": 300,
    "open_source_contributions": 10,
    "certifications": [{"name": "AWS"}, {"name": "GCP"}],
    "hackathon_wins": 1,
    "skills": [{"name": "Python"}],
    "experience": [],
}

VJTI_CANDIDATE = {
    "id": "T_VJTI",
    "name": "VJTI Candidate",
    "education": [{"institution": "VJTI Mumbai", "cgpa": 9.1, "graduation_year": 2023}],
    "github_commits_last_year": 380,
    "open_source_contributions": 15,
    "certifications": [{"name": "C1"}, {"name": "C2"}, {"name": "C3"}],
    "hackathon_wins": 2,
    "skills": [{"name": "Python"}, {"name": "PyTorch"}],
    "experience": [],
}

WEAK_CANDIDATE = {
    "id": "T_WEAK",
    "name": "Low Signal Candidate",
    "education": [{"institution": "XYZ Engineering College", "cgpa": 6.5, "graduation_year": 2023}],
    "github_commits_last_year": 10,
    "open_source_contributions": 0,
    "certifications": [],
    "hackathon_wins": 0,
    "skills": [],
    "experience": [],
}


# ─── Institution Tier ─────────────────────────────────────────────────────────

def test_iit_is_premier():
    assert normalizer.get_institution_tier(IIT_CANDIDATE) == "premier"

def test_vjti_is_strong():
    tier = normalizer.get_institution_tier(VJTI_CANDIDATE)
    assert tier in ("strong", "tier2", "other")  # VJTI should not be premier

def test_unknown_college():
    cand = {**WEAK_CANDIDATE, "education": [{"institution": "Random College"}]}
    assert normalizer.get_institution_tier(cand) == "other"

def test_no_education():
    assert normalizer.get_institution_tier({"education": []}) == "other"


# ─── Hidden Gem Detection ─────────────────────────────────────────────────────

def test_iit_not_hidden_gem():
    is_gem, _ = normalizer.detect_hidden_gem(IIT_CANDIDATE)
    assert not is_gem, "Premier institution candidates are not hidden gems"

def test_vjti_strong_signals_is_gem():
    is_gem, reason = normalizer.detect_hidden_gem(VJTI_CANDIDATE)
    assert is_gem, "VJTI + strong GitHub + certs + hackathon should be a Hidden Gem"
    assert len(reason) > 0

def test_weak_candidate_not_gem():
    is_gem, _ = normalizer.detect_hidden_gem(WEAK_CANDIDATE)
    assert not is_gem, "Weak signals → not a hidden gem"


# ─── Fairness Boost ──────────────────────────────────────────────────────────

def test_no_boost_for_premier():
    adj, boost = normalizer.apply_fairness_boost(0.80, IIT_CANDIDATE, is_hidden_gem=False)
    assert boost == 0.0
    assert adj == 0.80

def test_boost_for_hidden_gem():
    adj, boost = normalizer.apply_fairness_boost(0.80, VJTI_CANDIDATE, is_hidden_gem=True)
    assert boost > 0.0
    assert adj > 0.80
    assert adj <= 1.0

def test_boost_capped_at_one():
    adj, boost = normalizer.apply_fairness_boost(0.999, VJTI_CANDIDATE, is_hidden_gem=True)
    assert adj <= 1.0


# ─── normalise (full function) ────────────────────────────────────────────────

def test_normalise_returns_expected_keys():
    result = normalizer.normalise(VJTI_CANDIDATE, 0.75)
    assert "institution_tier"  in result
    assert "is_hidden_gem"     in result
    assert "hidden_gem_reason" in result
    assert "fairness_boost"    in result
    assert "adjusted_score"    in result

def test_normalise_adjusted_score_gte_original():
    result = normalizer.normalise(VJTI_CANDIDATE, 0.75)
    assert result["adjusted_score"] >= 0.75

def test_normalise_premier_no_boost():
    result = normalizer.normalise(IIT_CANDIDATE, 0.85)
    assert result["fairness_boost"] == 0.0
    assert result["adjusted_score"] == 0.85


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
