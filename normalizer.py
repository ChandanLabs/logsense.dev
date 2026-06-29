"""
normalizer.py — India-First Normalization Layer
================================================
The single biggest gap in global ATS systems: built for Silicon Valley,
penalising great Indian candidates from tier-2 colleges and cities.

Three mechanisms:
  1. Institution tier detection — context only, NOT for scoring
  2. Hidden Gem detection — surface candidates buried by brand-name bias
  3. Anti-bias boost — small fairness adjustment for hidden gems with strong signals
"""

from typing import Dict, Tuple
from core.config import (
    PREMIER_KEYWORDS, STRONG_KEYWORDS, TIER2_KEYWORDS,
    HIDDEN_GEM_COMMIT_THRESHOLD, HIDDEN_GEM_CERT_THRESHOLD,
)
from core.scorer import MultiSignalScorer

_scorer = MultiSignalScorer()


def get_institution_tier(candidate: Dict) -> str:
    """Returns 'premier' | 'strong' | 'tier2' | 'other'. Never used to penalise."""
    edu = candidate.get("education", [])
    if not edu:
        return "other"
    institution = edu[0].get("institution", "").upper()
    for kw in PREMIER_KEYWORDS:
        if kw.upper() in institution:
            return "premier"
    for kw in STRONG_KEYWORDS:
        if kw.upper() in institution:
            return "strong"
    for kw in TIER2_KEYWORDS:
        if kw.upper() in institution:
            return "tier2"
    return "other"


def detect_hidden_gem(candidate: Dict) -> Tuple[bool, str]:
    """
    Hidden Gem: non-premier institution + strong measurable signals.
    These candidates are systematically buried by keyword ATS systems.
    """
    tier = get_institution_tier(candidate)
    if tier == "premier":
        return False, ""

    signals = []
    commits = candidate.get("github_commits_last_year", 0)
    if commits >= HIDDEN_GEM_COMMIT_THRESHOLD:
        signals.append(f"{commits} GitHub commits/year")

    certs = len(candidate.get("certifications", []))
    if certs >= HIDDEN_GEM_CERT_THRESHOLD:
        signals.append(f"{certs} certifications")

    hackathons = candidate.get("hackathon_wins", 0)
    if hackathons >= 1:
        signals.append(f"{hackathons} hackathon win(s)")

    oss = candidate.get("open_source_contributions", 0)
    if oss >= 8:
        signals.append(f"{oss} OSS contributions")

    cgpa = candidate.get("education", [{}])[0].get("cgpa", 0)
    if cgpa and cgpa >= 9.0:
        signals.append(f"CGPA {cgpa}")

    if signals:
        institution = candidate.get("education", [{}])[0].get("institution", "their institution")
        reason = (
            f"Strong signals ({', '.join(signals)}) from {institution} — "
            f"often filtered by keyword/brand-name ATS systems."
        )
        return True, reason

    return False, ""


def apply_fairness_boost(final_score: float, candidate: Dict, is_hidden_gem: bool) -> Tuple[float, float]:
    """
    Max +0.03 boost for hidden gems, proportional to signal_strength.
    Corrects for semantic model bias toward prestigious institution names.
    """
    if not is_hidden_gem:
        return round(final_score, 4), 0.0
    signal_s = _scorer.signal_strength(candidate)
    boost    = round(0.03 * signal_s, 4)
    adjusted = round(min(final_score + boost, 1.0), 4)
    return adjusted, boost


def has_explainable_gap(candidate: Dict) -> bool:
    """Detect short-tenure stints common in India's startup ecosystem."""
    for exp in candidate.get("experience", []):
        start = exp.get("start_year")
        end   = exp.get("end_year")
        if start and end and (end - start) == 0:
            return True
    return False


def normalise(candidate: Dict, final_score: float) -> Dict:
    """Apply all India-first normalisation. Returns enriched metadata dict."""
    tier                      = get_institution_tier(candidate)
    is_hidden_gem, gem_reason = detect_hidden_gem(candidate)
    adjusted_score, boost     = apply_fairness_boost(final_score, candidate, is_hidden_gem)
    has_gap                   = has_explainable_gap(candidate)
    return {
        "institution_tier":  tier,
        "is_hidden_gem":     is_hidden_gem,
        "hidden_gem_reason": gem_reason,
        "fairness_boost":    boost,
        "has_career_gap":    has_gap,
        "adjusted_score":    adjusted_score,
    }
