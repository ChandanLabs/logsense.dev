"""
scorer.py — Multi-Signal Scoring Engine
========================================
Computes BharatScore using 5 weighted signals:

  BharatScore = 0.35 × SemanticFit
              + 0.20 × CareerVelocity
              + 0.20 × SkillRecency
              + 0.15 × SignalStrength
              + 0.10 × RoleAlignment
"""

from typing import Dict, List, Optional
from core.config import (
    WEIGHTS, CURRENT_YEAR,
    RECENCY_WEIGHTS, RECENCY_FLOOR,
    GITHUB_WEIGHT, OSS_WEIGHT, CERTIFICATION_WEIGHT, HACKATHON_WEIGHT,
    GITHUB_MAX_COMMITS, OSS_MAX_CONTRIBS, CERT_MAX_COUNT,
)


class MultiSignalScorer:

    # ─── Signal 1: Career Velocity ────────────────────────────────────

    @staticmethod
    def career_velocity(candidate: Dict) -> float:
        """
        How fast is the candidate growing?
        Freshers get 0.50 (neutral — they haven't had time to advance yet).
        Max velocity = 1 unique role per year (normalised to 1.0).
        """
        total_years = candidate.get("years_total_experience", 0)
        if total_years < 0.5:
            return 0.50   # fresher: neutral

        role_count = len(set(
            exp.get("role", "") for exp in candidate.get("experience", [])
        ))
        velocity = role_count / total_years
        # Cap at 1.0: 1 role change/year is peak velocity (above = aggressive hopping)
        return round(min(velocity, 1.0), 4)

    # ─── Signal 2: Skill Recency ──────────────────────────────────────

    @staticmethod
    def skill_recency(candidate: Dict) -> float:
        """
        Weighted average recency of skills + certifications.
        Recent skills score higher; old skills decay.
        Certifications get a 20% bonus (explicit, dated credential).
        """
        weights: List[float] = []

        for skill in candidate.get("skills", []):
            acquired = skill.get("acquired_year")
            if acquired:
                age = max(0, CURRENT_YEAR - acquired)
                w = RECENCY_WEIGHTS.get(age, RECENCY_FLOOR)
            else:
                w = 0.55   # unknown acquisition → mid-range
            weights.append(w)

        for cert in candidate.get("certifications", []):
            year = cert.get("year")
            if year:
                age = max(0, CURRENT_YEAR - year)
                w = min(RECENCY_WEIGHTS.get(age, RECENCY_FLOOR) * 1.20, 1.0)
            else:
                w = 0.60
            weights.append(w)

        if not weights:
            return 0.40
        return round(min(sum(weights) / len(weights), 1.0), 4)

    # ─── Signal 3: Signal Strength ────────────────────────────────────

    @staticmethod
    def signal_strength(candidate: Dict) -> float:
        """
        Combines GitHub activity, open source contributions,
        certifications, and hackathon wins.
        """
        commits     = candidate.get("github_commits_last_year", 0)
        oss         = candidate.get("open_source_contributions", 0)
        certs       = len(candidate.get("certifications", []))
        hack_wins   = candidate.get("hackathon_wins", 0)

        github_s = min(commits / GITHUB_MAX_COMMITS, 1.0) * GITHUB_WEIGHT
        oss_s    = min(oss / OSS_MAX_CONTRIBS, 1.0)      * OSS_WEIGHT
        cert_s   = min(certs / CERT_MAX_COUNT, 1.0)      * CERTIFICATION_WEIGHT
        hack_s   = min(hack_wins / 3, 1.0)               * HACKATHON_WEIGHT

        return round(min(github_s + oss_s + cert_s + hack_s, 1.0), 4)

    # ─── Signal 4: Role Alignment ─────────────────────────────────────

    @staticmethod
    def role_alignment(candidate: Dict, jd: Dict) -> float:
        """
        Does the candidate's career arc point toward this role?
        Combines:
          - past role title similarity to JD role title (40%)
          - candidate skill overlap with JD required skills (60%)
        """
        target_role   = jd.get("role_title", "").lower()
        required      = {s.lower() for s in jd.get("required_skills", [])}
        cand_skills   = {s["name"].lower() for s in candidate.get("skills", [])}
        cand_roles    = [exp.get("role", "").lower() for exp in candidate.get("experience", [])]

        # Role-title word overlap
        role_score = 0.0
        target_words = set(target_role.split())
        if target_words:
            for role in cand_roles:
                overlap = len(target_words & set(role.split())) / len(target_words)
                role_score = max(role_score, overlap)

        # Skill overlap
        skill_score = 0.0
        if required:
            matched = required & cand_skills
            skill_score = len(matched) / len(required)

        alignment = (role_score * 0.40) + (skill_score * 0.60)
        return round(min(alignment, 1.0), 4)

    # ─── Final BharatScore ────────────────────────────────────────────

    def compute(
        self,
        candidate: Dict,
        semantic_score: float,
        jd: Dict,
    ) -> Dict:
        """
        Compute the full BharatScore with all signal breakdown.
        Returns a dict ready to be merged into the ranked output.
        """
        signals = {
            "semantic_fit":    round(semantic_score, 4),
            "career_velocity": self.career_velocity(candidate),
            "skill_recency":   self.skill_recency(candidate),
            "signal_strength": self.signal_strength(candidate),
            "role_alignment":  self.role_alignment(candidate, jd),
        }

        final_score = round(
            sum(signals[k] * WEIGHTS[k] for k in signals), 4
        )

        return {
            "candidate_id": candidate["id"],
            "final_score":  final_score,
            "signals":      signals,
            "weights":      WEIGHTS,
        }
