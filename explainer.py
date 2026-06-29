"""
explainer.py — Explainability Engine
=====================================
Generates human-readable match reports. Recruiters don't trust black boxes.
Two modes: Gemini LLM (rich contextual) or template-based (no key needed).
"""
import json
from typing import Dict, List
from core.config import GEMINI_API_KEY, WEIGHTS

try:
    import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

_EXPLAIN_PROMPT = """
You are a senior technical recruiter in India with expertise in AI/ML hiring.
A candidate has been ranked #{rank} for the role of "{role}" at {company}.

Candidate snapshot:
{candidate_snapshot}

BharatScore: {score:.3f}
Signal breakdown: {signals}

Write a 2–3 sentence match report for the recruiter. Be specific:
- What makes this candidate a strong (or moderate) fit?
- What is their most impressive signal?
- Any notable gap or risk?

Write in third person. No markdown. No bullet points. No preamble.
"""

class BharatExplainer:
    def __init__(self):
        self._llm_enabled = bool(GEMINI_API_KEY) and _GENAI_AVAILABLE
        if self._llm_enabled:
            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
            print("  ✅ Explainer: Gemini 1.5 Flash enabled.")
        else:
            print("  ⚠️  Explainer: Using template-based explanations.")

    def explain(self, candidate: Dict, rank: int, score_data: Dict, jd: Dict) -> str:
        if self._llm_enabled:
            try:
                return self._llm_explain(candidate, rank, score_data, jd)
            except Exception as e:
                print(f"  ⚠️  Gemini explain failed ({e}). Using template.")
        return self._template_explain(candidate, rank, score_data, jd)

    def explain_batch(self, ranked: List[Dict], candidates_map: Dict[str, Dict], jd: Dict) -> List[str]:
        reports = []
        for entry in ranked:
            cid = entry["candidate_id"]
            report = self.explain(candidates_map.get(cid, {}), entry["rank"], entry, jd)
            reports.append(report)
        return reports

    def _llm_explain(self, candidate: Dict, rank: int, score_data: Dict, jd: Dict) -> str:
        snapshot = {
            "name": candidate.get("name"),
            "location": candidate.get("location"),
            "institution": candidate.get("education", [{}])[0].get("institution"),
            "current_role": next((e["role"]+" at "+e["company"] for e in candidate.get("experience",[]) if e.get("is_current")), "N/A"),
            "years_exp": candidate.get("years_total_experience"),
            "top_skills": [s["name"] for s in candidate.get("skills", [])[:5]],
            "certifications": [c["name"] for c in candidate.get("certifications", [])],
            "github_commits": candidate.get("github_commits_last_year"),
            "hackathon_wins": candidate.get("hackathon_wins"),
        }
        signals_str = ", ".join(f"{k}={v:.2f}(x{WEIGHTS[k]})" for k,v in score_data.get("signals",{}).items())
        prompt = _EXPLAIN_PROMPT.format(
            rank=rank, role=jd.get("role_title",""), company=jd.get("company",""),
            candidate_snapshot=json.dumps(snapshot, indent=2),
            score=score_data.get("adjusted_score", score_data.get("final_score",0)),
            signals=signals_str,
        )
        response = self._model.generate_content(prompt)
        return response.text.strip()

    def _template_explain(self, candidate: Dict, rank: int, score_data: Dict, jd: Dict) -> str:
        signals = score_data.get("signals", {})
        score   = score_data.get("adjusted_score", score_data.get("final_score", 0))
        name    = candidate.get("name", "Candidate")
        skills  = [s["name"] for s in candidate.get("skills", [])[:4]]
        role    = next((e["role"]+" at "+e["company"] for e in candidate.get("experience",[]) if e.get("is_current")), "recent role")
        exp_yrs = candidate.get("years_total_experience", 0)
        is_gem  = score_data.get("is_hidden_gem", False)

        strength_parts = []
        if signals.get("semantic_fit", 0) >= 0.75:
            strength_parts.append("strong semantic alignment with job requirements")
        if signals.get("skill_recency", 0) >= 0.78:
            strength_parts.append("highly current and relevant skill set")
        if signals.get("signal_strength", 0) >= 0.65:
            strength_parts.append(f"active technical presence ({candidate.get('github_commits_last_year',0)} GitHub commits/year)")
        if signals.get("career_velocity", 0) >= 0.70:
            strength_parts.append("rapid career progression")
        if not strength_parts:
            top_s = max(signals, key=signals.get) if signals else "semantic_fit"
            strength_parts.append(f"solid {top_s.replace('_',' ')} ({signals.get(top_s,0):.2f})")

        gem_clause = (
            " Notably a Hidden Gem — non-premier institution candidate "
            "outperforming pedigree candidates on measurable signals." if is_gem else ""
        )
        return (
            f"{name} ranks #{rank} (BharatScore: {score:.3f}) with {' and '.join(strength_parts[:2])}. "
            f"Currently {role} with {exp_yrs} year(s) experience; "
            f"top skills: {', '.join(skills)}.{gem_clause}"
        )
