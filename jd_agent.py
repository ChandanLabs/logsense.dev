"""
jd_agent.py — LLM-Powered JD Understanding Agent
===================================================
Uses google-genai (current SDK) to deconstruct job descriptions.
Gracefully falls back to rule-based enrichment when no API key.
"""
import json, os
from typing import Dict
from core.config import GEMINI_API_KEY

_DECONSTRUCT_PROMPT = """
You are an expert Indian tech recruiter with 10+ years of experience.
Analyse this Job Description and return JSON with these keys:
- "hidden_intent": What culture/person-type the company really wants (1-2 sentences)
- "normalised_skills": Expanded concrete skill list (array of strings)
- "role_category": One of [ml_engineer, data_scientist, full_stack, backend, frontend, devops, research, analyst, other]
- "enriched_query": 60-80 word paragraph for semantic search combining explicit + implicit requirements

Respond ONLY with valid JSON. No markdown fences, no preamble.

JOB DESCRIPTION:
{jd_text}
"""

_CATEGORY_MAP = {
    "ml engineer": "ml_engineer", "machine learning": "ml_engineer",
    "data scientist": "data_scientist", "data science": "data_scientist",
    "full stack": "full_stack", "fullstack": "full_stack",
    "backend": "backend", "front end": "frontend", "frontend": "frontend",
    "devops": "devops", "research": "research", "analyst": "analyst",
}

class JDAgent:
    def __init__(self):
        self._llm_enabled = bool(GEMINI_API_KEY)
        self._client = None
        if self._llm_enabled:
            try:
                from google import genai
                self._client = genai.Client(api_key=GEMINI_API_KEY)
                print("  ✅ JD Agent: Gemini 2.0 Flash enabled.")
            except Exception as e:
                try:
                    import google.generativeai as old_genai
                    old_genai.configure(api_key=GEMINI_API_KEY)
                    self._client = old_genai.GenerativeModel("gemini-1.5-flash")
                    self._legacy = True
                    print("  ✅ JD Agent: Gemini 1.5 Flash (legacy SDK) enabled.")
                except Exception:
                    self._llm_enabled = False
                    print(f"  ⚠️  JD Agent: Gemini unavailable ({e}). Using rule-based fallback.")
        else:
            print("  ⚠️  JD Agent: No GEMINI_API_KEY — using rule-based fallback.")
        self._legacy = getattr(self, "_legacy", False)

    def deconstruct(self, jd: Dict) -> Dict:
        if self._llm_enabled and self._client:
            try:
                return self._llm_deconstruct(jd)
            except Exception as e:
                print(f"  ⚠️  Gemini call failed ({e}). Using fallback.")
        return self._rule_based_fallback(jd)

    def _llm_deconstruct(self, jd: Dict) -> Dict:
        prompt = _DECONSTRUCT_PROMPT.format(jd_text=self._jd_to_text(jd))
        if self._legacy:
            response = self._client.generate_content(prompt)
            raw = response.text.strip()
        else:
            from google import genai
            response = self._client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        enriched = {**jd, **parsed}
        print(f"  ✅ JD Agent enriched '{jd['role_title']}' → category={parsed.get('role_category','?')}")
        return enriched

    def _rule_based_fallback(self, jd: Dict) -> Dict:
        title_lower = jd.get("role_title", "").lower()
        desc_lower  = jd.get("description", "").lower()
        role_category = "other"
        for kw, cat in _CATEGORY_MAP.items():
            if kw in title_lower or kw in desc_lower:
                role_category = cat; break
        normalised_skills = list(set(jd.get("required_skills", []) + jd.get("nice_to_have_skills", [])))
        enriched_query = (
            f"{jd.get('role_title')} engineer proficient in "
            f"{', '.join(jd.get('required_skills', [])[:5])}. "
            f"{jd.get('description', '')[:200]}"
        )
        return {
            **jd,
            "role_category": role_category,
            "normalised_skills": normalised_skills,
            "hidden_intent": "; ".join(jd.get("implicit_signals", [])),
            "enriched_query": enriched_query,
        }

    @staticmethod
    def _jd_to_text(jd: Dict) -> str:
        return (
            f"Role: {jd.get('role_title')}\nCompany: {jd.get('company')} ({jd.get('company_type','')})\n"
            f"Experience: {jd.get('experience_min')}-{jd.get('experience_max')} years\n"
            f"Description: {jd.get('description','')}\n"
            f"Required: {', '.join(jd.get('required_skills',[]))}\n"
            f"Nice to have: {', '.join(jd.get('nice_to_have_skills',[]))}\n"
            f"Responsibilities: {'; '.join(jd.get('responsibilities',[]))}\n"
            f"Cultural signals: {'; '.join(jd.get('implicit_signals',[]))}"
        )
