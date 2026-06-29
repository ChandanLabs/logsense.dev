"""
ranker.py — BharatRank Main Orchestrator
=========================================
Ties every module together into a single clean pipeline:

  JD + Candidates
      ↓
  JDAgent.deconstruct()         — LLM enrichment
      ↓
  BharatEmbedder.build_index()  — multilingual FAISS index
  BharatEmbedder.search()       — top-K semantic retrieval
      ↓
  MultiSignalScorer.compute()   — 5-signal BharatScore
      ↓
  normaliser.normalise()        — India-first adjustments
      ↓
  BharatExplainer.explain()     — human-readable match reports
      ↓
  Sorted ranked list
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

from core.config import TOP_K_SEMANTIC
from core.embedder import BharatEmbedder
from core.scorer import MultiSignalScorer
from core.jd_agent import JDAgent
from core import normalizer
from core.explainer import BharatExplainer

_DATA_DIR = Path(__file__).parent.parent / "data"
_OUTPUT_DIR = Path(__file__).parent.parent / "output"


class BharatRanker:
    """
    End-to-end candidate ranking pipeline.

    Usage:
        ranker = BharatRanker()
        ranker.load_data()
        results = ranker.rank(jd_id="JD001", top_k=10)
    """

    def __init__(self, verbose: bool = True):
        self.verbose   = verbose
        self.embedder  = BharatEmbedder()
        self.scorer    = MultiSignalScorer()
        self.jd_agent  = JDAgent()
        self.explainer = BharatExplainer()

        self.candidates: List[Dict] = []
        self.jds: List[Dict] = []
        self._candidates_map: Dict[str, Dict] = {}
        self._jds_map: Dict[str, Dict] = {}
        self._index_built = False

    # ─── Data Loading ────────────────────────────────────────────────────────

    def load_data(
        self,
        candidates_path: Optional[str] = None,
        jds_path: Optional[str] = None,
    ) -> None:
        c_path = Path(candidates_path) if candidates_path else _DATA_DIR / "candidates.json"
        j_path = Path(jds_path) if jds_path else _DATA_DIR / "job_descriptions.json"

        with open(c_path) as f:
            self.candidates = json.load(f)
        with open(j_path) as f:
            self.jds = json.load(f)

        self._candidates_map = {c["id"]: c for c in self.candidates}
        self._jds_map        = {j["id"]: j for j in self.jds}
        self._log(f"  ✅ Loaded {len(self.candidates)} candidates, {len(self.jds)} JDs.")

    def build_index(self) -> None:
        """Build the FAISS embedding index over all candidates."""
        self.embedder.build_index(self.candidates)
        self._index_built = True

    # ─── Core Ranking ────────────────────────────────────────────────────────

    def rank(self, jd_id: str, top_k: int = 10) -> Dict:
        """
        Run the full BharatRank pipeline for a single JD.
        Returns a structured result dict ready for JSON serialisation.
        """
        if not self.candidates:
            raise RuntimeError("Call load_data() first.")
        if not self._index_built:
            self.build_index()

        jd = self._jds_map.get(jd_id)
        if not jd:
            raise ValueError(f"JD '{jd_id}' not found.")

        # ── Step 1: Enrich JD ────────────────────────────────────────────────
        self._log(f"\n  🔍 Ranking for: {jd['role_title']} @ {jd['company']}")
        enriched_jd = self.jd_agent.deconstruct(jd)

        # ── Step 2: Semantic retrieval ───────────────────────────────────────
        semantic_results = self.embedder.search(enriched_jd, top_k=TOP_K_SEMANTIC)
        self._log(f"  ✅ Semantic search returned {len(semantic_results)} candidates.")

        # ── Step 3: Multi-signal scoring ─────────────────────────────────────
        scored = []
        for result in semantic_results:
            candidate = self._candidates_map.get(result["candidate_id"])
            if not candidate:
                continue
            score_data = self.scorer.compute(candidate, result["semantic_score"], enriched_jd)
            scored.append(score_data)

        # ── Step 4: India normalization ──────────────────────────────────────
        for entry in scored:
            candidate = self._candidates_map[entry["candidate_id"]]
            norm      = normalizer.normalise(candidate, entry["final_score"])
            entry.update(norm)

        hidden_gem_count = sum(1 for e in scored if e.get("is_hidden_gem"))
        self._log(f"  ✅ Multi-signal scoring complete. {hidden_gem_count} Hidden Gems detected.")

        # ── Step 5: Sort by adjusted score ───────────────────────────────────
        scored.sort(key=lambda x: x.get("adjusted_score", x["final_score"]), reverse=True)

        # ── Step 6: Assign ranks and enrich metadata ─────────────────────────
        top_results = scored[:top_k]
        for i, entry in enumerate(top_results, 1):
            entry["rank"] = i
            candidate = self._candidates_map[entry["candidate_id"]]
            entry["candidate_name"]     = candidate.get("name")
            entry["location"]           = candidate.get("location")
            entry["institution"]        = candidate.get("education", [{}])[0].get("institution", "N/A")
            entry["years_experience"]   = candidate.get("years_total_experience", 0)
            entry["is_fresher"]         = candidate.get("is_fresher", False)
            entry["top_skills"]         = [s["name"] for s in candidate.get("skills", [])[:6]]
            entry["current_role"]       = next(
                (e["role"] + " @ " + e["company"]
                 for e in candidate.get("experience", []) if e.get("is_current")),
                "N/A"
            )
            entry["github_commits"]     = candidate.get("github_commits_last_year", 0)
            entry["certifications"]     = [c["name"] for c in candidate.get("certifications", [])]
            entry["hackathon_wins"]     = candidate.get("hackathon_wins", 0)

        # ── Step 7: Explain ──────────────────────────────────────────────────
        self._log("  ⏳ Generating match reports...")
        for entry in top_results:
            candidate = self._candidates_map[entry["candidate_id"]]
            entry["match_report"] = self.explainer.explain(candidate, entry["rank"], entry, enriched_jd)
        self._log("  ✅ Match reports generated.")

        return self._build_output(enriched_jd, top_results)

    def rank_all(self, top_k: int = 10) -> List[Dict]:
        """Rank candidates across all loaded JDs."""
        results = []
        for jd in self.jds:
            results.append(self.rank(jd["id"], top_k=top_k))
        return results

    # ─── Output ──────────────────────────────────────────────────────────────

    def save_output(self, result: Dict, filename: Optional[str] = None) -> Path:
        """Save ranking result to the output/ directory."""
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fname = filename or f"ranked_{result['jd_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path  = _OUTPUT_DIR / fname
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self._log(f"  💾 Saved → {path}")
        return path

    def save_submission(self, all_results: List[Dict]) -> Path:
        """Create the final hackathon submission file."""
        _OUTPUT_DIR.mkdir(exist_ok=True)
        submission = {
            "metadata": {
                "system":          "BharatRank v1.0",
                "description":     "Multi-Signal Agentic Candidate Intelligence Engine — India-First",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "llm_agent":       "gemini-1.5-flash (with rule-based fallback)",
                "scoring_formula": (
                    "BharatScore = 0.35×SemanticFit + 0.20×CareerVelocity + "
                    "0.20×SkillRecency + 0.15×SignalStrength + 0.10×RoleAlignment"
                ),
                "india_features":  [
                    "Multilingual embeddings (supports Indian-language terms)",
                    "Anti-pedigree bias: institution tier for context only, never for scoring",
                    "Hidden Gem detector: surfaces non-premier candidates with strong signals",
                    "Fairness boost: max +3% for Hidden Gems based on measurable signals",
                    "Indian career gap normalisation",
                ],
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "total_candidates": 20,
            },
            "rankings": all_results,
        }
        path = _OUTPUT_DIR / "bharatrank_submission.json"
        with open(path, "w") as f:
            json.dump(submission, f, indent=2, default=str)
        self._log(f"\n  🎯 Submission file saved → {path}")
        return path

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_output(jd: Dict, ranked: List[Dict]) -> Dict:
        return {
            "jd_id":       jd["id"],
            "jd_title":    f"{jd['role_title']} @ {jd['company']}",
            "total_ranked": len(ranked),
            "ranked_at":   datetime.now(timezone.utc).isoformat(),
            "results":     ranked,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
