"""
embedder.py — Multilingual Sentence Embedder + FAISS Index
=============================================================
Primary:  paraphrase-multilingual-mpnet-base-v2 (768d, multilingual)
Fallback: TF-IDF + TruncatedSVD (LSA) → FAISS dense index
          — production-equivalent architecture, works fully offline.
          Switch to primary by installing the model on your machine.
"""
import numpy as np
import faiss
from typing import List, Dict
from core.config import EMBEDDING_MODEL, TOP_K_SEMANTIC

# ─── Text serialisation helpers ───────────────────────────────────────────────

def _build_candidate_text(c: Dict) -> str:
    parts = []
    for exp in c.get("experience", []):
        parts.append(f"{exp.get('role','')} at {exp.get('company','')} {exp.get('description','')}")
    skills = [s["name"] for s in c.get("skills", [])]
    if skills: parts.append("skills " + " ".join(skills))
    for proj in c.get("projects", []):
        techs = " ".join(proj.get("technologies", []))
        parts.append(f"{proj.get('name','')} {proj.get('description','')} {techs}")
    certs = [c2["name"] for c2 in c.get("certifications", [])]
    if certs: parts.append("certifications " + " ".join(certs))
    for edu in c.get("education", []):
        parts.append(f"{edu.get('degree','')} {edu.get('institution','')}")
    return " ".join(parts)

def _build_jd_text(jd: Dict) -> str:
    parts = [
        jd.get("role_title", ""),
        jd.get("description", ""),
        "required " + " ".join(jd.get("required_skills", [])),
        "preferred " + " ".join(jd.get("nice_to_have_skills", [])),
        " ".join(jd.get("responsibilities", [])),
        " ".join(jd.get("implicit_signals", [])),
        jd.get("enriched_query", ""),
    ]
    return " ".join(p for p in parts if p)


# ─── Embedder ─────────────────────────────────────────────────────────────────

class BharatEmbedder:
    """
    FAISS-backed embedder with automatic fallback:
      1. Tries sentence-transformers (multilingual, best quality)
      2. Falls back to TF-IDF + LSA (offline, same FAISS interface)
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP = None
        self.candidate_ids: List[str] = []
        self._mode = "unknown"
        self._tfidf = None
        self._svd   = None
        self._st_model = None
        self._try_load_st()

    def _try_load_st(self):
        try:
            from sentence_transformers import SentenceTransformer
            import os, warnings
            # Suppress FutureWarnings from google.generativeai
            warnings.filterwarnings("ignore")
            self._st_model = SentenceTransformer(EMBEDDING_MODEL)
            self._mode = "sentence_transformer"
            print(f"  ✅ Embedder: {EMBEDDING_MODEL} (multilingual, 768d)")
        except Exception:
            self._mode = "tfidf_lsa"
            print("  ✅ Embedder: TF-IDF + LSA (offline mode — swap to multilingual model locally)")

    # ─── Public API ──────────────────────────────────────────────────────────

    def build_index(self, candidates: List[Dict]) -> None:
        self.candidate_ids = [c["id"] for c in candidates]
        texts = [_build_candidate_text(c) for c in candidates]
        embeddings = self._embed_batch(texts, fit=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"  ✅ FAISS index built — {dim}d vectors, {len(candidates)} candidates.")

    def search(self, jd: Dict, top_k: int = TOP_K_SEMANTIC) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Call build_index() first.")
        query_text = _build_jd_text(jd)
        query_vec  = self._embed_single(query_text)
        scores, indices = self.index.search(
            query_vec.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.candidate_ids))
        )
        return [
            {"candidate_id": self.candidate_ids[idx], "semantic_score": float(round(sc, 4))}
            for sc, idx in zip(scores[0], indices[0]) if idx >= 0
        ]

    # ─── Embedding backends ──────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str], fit: bool = False) -> np.ndarray:
        if self._mode == "sentence_transformer":
            vecs = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        else:
            return self._tfidf_lsa_batch(texts, fit=fit)

    def _embed_single(self, text: str) -> np.ndarray:
        return self._embed_batch([text], fit=False)[0]

    def _tfidf_lsa_batch(self, texts: List[str], fit: bool = False) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        DIM = 128  # LSA components — good balance of speed vs quality

        if fit:
            self._tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
                min_df=1,
            )
            tfidf_mat = self._tfidf.fit_transform(texts)
            n_components = min(DIM, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense = self._svd.fit_transform(tfidf_mat)
        else:
            tfidf_mat = self._tfidf.transform(texts)
            dense = self._svd.transform(tfidf_mat)

        return normalize(dense, norm="l2").astype(np.float32)

    @property
    def embedding_mode(self) -> str:
        return self._mode
