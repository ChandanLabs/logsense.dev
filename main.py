"""
main.py — BharatRank FastAPI Application
==========================================
Production-ready REST API with lifespan startup, CORS, OpenAPI docs.
"""
import time, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.ranker import BharatRanker
from api.schemas import RankRequest, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bharatrank")

_ranker: BharatRanker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ranker
    log.info("🚀 BharatRank API starting up...")
    _ranker = BharatRanker(verbose=True)
    _ranker.load_data()
    _ranker.build_index()
    log.info("✅ BharatRank API ready.")
    yield
    log.info("🛑 Shutting down.")

app = FastAPI(
    title="BharatRank API",
    description=(
        "**Multi-Signal Agentic Candidate Intelligence Engine — India-First**\n\n"
        "5 scoring signals: Semantic Fit, Career Velocity, Skill Recency, "
        "Signal Strength, Role Alignment.\n\n"
        "India-first: anti-pedigree bias, Hidden Gem detector, multilingual embeddings."
    ),
    version="1.0.0", lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def add_timing(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time()-start)*1000,1))
    return response

@app.get("/", tags=["System"])
async def root():
    return {"message": "🇮🇳 BharatRank API v1.0", "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["System"])
async def health():
    return HealthResponse(
        status="healthy", version="1.0.0",
        model="paraphrase-multilingual-mpnet-base-v2",
        candidates_loaded=len(_ranker.candidates) if _ranker else 0,
        jds_loaded=len(_ranker.jds) if _ranker else 0,
        index_ready=_ranker._index_built if _ranker else False,
    )

@app.post("/rank", tags=["Ranking"])
async def rank_candidates(request: RankRequest):
    """Rank all candidates against a JD using BharatRank multi-signal scoring."""
    try:
        result = _ranker.rank(jd_id=request.jd_id, top_k=request.top_k)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Ranking failed"); raise HTTPException(status_code=500, detail=str(e))

@app.get("/rank/{jd_id}", tags=["Ranking"])
async def rank_by_id(jd_id: str, top_k: int = Query(10, ge=1, le=50)):
    """GET shortcut for ranking — quick testing."""
    try:
        return JSONResponse(content=_ranker.rank(jd_id=jd_id, top_k=top_k))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rank-all", tags=["Ranking"])
async def rank_all(top_k: int = Query(10, ge=1, le=50)):
    """Rank candidates across ALL JDs."""
    try:
        all_results = _ranker.rank_all(top_k=top_k)
        return JSONResponse(content={"total_jds": len(all_results), "rankings": all_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates", tags=["Data"])
async def list_candidates():
    return JSONResponse(content={"total": len(_ranker.candidates), "candidates": _ranker.candidates})

@app.get("/candidates/{candidate_id}", tags=["Data"])
async def get_candidate(candidate_id: str):
    c = _ranker._candidates_map.get(candidate_id)
    if not c: raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")
    return JSONResponse(content=c)

@app.get("/jds", tags=["Data"])
async def list_jds():
    return JSONResponse(content={"total": len(_ranker.jds), "job_descriptions": _ranker.jds})

@app.get("/jds/{jd_id}", tags=["Data"])
async def get_jd(jd_id: str):
    jd = _ranker._jds_map.get(jd_id)
    if not jd: raise HTTPException(status_code=404, detail=f"JD '{jd_id}' not found.")
    return JSONResponse(content=jd)
