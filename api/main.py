"""
WAYNX — Hybrid Recommendation Engine
FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import uvicorn

from model import WaynxModel

# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WAYNX Recommendation API",
    description="Hybrid Tourism Recommendation System for Egypt",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
waynx = WaynxModel("waynx_model.pkl")

# ── Schemas ───────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    interests:          List[str] = Field(..., example=["history", "food", "adventure"],
                                          description="1–7 categories from: history, beach, food, wellness, religious, nature, adventure")
    budget:             str       = Field(..., example="medium",  description="low | medium | high")
    travel_companion:   str       = Field(..., example="couple",  description="solo | couple | family | friends")
    age_group:          str       = Field(..., example="adult",   description="teen | adult | senior")
    crowd_preference:   str       = Field(..., example="moderate",description="quiet | moderate | no_preference | crowded")
    season:             str       = Field(..., example="winter",  description="winter | spring | summer | autumn")
    trip_duration_days: int       = Field(..., ge=2, le=14,       description="Number of travel days (2–14)")

    @validator("interests")
    def validate_interests(cls, v):
        valid = {"history","beach","food","wellness","religious","nature","adventure"}
        bad = [i for i in v if i not in valid]
        if bad:
            raise ValueError(f"Invalid interests: {bad}. Choose from: {valid}")
        if len(v) < 1:
            raise ValueError("At least one interest is required.")
        return v

    @validator("budget")
    def validate_budget(cls, v):
        if v not in ("low", "medium", "high"):
            raise ValueError("budget must be: low | medium | high")
        return v

    @validator("travel_companion")
    def validate_companion(cls, v):
        if v not in ("solo", "couple", "family", "friends"):
            raise ValueError("travel_companion must be: solo | couple | family | friends")
        return v

    @validator("age_group")
    def validate_age(cls, v):
        if v not in ("teen", "adult", "senior"):
            raise ValueError("age_group must be: teen | adult | senior")
        return v

    @validator("season")
    def validate_season(cls, v):
        if v not in ("winter", "spring", "summer", "autumn"):
            raise ValueError("season must be: winter | spring | summer | autumn")
        return v


class RecommendRequest(BaseModel):
    user:    UserProfile
    top_n:   int = Field(10, ge=1, le=50, description="Number of results (1–50)")


class ItineraryRequest(BaseModel):
    user:    UserProfile
    pool_n:  int = Field(60, ge=20, le=150, description="Recommendation pool size")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "WAYNX Recommendation API",
        "version": "2.0.0",
        "status":  "running",
        "endpoints": [
            "GET  /health",
            "GET  /places",
            "GET  /places/{place_id}",
            "GET  /stats",
            "POST /recommend",
            "POST /itinerary",
        ]
    }


@app.get("/health")
def health():
    """Health check — returns model status."""
    return {
        "status":        "ok",
        "model_loaded":  waynx.loaded,
        "n_places":      waynx.n_places,
        "n_interactions":waynx.n_interactions,
    }


@app.get("/places")
def get_all_places(
    category: Optional[str] = None,
    city:     Optional[str] = None,
    budget:   Optional[str] = None,
):
    """
    Return all 150 places.
    Optional query filters: ?category=beach  ?city=Cairo  ?budget=low
    """
    return waynx.get_places(category=category, city=city, budget=budget)


@app.get("/places/{place_id}")
def get_place(place_id: int):
    """Return a single place by ID with its CF statistics."""
    place = waynx.get_place_by_id(place_id)
    if not place:
        raise HTTPException(status_code=404, detail=f"Place {place_id} not found")
    return place


@app.get("/stats")
def get_stats():
    """Return model & dataset statistics."""
    return waynx.get_stats()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Generate personalised place recommendations.

    Returns a ranked list of places with:
    - hybrid_score  (combined score)
    - content_score (preference-based cosine similarity)
    - match_pct     (percentage display)
    - score_breakdown (per-component contribution)
    """
    try:
        results = waynx.recommend(req.user.dict(), n=req.top_n)
        return {
            "status":  "ok",
            "user":    req.user.dict(),
            "count":   len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/itinerary")
def build_itinerary(req: ItineraryRequest):
    """
    Build a full multi-city day-by-day itinerary.

    Returns:
    - destinations list with city, days allocated, travel time from previous city
    - daily activity schedule per destination
    - summary stats
    """
    try:
        plan = waynx.build_itinerary(req.user.dict(), pool_n=req.pool_n)
        return {
            "status":       "ok",
            "user":         req.user.dict(),
            "total_days":   req.user.trip_duration_days,
            "itinerary":    plan["itinerary"],
            "summary":      plan["summary"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
