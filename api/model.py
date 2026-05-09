"""
WaynxModel — loads waynx_model.pkl and exposes all recommendation logic.
Used by the FastAPI app (main.py).
"""

import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any


# ── Constants ─────────────────────────────────────────────────────────────────
CATS = ["history", "beach", "food", "wellness", "religious", "nature", "adventure"]
BM   = {"low": 0, "medium": 1, "high": 2}
CM   = {"quiet": 0, "moderate": 1, "crowded": 2, "no_preference": 1}
COMP = ["solo", "couple", "family", "friends"]
AGES = ["teen", "adult", "senior"]

W_CONTENT    = 0.65
W_CF_LIKE    = 0.20
W_CF_RATING  = 0.10
W_CF_PROF    = 0.05
SEASON_BONUS = 0.06

USABLE_HOURS = 6
MIN_DAYS_CAT = {
    "beach": 2, "wellness": 2, "adventure": 2, "nature": 2,
    "history": 1, "religious": 1, "food": 1,
}

TRAVEL_HOURS: Dict[str, float] = {
    "Cairo-Luxor": 5,         "Cairo-Aswan": 8,
    "Cairo-Hurghada": 5,      "Cairo-Sharm El-Sheikh": 5,
    "Cairo-Alexandria": 2.5,  "Cairo-Dahab": 6,
    "Cairo-Marsa Alam": 8,    "Cairo-Siwa": 8,
    "Cairo-Marsa Matrouh": 4, "Cairo-Fayoum": 1.5,
    "Luxor-Aswan": 3,         "Luxor-Hurghada": 4,
    "Hurghada-Marsa Alam": 3, "Hurghada-Sharm El-Sheikh": 4,
    "Sharm El-Sheikh-Dahab": 1, "Alexandria-Marsa Matrouh": 3,
}


class WaynxModel:
    """
    Loads the pre-built waynx_model.pkl and exposes:
        - recommend(user_dict, n)     → ranked place list
        - build_itinerary(user_dict)  → multi-city day plan
        - get_places(...)             → filtered place catalogue
        - get_place_by_id(id)         → single place detail
        - get_stats()                 → dataset & model statistics
    """

    def __init__(self, pkl_path: str = "waynx_model.pkl"):
        self.loaded = False
        self._load(pkl_path)

    # ── Loader ────────────────────────────────────────────────────────────────

    def _load(self, path: str):
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self.places_df:      pd.DataFrame = bundle["places_enriched"]
        self.place_matrix:   np.ndarray   = bundle["place_matrix"]
        self.cf_signals:     dict         = bundle["cf_signals"]
        self.users_df:       pd.DataFrame = bundle["users"]
        self.interactions_df:pd.DataFrame = bundle["interactions"]
        self.meta:           dict         = bundle["meta"]

        self.n_places       = len(self.places_df)
        self.n_interactions = len(self.interactions_df)
        self.loaded         = True
        print(f"[WAYNX] Model loaded — {self.n_places} places, "
              f"{self.n_interactions:,} interactions")

    # ── Vector builders ───────────────────────────────────────────────────────

    def _place_vector(self, row: pd.Series) -> np.ndarray:
        v  = [1 if row["category"] == c else 0 for c in CATS]
        v += [BM.get(row["budget_level"], 1) / 2]
        v += [CM.get(row["crowd_level"],  1) / 2]
        v += [1 if co in str(row["suitable_for"]).split("|") else 0 for co in COMP]
        v += [1 if ag in str(row["suitable_age"]).split("|")  else 0 for ag in AGES]
        v += [row["rating"] / 5.0]
        return np.array(v, dtype=float)

    def _user_vector(self, u: dict) -> np.ndarray:
        ints = u.get("interests", [])
        v  = [1 if c in ints else 0 for c in CATS]
        v += [BM.get(u.get("budget", "medium"), 1) / 2]
        v += [CM.get(u.get("crowd_preference", "no_preference"), 1) / 2]
        v += [1 if u.get("travel_companion") == co else 0 for co in COMP]
        v += [1 if u.get("age_group")        == ag else 0 for ag in AGES]
        v += [1.0]
        return np.array(v, dtype=float)

    # ── Scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 0 else 0.0

    def _profile_signal(self, u: dict, category: str) -> float:
        key = f"{u.get('budget','medium')}|{u.get('travel_companion','solo')}|{u.get('age_group','adult')}"
        return self.cf_signals.get(key, {}).get(category, 0.5)

    def _hybrid_score(self, uv: np.ndarray, u: dict, row: pd.Series) -> dict:
        pv        = self._place_vector(row)
        content   = self._cosine(uv, pv)
        cf_like   = float(row.get("cf_like_rate",  0.5))
        cf_rat    = float(row.get("cf_avg_rating", row["rating"])) / 5.0
        cf_prof   = self._profile_signal(u, row["category"])
        season_b  = SEASON_BONUS if row["best_season"] in (u.get("season", ""), "any") else 0.0
        total     = W_CONTENT*content + W_CF_LIKE*cf_like + W_CF_RATING*cf_rat + W_CF_PROF*cf_prof + season_b
        return {
            "total":      round(total,   4),
            "content":    round(W_CONTENT   * content,  4),
            "cf_like":    round(W_CF_LIKE   * cf_like,  4),
            "cf_rating":  round(W_CF_RATING * cf_rat,   4),
            "cf_profile": round(W_CF_PROF   * cf_prof,  4),
            "season":     round(season_b,               4),
        }

    # ── Public: recommend ─────────────────────────────────────────────────────

    def recommend(self, user_dict: dict, n: int = 10) -> List[dict]:
        """
        Returns top-n places ranked by hybrid score.
        Each result includes full place details + scoring breakdown.
        """
        uv     = self._user_vector(user_dict)
        scored = []

        for _, row in self.places_df.iterrows():
            breakdown = self._hybrid_score(uv, user_dict, row)
            scored.append({
                "place_id":    int(row["place_id"]),
                "name":        row["place_name"],
                "city":        row["city"],
                "category":    row["category"],
                "budget":      row["budget_level"],
                "season":      row["best_season"],
                "crowd":       row["crowd_level"],
                "suitable_for":str(row["suitable_for"]).split("|"),
                "suitable_age":str(row["suitable_age"]).split("|"),
                "duration_h":  int(row["duration_needed"]),
                "rating":      float(row["rating"]),
                "description": row["description"],
                "cf_like_rate":round(float(row.get("cf_like_rate", 0.5)), 3),
                "cf_avg_rating":round(float(row.get("cf_avg_rating", row["rating"])), 3),
                "cf_review_count": int(row.get("cf_count", 0)),
                "hybrid_score":    breakdown["total"],
                "match_pct":       round(breakdown["total"] * 100, 1),
                "score_breakdown": breakdown,
            })

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:n]

    # ── Public: itinerary ─────────────────────────────────────────────────────

    def build_itinerary(self, user_dict: dict, pool_n: int = 60) -> dict:
        """
        Builds a multi-city itinerary from the top pool_n recommendations.
        Returns itinerary list + summary.
        """
        total   = user_dict.get("trip_duration_days", 7)
        pool    = self.recommend(user_dict, n=pool_n)
        used    = set()

        # Score cities
        city_info: Dict[str, dict] = {}
        for p in pool:
            c = p["city"]
            if c not in city_info:
                city_info[c] = {"score": 0.0, "count": 0, "top_cat": p["category"]}
            city_info[c]["score"] += p["hybrid_score"]
            city_info[c]["count"] += 1

        # Greedy nearest-neighbour city ordering
        sorted_cities = sorted(city_info.items(), key=lambda x: -x[1]["score"])
        ordered = [sorted_cities[0]]
        remaining = list(sorted_cities[1:])
        while remaining:
            last_city = ordered[-1][0]
            best_i = max(
                range(len(remaining)),
                key=lambda i: remaining[i][1]["score"] - self._travel_h(last_city, remaining[i][0]) * 0.02
            )
            ordered.append(remaining.pop(best_i))

        # Allocate days
        itinerary   = []
        days_left   = total
        total_acts  = 0

        for city, info in ordered:
            if days_left <= 0:
                break
            min_d = MIN_DAYS_CAT.get(info["top_cat"], 1)
            alloc = min(max(min_d, round(info["count"] / 2)), days_left)
            travel_h = self._travel_h(itinerary[-1]["city"], city) if itinerary else 0

            day_plan = self._fill_days(city, alloc, pool, used)
            acts_count = sum(len(d["activities"]) for d in day_plan)
            total_acts += acts_count

            itinerary.append({
                "destination_index": len(itinerary) + 1,
                "city":              city,
                "category_focus":    info["top_cat"],
                "days_allocated":    alloc,
                "travel_hours_from_previous": travel_h,
                "day_plan":          day_plan,
                "activities_count":  acts_count,
            })
            days_left -= alloc

        return {
            "itinerary": itinerary,
            "summary": {
                "total_days":       total,
                "days_planned":     total - days_left,
                "destinations":     len(itinerary),
                "total_activities": total_acts,
                "cities":           [d["city"] for d in itinerary],
            }
        }

    def _fill_days(self, city: str, n_days: int, pool: List[dict], used: set) -> List[dict]:
        city_pool = [p for p in pool if p["city"] == city]
        days = []
        for day_n in range(1, n_days + 1):
            hours_left, acts = USABLE_HOURS, []
            for p in city_pool:
                if p["place_id"] in used:
                    continue
                ph = min(p["duration_h"], USABLE_HOURS)
                if ph <= hours_left:
                    used.add(p["place_id"])
                    hours_left -= ph
                    acts.append({
                        "place_id":    p["place_id"],
                        "name":        p["name"],
                        "category":    p["category"],
                        "rating":      p["rating"],
                        "hours":       ph,
                        "description": p["description"],
                        "match_pct":   p["match_pct"],
                    })
            days.append({
                "day":        day_n,
                "activities": acts,
                "hours_used": USABLE_HOURS - hours_left,
                "free_hours": hours_left,
            })
        return days

    @staticmethod
    def _travel_h(city_a: str, city_b: str) -> float:
        if city_a == city_b:
            return 0.0
        key1 = f"{city_a}-{city_b}"
        key2 = f"{city_b}-{city_a}"
        return TRAVEL_HOURS.get(key1, TRAVEL_HOURS.get(key2, 4.0))

    # ── Public: catalogue ─────────────────────────────────────────────────────

    def get_places(
        self,
        category: Optional[str] = None,
        city:     Optional[str] = None,
        budget:   Optional[str] = None,
    ) -> dict:
        df = self.places_df.copy()
        if category: df = df[df["category"]     == category]
        if city:     df = df[df["city"]          == city]
        if budget:   df = df[df["budget_level"]  == budget]

        records = []
        for _, row in df.iterrows():
            records.append({
                "place_id":    int(row["place_id"]),
                "name":        row["place_name"],
                "city":        row["city"],
                "category":    row["category"],
                "budget":      row["budget_level"],
                "season":      row["best_season"],
                "crowd":       row["crowd_level"],
                "rating":      float(row["rating"]),
                "duration_h":  int(row["duration_needed"]),
                "description": row["description"],
                "cf_like_rate":   round(float(row.get("cf_like_rate", 0.5)), 3),
                "cf_avg_rating":  round(float(row.get("cf_avg_rating", row["rating"])), 3),
                "cf_review_count":int(row.get("cf_count", 0)),
            })
        return {"count": len(records), "places": records}

    def get_place_by_id(self, place_id: int) -> Optional[dict]:
        row = self.places_df[self.places_df["place_id"] == place_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "place_id":    int(r["place_id"]),
            "name":        r["place_name"],
            "city":        r["city"],
            "category":    r["category"],
            "budget":      r["budget_level"],
            "season":      r["best_season"],
            "crowd":       r["crowd_level"],
            "suitable_for":str(r["suitable_for"]).split("|"),
            "suitable_age":str(r["suitable_age"]).split("|"),
            "rating":      float(r["rating"]),
            "duration_h":  int(r["duration_needed"]),
            "description": r["description"],
            "cf_stats": {
                "like_rate":    round(float(r.get("cf_like_rate",  0.5)), 3),
                "avg_rating":   round(float(r.get("cf_avg_rating", r["rating"])), 3),
                "review_count": int(r.get("cf_count", 0)),
                "visit_rate":   round(float(r.get("cf_visit_rate", 0.3)), 3),
            }
        }

    # ── Public: stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "model": {
                "name":       self.meta.get("name", "WAYNX"),
                "version":    self.meta.get("version", "2.0"),
                "algorithm":  self.meta.get("algorithm"),
                "vector_dim": self.meta.get("vector_dim", 17),
                "weights": {
                    "content_based":   W_CONTENT,
                    "cf_like":         W_CF_LIKE,
                    "cf_rating":       W_CF_RATING,
                    "cf_profile":      W_CF_PROF,
                    "season_bonus":    SEASON_BONUS,
                },
            },
            "dataset": {
                "n_places":       self.n_places,
                "n_users":        len(self.users_df),
                "n_interactions": self.n_interactions,
                "like_rate":      round(float(self.interactions_df["liked"].mean()),  4),
                "visit_rate":     round(float(self.interactions_df["visited"].mean()),4),
                "avg_rating":     round(float(self.interactions_df["rating"].mean()), 4),
            },
            "places_by_category": (
                self.places_df["category"]
                .value_counts()
                .to_dict()
            ),
            "top_cities": (
                self.places_df["city"]
                .value_counts()
                .head(10)
                .to_dict()
            ),
        }
