"""
rebuild_model.py
================
شغّلي الملف ده مرة واحدة بس عشان تعيدي بناء waynx_model.pkl
على جهازك بالـ numpy اللي عندك.

الخطوة:
    python rebuild_model.py

المطلوب في نفس الفولدر:
    - kem_places__1_.csv
    - kem_users__1_.csv
    - kem_interactions__1___1_.csv
"""

import pandas as pd
import numpy as np
import pickle, os, sys

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
print("Loading CSV files...")

REQUIRED = [
    "kem_places__1_.csv",
    "kem_users__1_.csv",
    "kem_interactions__1___1_.csv",
]
for f in REQUIRED:
    if not os.path.exists(f):
        print(f"\n❌  Missing file: {f}")
        print("    ضعي الـ CSV files في نفس الفولدر وشغلي الأمر تاني.")
        sys.exit(1)

places       = pd.read_csv("kem_places__1_.csv")
users        = pd.read_csv("kem_users__1_.csv")
interactions = pd.read_csv("kem_interactions__1___1_.csv")

print(f"  ✅ Places:       {len(places)} rows")
print(f"  ✅ Users:        {len(users)} rows")
print(f"  ✅ Interactions: {len(interactions):,} rows")

# ── 2. Constants ──────────────────────────────────────────────────────────────
CATS = ["history", "beach", "food", "wellness", "religious", "nature", "adventure"]
BM   = {"low": 0, "medium": 1, "high": 2}
CM   = {"quiet": 0, "moderate": 1, "crowded": 2, "no_preference": 1}
COMP = ["solo", "couple", "family", "friends"]
AGES = ["teen", "adult", "senior"]

# ── 3. CF stats per place ─────────────────────────────────────────────────────
print("\nComputing collaborative filtering signals...")

place_cf = interactions.groupby("place_id").agg(
    cf_avg_rating  = ("rating",  "mean"),
    cf_like_rate   = ("liked",   "mean"),
    cf_visit_rate  = ("visited", "mean"),
    cf_count       = ("interaction_id", "count"),
).reset_index()

places_e = places.merge(place_cf, on="place_id", how="left")
places_e["cf_avg_rating"] = places_e["cf_avg_rating"].fillna(places_e["rating"])
places_e["cf_like_rate"]  = places_e["cf_like_rate"].fillna(0.5)
places_e["cf_visit_rate"] = places_e["cf_visit_rate"].fillna(0.3)
places_e["cf_count"]      = places_e["cf_count"].fillna(0).astype(int)

# ── 4. Place feature matrix (150 × 17) ───────────────────────────────────────
def place_vector(row):
    v  = [1 if row["category"] == c else 0 for c in CATS]
    v += [BM.get(row["budget_level"], 1) / 2]
    v += [CM.get(row["crowd_level"],  1) / 2]
    v += [1 if co in str(row["suitable_for"]).split("|") else 0 for co in COMP]
    v += [1 if ag in str(row["suitable_age"]).split("|")  else 0 for ag in AGES]
    v += [row["rating"] / 5.0]
    return v

place_matrix = np.array(
    [place_vector(r) for _, r in places_e.iterrows()],
    dtype=np.float64
)
print(f"  ✅ Place matrix: {place_matrix.shape}")

# ── 5. Profile-level CF signals ───────────────────────────────────────────────
ui = users.merge(interactions, on="user_id").merge(
     places[["place_id", "category"]], on="place_id")
ui["liked"] = ui["liked"].astype(int)

ps = ui.groupby(["budget", "travel_companion", "age_group", "category"])["liked"].mean().reset_index()

cf_signals = {}
for _, row in ps.iterrows():
    key = f"{row['budget']}|{row['travel_companion']}|{row['age_group']}"
    cf_signals.setdefault(key, {})[row["category"]] = round(float(row["liked"]), 4)

print(f"  ✅ CF signal profiles: {len(cf_signals)}")

# ── 6. Pack & save ────────────────────────────────────────────────────────────
model = {
    "meta": {
        "name":       "WAYNX Hybrid Recommendation Engine",
        "version":    "2.0",
        "algorithm":  "Content-Based + Collaborative Signals",
        "vector_dim": 17,
        "n_places":   len(places_e),
        "n_users":    len(users),
        "n_interactions": len(interactions),
        "feature_schema": (
            CATS + ["budget_norm", "crowd_norm"] + COMP + AGES + ["rating_norm"]
        ),
        "stats": {
            "like_rate":  round(float(interactions["liked"].mean()),   4),
            "visit_rate": round(float(interactions["visited"].mean()),  4),
            "avg_rating": round(float(interactions["rating"].mean()),   4),
        },
    },
    "weights": {
        "W_CONTENT":    0.65,
        "W_CF_LIKE":    0.20,
        "W_CF_RATING":  0.10,
        "W_CF_PROF":    0.05,
        "SEASON_BONUS": 0.06,
    },
    "places_enriched": places_e,
    "place_matrix":    place_matrix,
    "cf_signals":      cf_signals,
    "users":           users,
    "interactions":    interactions,
    "CATS": CATS, "BM": BM, "CM": CM, "COMP": COMP, "AGES": AGES,
}

out = "waynx_model.pkl"
with open(out, "wb") as f:
    pickle.dump(model, f, protocol=4)   # protocol=4 — compatible with Python 3.8+

size = os.path.getsize(out) / 1024
print(f"\n✅  waynx_model.pkl rebuilt successfully ({size:.1f} KB)")
print("    الآن شغّلي:  python main.py")
