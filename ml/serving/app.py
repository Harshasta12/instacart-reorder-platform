"""Reorder recommendation API: GET /recommend/{user_id}?k=10"""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

MODELS = Path("models")
LAKE = Path("data/lake")

app = FastAPI(title="Instacart Reorder API", version="1.0")

model = lgb.Booster(model_file=str(MODELS / "lgbm_v2.txt"))
features = json.loads((MODELS / "features.json").read_text())
feats = pd.read_parquet(MODELS / "serving_features.parquet").sort_values("user_id").set_index("user_id")
names = pd.read_parquet(LAKE / "products.parquet")[["product_id", "product_name"]]


@app.get("/health")
def health():
    return {"status": "ok", "n_users": int(feats.index.nunique()), "n_features": len(features)}


@app.get("/sample_users")
def sample_users(n: int = Query(5, ge=1, le=50)):
    return {"user_ids": [int(u) for u in feats.index.unique()[:n]]}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=50)):
    if user_id not in feats.index:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found; try /sample_users")
    rows = feats.loc[[user_id]].copy()
    rows["prob"] = model.predict(rows[features])
    top = rows.nlargest(k, "prob")[["product_id", "prob"]].merge(names, on="product_id", how="left")
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "product_id": int(r.product_id),
                "product_name": r.product_name,
                "reorder_probability": round(float(r.prob), 4),
            }
            for r in top.sort_values("prob", ascending=False).itertuples()
        ],
    }