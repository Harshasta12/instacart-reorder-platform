"""Train the final v2 model on all sampled users and export artifacts for the API."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb

from train_v2 import CATEGORICAL, load_data


def main(sample: float, rounds: int):
    df = load_data(sample)
    features = [c for c in df.columns if c not in ("user_id", "product_id", "label")]
    params = dict(
        objective="binary", learning_rate=0.1, num_leaves=63,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
    )
    model = lgb.train(
        params,
        lgb.Dataset(df[features], df["label"], categorical_feature=CATEGORICAL),
        num_boost_round=rounds,
    )
    out = Path("models")
    out.mkdir(exist_ok=True)
    model.save_model(str(out / "lgbm_v2.txt"))
    (out / "features.json").write_text(json.dumps(features))
    df[["user_id", "product_id"] + features].to_parquet(out / "serving_features.parquet", index=False)
    print(f"saved model + {len(df):,} serving rows for {df.user_id.nunique():,} users -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=90, help="best iteration from v2 early stopping (~89)")
    a = ap.parse_args()
    main(a.sample, a.rounds)