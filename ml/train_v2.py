"""v2 reorder model: baseline features + product context, basket size, and recency features."""
import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

LAKE = Path("data/lake")
WAREHOUSE = "data/warehouse.duckdb"
CATEGORICAL = ["aisle_id", "department_id"]


def lake(name: str) -> str:
    return f"read_parquet('{(LAKE / (name + '.parquet')).resolve().as_posix()}')"


def load_data(sample: float):
    con = duckdb.connect(WAREHOUSE, read_only=True)
    sql = f"""
    with users as (
        select user_id, order_id as train_order_id, order_dow as train_dow,
               order_hour_of_day as train_hour, days_since_prior_order as train_gap
        from {lake('orders')}
        where eval_set = 'train' and (hash(user_id) % 1000) < {int(sample * 1000)}
    ),
    orders_prior as (
        select o.user_id, o.order_id, o.order_number, o.days_since_prior_order,
               sum(coalesce(o.days_since_prior_order, 0))
                   over (partition by o.user_id order by o.order_number) as cum_days
        from {lake('orders')} o
        where o.eval_set = 'prior' and o.user_id in (select user_id from users)
    ),
    user_feats as (
        select user_id, max(order_number) as user_orders,
               avg(days_since_prior_order) as user_avg_gap,
               max(cum_days) as user_total_days
        from orders_prior group by 1
    ),
    order_sizes as (
        select op.order_id, count(*) as size
        from {lake('order_products_prior')} op
        join orders_prior using (order_id)
        group by 1
    ),
    user_basket as (
        select op.user_id, avg(s.size) as user_avg_basket,
               arg_max(s.size, op.order_number) as last_order_size
        from orders_prior op join order_sizes s using (order_id)
        group by 1
    ),
    user_hist as (
        select user_id, count(*) as user_distinct_products,
               sum(times_reordered) * 1.0 / sum(times_bought) as user_reorder_rate
        from int_user_product_history group by 1
    ),
    prod_feats as (
        select product_id, count(*) as prod_orders, avg(reordered) as prod_reorder_rate
        from {lake('order_products_prior')} group by 1
    ),
    products as (
        select product_id, aisle_id, department_id from {lake('products')}
    )
    select h.user_id, h.product_id,
           h.times_bought, h.times_reordered, h.first_order_number, h.last_order_number,
           h.avg_cart_position, h.orders_since_last_bought,
           uf.user_orders, uf.user_avg_gap,
           h.times_bought * 1.0 / uf.user_orders as up_order_rate,
           uh.user_distinct_products, uh.user_reorder_rate,
           pf.prod_orders, pf.prod_reorder_rate,
           u.train_dow, u.train_hour, u.train_gap,
           pr.aisle_id, pr.department_id,
           ub.user_avg_basket, ub.last_order_size,
           (uf.user_total_days - cl.cum_days) + coalesce(u.train_gap, 0) as days_since_last_bought,
           (cl.cum_days - cf.cum_days) / nullif(h.times_bought - 1, 0) as avg_days_between_purchases,
           row_number() over (
               partition by h.user_id order by h.times_bought desc, h.last_order_number desc
           ) as up_rank,
           (t.product_id is not null)::int as label
    from int_user_product_history h
    join users u using (user_id)
    join user_feats uf using (user_id)
    join user_hist uh using (user_id)
    join user_basket ub using (user_id)
    join prod_feats pf using (product_id)
    join products pr using (product_id)
    join orders_prior cl on cl.user_id = h.user_id and cl.order_number = h.last_order_number
    join orders_prior cf on cf.user_id = h.user_id and cf.order_number = h.first_order_number
    left join {lake('order_products_train')} t
        on t.order_id = u.train_order_id and t.product_id = h.product_id
    """
    df = con.execute(sql).df()
    df["in_last_order"] = (df["last_order_number"] == df["user_orders"]).astype(int)
    df["days_ratio"] = df["days_since_last_bought"] / df["avg_days_between_purchases"]
    return df.replace([np.inf, -np.inf], np.nan)


def main(sample: float):
    df = load_data(sample)
    print(f"rows: {len(df):,} | users: {df.user_id.nunique():,} | positive rate: {df.label.mean():.3f}")

    is_valid = (df.user_id % 5 == 0).values  # same user-level split as baseline
    features = [c for c in df.columns if c not in ("user_id", "product_id", "label")]
    X_tr, y_tr = df.loc[~is_valid, features], df.loc[~is_valid, "label"]
    X_va, y_va = df.loc[is_valid, features], df.loc[is_valid, "label"]

    params = dict(
        objective="binary", learning_rate=0.1, num_leaves=63,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
    )
    mlflow.set_experiment("instacart-reorder")
    with mlflow.start_run(run_name="lgbm-v2"):
        mlflow.log_params({**params, "sample": sample, "n_features": len(features)})
        mlflow.set_tag("features_added", "aisle,dept,basket_size,recency,up_rank,in_last_order")
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CATEGORICAL)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=CATEGORICAL, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=500, valid_sets=[dva],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )
        p = model.predict(X_va)
        auc = roc_auc_score(y_va, p)
        ap = average_precision_score(y_va, p)
        prec, rec, _ = precision_recall_curve(y_va, p)
        best_f1 = float(np.max(2 * prec * rec / np.clip(prec + rec, 1e-9, None)))
        mlflow.log_metrics({"val_auc": auc, "val_avg_precision": ap, "val_best_f1": best_f1})
        mlflow.lightgbm.log_model(model, name="model")

        imp = sorted(zip(features, model.feature_importance("gain")), key=lambda x: -x[1])
        print(f"\nAUC={auc:.4f}  AP={ap:.4f}  best F1={best_f1:.4f}")
        print("Top features:", [f for f, _ in imp[:8]])


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--sample", type=float, default=0.2, help="fraction of users to use")
    main(ap_.parse_args().sample)