"""Convert raw Instacart CSVs to typed Parquet files (the lake layer)."""
import argparse
from pathlib import Path

import duckdb

SCHEMAS = {
    "aisles": ("aisles.csv", {"aisle_id": "INTEGER", "aisle": "VARCHAR"}),
    "departments": ("departments.csv", {"department_id": "INTEGER", "department": "VARCHAR"}),
    "products": (
        "products.csv",
        {"product_id": "INTEGER", "product_name": "VARCHAR", "aisle_id": "INTEGER", "department_id": "INTEGER"},
    ),
    "orders": (
        "orders.csv",
        {
            "order_id": "INTEGER",
            "user_id": "INTEGER",
            "eval_set": "VARCHAR",
            "order_number": "SMALLINT",
            "order_dow": "SMALLINT",
            "order_hour_of_day": "SMALLINT",
            "days_since_prior_order": "DOUBLE",
        },
    ),
    "order_products_prior": (
        "order_products__prior.csv",
        {"order_id": "INTEGER", "product_id": "INTEGER", "add_to_cart_order": "SMALLINT", "reordered": "SMALLINT"},
    ),
    "order_products_train": (
        "order_products__train.csv",
        {"order_id": "INTEGER", "product_id": "INTEGER", "add_to_cart_order": "SMALLINT", "reordered": "SMALLINT"},
    ),
}


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for table, (csv_name, cols) in SCHEMAS.items():
        src = (raw / csv_name).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Put the Instacart CSVs in {raw}/")
        dst = (out / f"{table}.parquet").resolve()
        types = ", ".join(f"'{c}': '{t}'" for c, t in cols.items())
        con.execute(
            f"COPY (SELECT * FROM read_csv('{src.as_posix()}', header=true, columns={{{types}}})) "
            f"TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dst.as_posix()}')").fetchone()[0]
        print(f"{table}: {n:,} rows -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/lake"))
    a = ap.parse_args()
    main(a.raw, a.out)
