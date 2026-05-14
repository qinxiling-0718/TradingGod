"""Data quality check — validates ingested data completeness.

Usage:  uv run python scripts/check_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"

AI_SECTORS = [
    "sw_nonferrous", "sw_chemical",
    "sw_electronics", "sw_semiconductor", "sw_optoelectronics",
    "sw_computer_equipment", "sw_it_services", "sw_software",
    "sw_telecom_equipment", "sw_telecom_services",
    "sw_mechanical_equipment", "sw_specialized_equipment",
    "sw_electric_equipment", "sw_auto",
    "sw_media", "sw_pharma", "sw_national_defense",
]

store = DuckDBStore(DB_PATH)


def main():
    print("=" * 60)
    print("TradingGod Data Quality Check")
    print("=" * 60)

    all_ok = True
    for check in [
        check_tables,
        check_sector_coverage,
        check_date_ranges,
        check_missing_values,
        check_weekly_feasibility,
    ]:
        print(f"\n--- {check.__doc__} ---")
        try:
            if not check():
                all_ok = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_ok = False

    print("\n" + "=" * 60)
    print("All checks PASSED" if all_ok else "Some checks FAILED")
    print("=" * 60)


def check_tables():
    """Required tables exist"""
    required = ["sw_index_daily", "stock_list"]
    macro_tables = [
        "macro_pmi_manufacturing", "macro_pmi_non_manufacturing",
        "macro_cpi", "macro_ppi", "macro_m2", "macro_social_financing",
    ]
    required.extend(macro_tables)

    ok = True
    for t in required:
        exists = store.table_exists(t)
        tag = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{tag}] {t}")
    return ok


def check_sector_coverage():
    """AI sector coverage"""
    if not store.table_exists("sw_index_daily"):
        print("  SKIP: no sw_index_daily")
        return False

    df = store.read_df("sw_index_daily")
    found = set(df["sector_code"].unique())

    ok = True
    for s in AI_SECTORS:
        tag = "OK" if s in found else "MISSING"
        if s not in found:
            ok = False
        print(f"  [{tag}] {s}")
    return ok


def check_date_ranges():
    """Date coverage per sector"""
    if not store.table_exists("sw_index_daily"):
        print("  SKIP: no data")
        return False

    df = store.read_df("sw_index_daily")
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Overall: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Total rows: {len(df)}")
    print()

    for s in AI_SECTORS:
        sec = df[df["sector_code"] == s]
        if sec.empty:
            print(f"  [{s}] NO DATA")
            continue
        d = sec["date"]
        print(f"  {s}: {d.min().date()} -> {d.max().date()} ({len(sec)} days)")


def check_missing_values():
    """Missing values in OHLCV columns"""
    if not store.table_exists("sw_index_daily"):
        print("  SKIP: no data")
        return False

    df = store.read_df("sw_index_daily")
    ohlc = ["开盘", "收盘", "最高", "最低"]
    cols = [c for c in ohlc if c in df.columns]

    if not cols:
        print(f"  No OHLCV columns found. Available: {list(df.columns[:10])}")
        return True

    ok = True
    for col in cols:
        nulls = int(df[col].isna().sum())
        pct = nulls / len(df) * 100
        tag = "OK" if pct < 1 else "WARN"
        if pct >= 1:
            ok = False
        print(f"  [{tag}] {col}: {nulls}/{len(df)} ({pct:.1f}%)")
    return ok


def check_weekly_feasibility():
    """Weekly resample feasibility"""
    if not store.table_exists("sw_index_daily"):
        print("  SKIP: no data")
        return False

    df = store.read_df("sw_index_daily")
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek

    for d in range(5):
        name = ["Mon", "Tue", "Wed", "Thu", "Fri"][d]
        cnt = int((df["dow"] == d).sum())
        print(f"  {name}: {cnt} rows")


if __name__ == "__main__":
    main()
