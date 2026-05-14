"""Data ingestion — pull AI supply chain data from AKShare into DuckDB.

Uses:
  - index_hist_sw() for SW industry index daily data (not rate-limited)
  - Macro indicators (PMI, CPI, PPI, M2, social financing)
  - Stock list

Usage:  uv run python scripts/ingest_data.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore

START_DATE = "20150101"
END_DATE = datetime.today().strftime("%Y%m%d")
DB_PATH = "data/trading_god.duckdb"

# AI supply chain SW indices (first-level + key second-level)
# (sw_code, our_sector_code, display_name)
AI_SW_INDICES = [
    # ── Layer 1: Materials ──
    ("801050", "sw_nonferrous", "有色金属"),
    ("801030", "sw_chemical", "化工"),
    # ── Layer 2: Core Tech ──
    ("801080", "sw_electronics", "电子"),
    ("801081", "sw_semiconductor", "半导体"),
    ("801084", "sw_optoelectronics", "光学光电子"),
    ("801101", "sw_computer_equipment", "计算机设备"),
    ("801103", "sw_it_services", "IT服务"),
    ("801104", "sw_software", "软件开发"),
    ("801102", "sw_telecom_equipment", "通信设备"),
    ("801223", "sw_telecom_services", "通信服务"),
    ("801070", "sw_mechanical_equipment", "机械设备"),
    ("801074", "sw_specialized_equipment", "专用设备"),
    # ── Layer 3: Infrastructure ──
    ("801730", "sw_electric_equipment", "电气设备"),
    ("801110", "sw_auto", "汽车"),
    # ── Layer 4: Applications ──
    ("801760", "sw_media", "传媒"),
    ("801150", "sw_pharma", "医药生物"),
    ("801740", "sw_national_defense", "国防军工"),
]

store = DuckDBStore(DB_PATH)


def main():
    print("=" * 60)
    print("TradingGod Data Ingestion (SW index API)")
    print("=" * 60)

    fetch_sw_indices()
    fetch_macro_data()
    fetch_stock_list()

    print("\n" + "=" * 60)
    print_summary()


# ── 1. SW Industry Index Daily Data ─────────────────────────────────

def fetch_sw_indices():
    """Pull daily SW industry index data for AI supply chain sectors."""
    print(f"\n[1/3] SW industry indices ({len(AI_SW_INDICES)} sectors)...")

    all_frames = []
    for sw_code, our_code, name in AI_SW_INDICES:
        try:
            df = ak.index_hist_sw(symbol=sw_code, period="day")
            if df is None or df.empty:
                print(f"  SKIP {name} ({sw_code}): no data")
                continue

            # Standardize columns
            df = df.rename(columns={
                "日期": "date",
            })
            # Keep original Chinese cols + add metadata
            df["sw_code"] = sw_code
            df["sector_code"] = our_code
            df["sector_name"] = name

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            all_frames.append(df)
            date_range = f"{df['date'].min().date()}" if "date" in df.columns else "N/A"
            print(f"  OK   {name} ({sw_code}): {len(df)} rows, from {date_range}")
        except Exception as e:
            print(f"  FAIL {name} ({sw_code}): {e}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        store.write_df("sw_index_daily", combined, mode="replace")
        print(f"  -> Stored {len(combined)} rows to sw_index_daily")
    else:
        print("  -> No data fetched")


# ── 2. Macro Indicators ─────────────────────────────────────────────

def fetch_macro_data():
    """Pull key macro indicators."""
    print(f"\n[2/3] Macro indicators...")

    calls = [
        ("macro_china_pmi", "pmi_manufacturing", "PMI制造业"),
        ("macro_china_non_man_pmi", "pmi_non_manufacturing", "PMI非制造业"),
        ("macro_china_cpi_monthly", "cpi", "CPI"),
        ("macro_china_ppi", "ppi", "PPI"),
        ("macro_china_money_supply", "m2", "M2"),
        ("macro_china_shrzgm", "social_financing", "社融"),
    ]

    for func_name, code, label in calls:
        try:
            func = getattr(ak, func_name)
            df = func()
            if df is not None and not df.empty:
                store.write_df(f"macro_{code}", df, mode="replace")
                print(f"  OK   {label}: {len(df)} rows")
            else:
                print(f"  SKIP {label}: empty")
        except Exception as e:
            print(f"  FAIL {label}: {e}")


# ── 3. Stock List ───────────────────────────────────────────────────

def fetch_stock_list():
    """Pull A-share stock list."""
    print(f"\n[3/3] Stock list...")
    try:
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            store.write_df("stock_list", df, mode="replace")
            print(f"  OK   {len(df)} symbols")
        else:
            print("  SKIP empty")
    except Exception as e:
        print(f"  FAIL: {e}")


# ── Summary ─────────────────────────────────────────────────────────

def print_summary():
    print("-" * 40)
    tables = store.conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    for (name,) in tables:
        try:
            cnt = store.query(f"SELECT count(*) as n FROM {name}")["n"].iloc[0]
            print(f"  {name}: {cnt} rows")
        except Exception:
            print(f"  {name}: (error)")


if __name__ == "__main__":
    main()
