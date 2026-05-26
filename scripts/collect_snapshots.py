"""Weekly forecast snapshot collector.

Pulls THS analyst consensus EPS for ~100 AI supply chain stocks.
Each run creates a new snapshot — after 4 weeks, the revision time
series becomes usable for expectation gap factors.

Run weekly:  uv run python scripts/collect_snapshots.py
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"
CALL_DELAY = 1.5

# Import shared stock list from ingest_tushare
from scripts.ingest_tushare import AI_STOCKS

store = DuckDBStore(DB_PATH)


def main():
    snapshot_date = date.today()

    print("=" * 60)
    print(f"Forecast Snapshot Collector — {snapshot_date}")
    print(f"Target: {len(AI_STOCKS)} AI supply chain stocks")
    print("=" * 60)

    collect_snapshot(snapshot_date)

    print()
    print_summary(snapshot_date)


def collect_snapshot(snapshot_date: date):
    import akshare as ak

    all_rows = []
    success = 0
    skipped = 0

    for i, (code, name, _label) in enumerate(AI_STOCKS):
        symbol = code.split(".")[0]

        if i > 0 and (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(AI_STOCKS)} done, waiting 5s ...")
            time.sleep(5)  # extra cooldown every 20 stocks

        try:
            df = ak.stock_profit_forecast_ths(symbol=symbol)
            if df is None or df.empty:
                skipped += 1
                continue

            df["snapshot_date"] = snapshot_date
            df["ts_code"] = code
            df["symbol"] = symbol
            df["name"] = name
            all_rows.append(df)
            success += 1

        except Exception as e:
            err = str(e)[:60]
            if i < 3:
                print(f"  FAIL {name} ({symbol}): {err}")

        time.sleep(CALL_DELAY)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        store.write_df("forecast_snapshots", combined, mode="append")
        print(f"\n  OK {success} stocks collected, {skipped} no data")
        print(f"  → Appended {len(combined)} rows to forecast_snapshots")

        # Track metadata
        meta = pd.DataFrame([{
            "snapshot_date": snapshot_date,
            "stocks_success": success,
            "stocks_skipped": skipped,
            "total_rows": len(combined),
            "collected_at": datetime.now().isoformat(),
        }])
        store.write_df("snapshot_metadata", meta, mode="append")


def print_summary(snapshot_date: date):
    if not store.table_exists("forecast_snapshots"):
        print("No snapshots yet.")
        return

    df = store.read_df("forecast_snapshots")
    snaps = df["snapshot_date"].nunique()
    stocks = df["ts_code"].nunique()
    print(f"Total: {snaps} snapshots, {stocks} unique stocks, {len(df)} rows")

    if snaps >= 4:
        print("OK Revision time series ready (4+ snapshots)")
    elif snaps > 0:
        print(f"[{snaps}/4] Accumulating... {4-snaps} more weeks until revision factors activate")


if __name__ == "__main__":
    main()
