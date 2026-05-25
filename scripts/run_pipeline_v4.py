"""Pipeline v4 — Individual stock backtest with PEG + PRG + dispersion factors.

Key improvements over v3:
  - 50-100 AI stocks (not 12 ETFs)
  - Static factor base + weekly momentum overlay
  - No look-ahead (T-1 signal → T return)

Usage:  uv run python scripts/run_pipeline_v4.py
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tushare as ts

from data.store.duckdb_store import DuckDBStore
from backtest.engine import WeeklyBacktestEngine, BacktestConfig
from backtest.evaluation import evaluate

DB_PATH = "data/trading_god.duckdb"
TUSHARE_TOKEN = "51c1318e74cde3b12e99eced7b536b2983faa25866308634823f7726"
store = DuckDBStore(DB_PATH)

ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api()

# ── Config ──────────────────────────────────────────────────────────

TOP_N = 15           # Top N stocks to hold
ALPHA = 0.35         # Base momentum weight
START = "20200101"
CALL_DELAY = 1.0     # Tushare rate limit

# ── Step 1: Universe ────────────────────────────────────────────────

def build_universe() -> pd.DataFrame:
    """Build stock universe from analyst_forecast with valid data."""
    print("[1/4] Building universe...")

    fc = store.read_df("analyst_forecast")
    if fc.empty:
        raise RuntimeError("No analyst_forecast. Run ingest_tushare.py.")

    # Get unique stocks with their sector info
    universe = fc[["ts_code", "symbol", "name", "sector_label"]].drop_duplicates("ts_code")
    print(f"  {len(universe)} stocks with forecast data")

    # Compute static factor scores (PEG + PRG hybrid)
    from factors.peg_factor import compute_peg_factors
    peg_df = compute_peg_factors(use_revisions=False)
    if peg_df.empty:
        raise RuntimeError("PEG factor computation failed.")

    # Merge static scores
    universe = universe.merge(
        peg_df[["ts_code", "peg", "consensus_growth", "profit_quality", "hybrid_signal"]],
        on="ts_code", how="inner"
    )
    print(f"  {len(universe)} stocks with valid PEG/PRG scores")
    return universe


# ── Step 2: Weekly Prices ──────────────────────────────────────────

def fetch_weekly_prices(universe: pd.DataFrame) -> pd.DataFrame:
    """Pull daily prices from Tushare, resample to weekly Friday."""
    print(f"[2/4] Fetching prices for {len(universe)} stocks (Tushare)...")

    if store.table_exists("stock_prices_weekly"):
        cached = store.read_df("stock_prices_weekly")
        cached["date"] = pd.to_datetime(cached["date"])
        cached_symbols = cached["symbol"].nunique()
        if cached_symbols >= len(universe) * 0.7:
            print(f"  Using cached: {len(cached)} rows, {cached_symbols} stocks")
            return cached

    all_frames = []
    success, fail = 0, 0
    for i, (_, row) in enumerate(universe.iterrows()):
        ts_code = row["ts_code"]
        try:
            df = PRO.daily(ts_code=ts_code, start_date=START, end_date="20260525")
            if df is not None and not df.empty:
                df["symbol"] = row["symbol"]
                df["date"] = pd.to_datetime(df["trade_date"])
                # Resample daily → weekly Friday close
                df = df.sort_values("date")
                df_w = df.set_index("date").resample("W-FRI").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "vol": "sum",
                }).dropna()
                df_w["symbol"] = row["symbol"]
                df_w["date"] = df_w.index
                all_frames.append(df_w.reset_index(drop=True))
                success += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        time.sleep(CALL_DELAY)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(universe)} ({success} ok, {fail} fail)")

    if not all_frames:
        raise RuntimeError("No price data fetched.")
    combined = pd.concat(all_frames, ignore_index=True)
    store.write_df("stock_prices_weekly", combined, mode="replace")
    print(f"  {success}/{len(universe)} stocks, {len(combined)} rows -> stock_prices_weekly")
    return combined


# ── Step 3: Weekly Signals ─────────────────────────────────────────

def compute_weekly_signals(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Generate time-varying signals: static factor + dynamic momentum."""
    print("[3/4] Computing weekly signals...")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    records = []
    for symbol in prices["symbol"].unique():
        sym_prices = prices[prices["symbol"] == symbol].sort_values("date")
        if len(sym_prices) < 52:
            continue

        close = sym_prices["close"].values
        dates = sym_prices["date"].values

        # Dynamic: trend deviation + 4w momentum
        ema52 = pd.Series(close).ewm(span=52, adjust=False).mean().values
        std52 = pd.Series(close).rolling(52).std().values
        trend_z = np.where(std52 > 0, (close - ema52) / std52, 0)
        trend_z = np.clip(trend_z, -3, 3)

        ret4 = np.zeros(len(close))
        ret4[4:] = close[4:] / close[:-4] - 1
        ret4_mean = pd.Series(ret4).rolling(52).mean().values
        ret4_std = pd.Series(ret4).rolling(52).std().values
        ret4_z = np.where(ret4_std > 0, (ret4 - ret4_mean) / ret4_std, 0)
        ret4_z = np.clip(ret4_z, -3, 3)

        dynamic = 0.5 * trend_z + 0.5 * ret4_z

        # Static factor score for this stock
        stock_info = universe[universe["symbol"] == symbol]
        if stock_info.empty:
            continue
        static_score = float(stock_info["hybrid_signal"].iloc[0])

        # Quality-gated momentum: good companies get full boost, weak ones get reduced
        quality_gate = 1.0 / (1.0 + np.exp(-static_score * 3))  # sigmoid: 0→1
        effective_alpha = ALPHA * quality_gate

        for i in range(len(close)):
            records.append({
                "date": dates[i],
                "symbol": symbol,
                "close": close[i],
                "static_score": static_score,
                "dynamic_z": dynamic[i],
                "quality_gate": quality_gate,
                "signal_score": static_score + effective_alpha * dynamic[i],
            })

    signals = pd.DataFrame(records)
    print(f"  {len(signals)} weekly signals, {signals['symbol'].nunique()} stocks")
    return signals


# ── Step 4: Backtest ────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame, universe: pd.DataFrame):
    print("[4/4] Running backtest...")

    prices = signals[["date", "symbol", "close"]].copy()

    config = BacktestConfig(
        name="v4 Stock-Level Factor Backtest",
        start_date=str(signals["date"].min().date()),
        end_date=str(signals["date"].max().date()),
        frequency="weekly",
        top_n=TOP_N,
        max_position_weight=1.0 / TOP_N,
        max_turnover=0.50,
        initial_capital=10_000_000,
        cash_buffer_pct=0.05,
        stamp_duty=0.001,
        commission=0.0003,
        slippage_bps=10.0,
    )

    engine = WeeklyBacktestEngine()
    result = engine.run(config, signals, prices)
    report = evaluate(result)

    print()
    print("-" * 60)
    print("BACKTEST RESULTS (v4 — Stock-Level Factors)")
    print("-" * 60)
    print(report.summary)
    print(f"  Total Cost: {result.total_cost:,.0f}")
    print()
    print(report.metrics_table.to_string(index=False))

    # Current top holdings
    latest_date = signals["date"].max()
    latest = signals[signals["date"] == latest_date].nlargest(10, "signal_score")
    print(f"\n  Top 10 stocks ({latest_date.date()}):")
    print(f"  {'':>4s} {'Stock':<12s} {'Total':>7s} {'Static':>7s} {'Dynamic':>7s} {'Gate':>5s}")
    for i, (_, r) in enumerate(latest.iterrows()):
        name = universe[universe["symbol"] == r["symbol"]]
        stock_name = name["name"].iloc[0] if not name.empty else r["symbol"]
        gate = r.get("quality_gate", r.get("dynamic_z", 0) * 0.35 / max(r["dynamic_z"] * 0.35, 0.01) if r["dynamic_z"] != 0 else 0.5)
        print(f"  {i+1:>3d}. {stock_name:<12s} {r['signal_score']:>+6.3f} {r['static_score']:>+7.3f} "
              f"{r['dynamic_z']:>+7.2f} {gate:>4.2f}")

    return result, report


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TradingGod v4 — Stock-Level Factor Backtest")
    print("=" * 60)

    universe = build_universe()
    prices = fetch_weekly_prices(universe)
    signals = compute_weekly_signals(universe, prices)
    result, report = run_backtest(signals, universe)

    print("\nPipeline complete.")
    return result, report


if __name__ == "__main__":
    result, report = main()
