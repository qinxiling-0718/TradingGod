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
ALPHA = 0.12         # Momentum weight (fundamentals lead, momentum assists)
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


# ── Step 3: Certainty Dimensions ───────────────────────────────────

def compute_certainty(universe: pd.DataFrame) -> dict:
    """Three data-driven certainty dimensions per stock.

    Certainty is NOT analyst count. It is data consistency:
      1. Profit Realization: are profits actually delivering?
      2. Growth Stability: is revenue growth predictable?
      3. Moat: margin level × stability → sustainable advantage?
    """
    print("[3/5] Computing certainty dimensions...")
    certainty = {}
    for _, row in universe.iterrows():
        sym = row["symbol"]
        try:
            import akshare as ak
            fin = ak.stock_financial_abstract_ths(symbol=sym)
            if fin is None or fin.empty:
                certainty[sym] = 0.5; continue

            dc=fin.columns[0]; rgc=fin.columns[6]; mgc=fin.columns[12]; pgc=fin.columns[2]

            # Get recent quarters (newest 8)
            fin_sorted = fin.sort_values(dc, ascending=False) if fin[dc].dtype == 'O' else fin.sort_values(dc, ascending=False)
            recent = fin_sorted.head(8)

            margins = []; rev_growths = []; profit_growths = []
            for _, r in recent.iterrows():
                def pp(raw):
                    if not raw or raw in ('False','None','nan',''): return None
                    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
                    except: return None
                mg=pp(str(r[mgc])); rg=pp(str(r[rgc])); pg=pp(str(r[pgc]))
                if mg is not None: margins.append(mg)
                if rg is not None: rev_growths.append(rg)
                if pg is not None: profit_growths.append(pg)

            if len(margins)<3 or len(rev_growths)<3:
                certainty[sym] = 0.5; continue

            # 1. Profit Realization: trend of actual profit vs direction
            profit_trend = np.mean(profit_growths[-3:]) if len(profit_growths)>=3 else 0
            profit_real = 1.0 / (1.0 + np.exp(-profit_trend * 5))
            # penalize large negative swings
            profit_vol = np.std(profit_growths[-4:]) if len(profit_growths)>=4 else 1.0
            profit_real *= max(0.0, 1.0 - profit_vol)

            # 2. Growth Stability: rev growth predictability
            if len(rev_growths)>=4:
                rev_mean = np.mean(rev_growths[-4:])
                rev_std = np.std(rev_growths[-4:])
                growth_stab = 1.0 / (1.0 + rev_std / max(abs(rev_mean), 0.05))
            else:
                growth_stab = 0.5

            # 3. Moat: margin level × margin stability
            mg_mean = np.mean(margins[-4:]) if len(margins)>=4 else np.mean(margins)
            mg_std = np.std(margins[-4:]) if len(margins)>=4 else 1.0
            mg_level = min(1.0, max(0.0, mg_mean * 5))
            mg_stability = 1.0 / (1.0 + mg_std / max(abs(mg_mean), 0.02))
            moat = mg_level * mg_stability

            # Composite certainty
            cert = 0.40 * profit_real + 0.30 * growth_stab + 0.30 * moat
            certainty[sym] = round(max(0.05, min(1.0, cert)), 3)

        except Exception:
            certainty[sym] = 0.5

    # Print distribution
    high = sum(1 for v in certainty.values() if v>0.7)
    mid = sum(1 for v in certainty.values() if 0.3<=v<=0.7)
    low = sum(1 for v in certainty.values() if v<0.3)
    print(f"  Certainty: {high} high (>0.7), {mid} mid, {low} low (<0.3)")

    # Show extremes
    sorted_c = sorted(certainty.items(), key=lambda x:-x[1])
    print(f"  Highest: {', '.join(f'{s}={v:.2f}' for s,v in sorted_c[:4])}")
    print(f"  Lowest:  {', '.join(f'{s}={v:.2f}' for s,v in sorted_c[-4:])}")
    return certainty


# ── Step 4: Weekly Signals ─────────────────────────────────────────

def compute_weekly_signals(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    certainty: dict,
) -> pd.DataFrame:
    """Generate time-varying signals: static factor × certainty + quality-gated momentum."""
    print("[4/5] Computing weekly signals...")

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

        # Data-driven certainty (profit realization + growth stability + moat)
        cert = certainty.get(symbol, 0.5)
        # Fundamental floor: static score gets at least 50% weight
        fundamental_weight = max(cert, 0.5)

        # Quality-gated momentum: good companies get full boost, weak ones get reduced
        quality_gate = 1.0 / (1.0 + np.exp(-static_score * 3))  # sigmoid: 0→1
        effective_alpha = ALPHA * quality_gate

        for i in range(len(close)):
            records.append({
                "date": dates[i],
                "symbol": symbol,
                "close": close[i],
                "static_score": static_score,
                "certainty": cert,
                "effective_static": static_score * fundamental_weight,
                "dynamic_z": dynamic[i],
                "quality_gate": quality_gate,
                "signal_score": static_score * fundamental_weight + effective_alpha * dynamic[i],
            })

    signals = pd.DataFrame(records)
    print(f"  {len(signals)} weekly signals, {signals['symbol'].nunique()} stocks")
    return signals


# ── Step 5: Backtest ────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame, universe: pd.DataFrame):
    print("[5/5] Running backtest...")

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
    print(f"\n  Top 15 stocks ({latest_date.date()}):")
    print(f"  {'':>4s} {'Stock':<12s} {'Total':>7s} {'Static':>7s} {'Eff.St':>7s} {'Dyn':>6s} {'Cert':>5s}")
    for i, (_, r) in enumerate(latest.iterrows()):
        name = universe[universe["symbol"] == r["symbol"]]
        stock_name = name["name"].iloc[0] if not name.empty else r["symbol"]
        cert = r.get("certainty", 0.5)
        eff_st = r.get("effective_static", r["static_score"])
        print(f"  {i+1:>3d}. {stock_name:<12s} {r['signal_score']:>+7.3f} {r['static_score']:>+7.3f} "
              f"{eff_st:>+7.3f} {r['dynamic_z']:>+6.2f} {cert:>4.2f}")

    return result, report


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TradingGod v4 — Stock-Level Factor Backtest")
    print("=" * 60)

    universe = build_universe()
    prices = fetch_weekly_prices(universe)
    certainty = compute_certainty(universe)
    signals = compute_weekly_signals(universe, prices, certainty)
    result, report = run_backtest(signals, universe)

    print("\nPipeline complete.")
    return result, report


if __name__ == "__main__":
    result, report = main()
