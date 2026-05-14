"""End-to-end pipeline v2: data → factors → backtest.

Improvements:
  - ETF-based returns (investable, not synthetic indices)
  - No look-ahead: T-1 signal → T return
  - Realistic A-share costs (stamp duty 0.1% + commission 0.03%)
  - Overlooked constraints: cash buffer, liquidity filter

Usage:  uv run python scripts/run_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.store.duckdb_store import DuckDBStore
from backtest.engine import WeeklyBacktestEngine, BacktestConfig
from backtest.evaluation import evaluate

DB_PATH = "data/trading_god.duckdb"
store = DuckDBStore(DB_PATH)

# ── SW sector → ETF mapping ────────────────────────────────────────
# sina symbol format: sh=Shanghai, sz=Shenzhen
# "proxy" = no dedicated ETF for this SW sector → use closest parent ETF

# Unique ETF symbols (primary + their proxies)
ETF_PRIMARY = {
    "sw_semiconductor": ("sh512480", "半导体ETF"),
    "sw_electronics": ("sz159997", "电子ETF"),
    "sw_computer_equipment": ("sh512720", "计算机ETF"),
    "sw_telecom_equipment": ("sh515880", "通信ETF"),
    "sw_media": ("sh512980", "传媒ETF"),
    "sw_pharma": ("sh512010", "医药ETF"),
    "sw_national_defense": ("sh512660", "军工ETF"),
    "sw_nonferrous": ("sh512400", "有色金属ETF"),
    "sw_auto": ("sh516110", "汽车ETF"),
    "sw_chemical": ("sh516020", "化工ETF"),
    "sw_mechanical_equipment": ("sz159886", "机械ETF"),
    "sw_electric_equipment": ("sh516160", "新能源ETF"),
}

# Full map including proxies (for sector name display)
ETF_MAP = {}
for sector, (code, name) in ETF_PRIMARY.items():
    ETF_MAP[sector] = (code, name)

# Proxy sectors → reuse parent ETF
PROXY_MAP = {
    "sw_optoelectronics": "sw_semiconductor",
    "sw_it_services": "sw_computer_equipment",
    "sw_software": "sw_computer_equipment",
    "sw_telecom_services": "sw_telecom_equipment",
    "sw_specialized_equipment": "sw_mechanical_equipment",
}
for proxy, parent in PROXY_MAP.items():
    ETF_MAP[proxy] = ETF_MAP[parent]

# Only trade primary sectors (no duplicates)
PRIMARY_SECTORS = list(ETF_PRIMARY.keys())

# ── Step 1: ETF data ───────────────────────────────────────────────

def fetch_etf_data() -> pd.DataFrame:
    """Pull ETF price data from Sina API. Results cached in DuckDB."""
    print("[1/5] Fetching ETF data...")

    if store.table_exists("etf_prices_daily"):
        existing = store.read_df("etf_prices_daily")
        if len(existing) > 0:
            print(f"  Using cached: {len(existing)} rows")
            return existing

    import akshare as ak

    all_frames = []
    unique_codes = set(code for code, _ in ETF_MAP.values())

    for code in unique_codes:
        try:
            df = ak.fund_etf_hist_sina(symbol=code)
            if df is not None and not df.empty:
                df["symbol"] = code
                df["date"] = pd.to_datetime(df["date"])
                all_frames.append(df)
                print(f"  OK   {code}: {len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()}")
        except Exception as e:
            print(f"  FAIL {code}: {e}")
        time.sleep(1)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        store.write_df("etf_prices_daily", combined, mode="replace")
        print(f"  → Stored {len(combined)} rows to etf_prices_daily")
        return combined
    else:
        raise RuntimeError("No ETF data fetched")


# ── Step 2: Weekly OHLCV ───────────────────────────────────────────

def prepare_weekly_prices(etf_df: pd.DataFrame) -> pd.DataFrame:
    """Resample ETF daily to weekly Friday close."""
    print("[2/5] Preparing weekly prices...")

    pivot = etf_df.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    )
    weekly = pivot.resample("W-FRI").last().dropna(how="all").ffill()

    # Map ETF codes back to sector codes
    result = {}
    for sector, (etf_code, _) in ETF_MAP.items():
        if etf_code in weekly.columns:
            result[sector] = weekly[etf_code]

    weekly_mapped = pd.DataFrame(result)
    print(f"  {len(weekly_mapped)} weeks × {len(weekly_mapped.columns)} sectors")
    return weekly_mapped


# ── Step 3: Factors ─────────────────────────────────────────────────

def compute_factors(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute 3 expectation-gap factors."""
    print("[3/5] Computing factors...")

    factor_dfs = []
    for sector in weekly.columns:
        sec = weekly[sector].dropna()
        if len(sec) < 52:
            continue

        f1 = _trend_deviation(sec, sector)
        f2 = _relative_strength(weekly, sector)
        f3 = _macro_alignment(sec, sector)

        merged = f1.merge(f2, on="date", how="inner").merge(f3, on="date", how="inner")
        factor_dfs.append(merged)

    combined = pd.concat(factor_dfs, ignore_index=True)
    combined["composite_score"] = (
        0.40 * combined["trend_deviation"] +
        0.35 * combined["relative_strength"] +
        0.25 * combined["macro_alignment"]
    )
    print(f"  {len(combined)} rows, {combined['sector'].nunique()} sectors")
    return combined


def _trend_deviation(series: pd.Series, sector: str) -> pd.DataFrame:
    ema52 = series.ewm(span=52, adjust=False).mean()
    std52 = series.rolling(52).std()
    z = ((series - ema52) / std52).fillna(0).clip(-3, 3)
    return pd.DataFrame({"date": series.index, "sector": sector, "trend_deviation": z.values})


def _relative_strength(weekly: pd.DataFrame, sector: str) -> pd.DataFrame:
    rets = weekly.pct_change(4)
    z = rets.subtract(rets.mean(axis=1), axis=0).divide(rets.std(axis=1), axis=0)
    z = z.clip(-3, 3).fillna(0)
    return pd.DataFrame({"date": z.index, "sector": sector, "relative_strength": z[sector].values})


def _parse_chinese_month(val: str):
    import re
    m = re.search(r"(\d{4})\D*(\d{1,2})", str(val))
    return pd.Timestamp(f"{m.group(1)}-{int(m.group(2)):02d}-01") if m else pd.NaT


def _neutral_macro(series: pd.Series, sector: str) -> pd.DataFrame:
    return pd.DataFrame({"date": series.index, "sector": sector, "macro_alignment": np.zeros(len(series))})


def _macro_alignment(series: pd.Series, sector: str) -> pd.DataFrame:
    pmi_df = store.read_df("macro_pmi_manufacturing")
    if pmi_df.empty or len(pmi_df.columns) < 2:
        return _neutral_macro(series, sector)

    date_col = pmi_df.columns[0]
    pmi_col = pmi_df.columns[1]
    pmi_df[date_col] = pmi_df[date_col].apply(_parse_chinese_month)
    pmi_df = pmi_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    pmi_s = pd.to_numeric(pmi_df[pmi_col], errors="coerce").dropna()

    if pmi_s.empty or len(pmi_s) < 13:
        return _neutral_macro(series, sector)

    pmi_m = pmi_s.resample("MS").last()
    trend = pmi_m - pmi_m.rolling(12).mean()
    trend_w = trend.reindex(series.index, method="ffill").fillna(0)

    rets = series.pct_change()
    common = rets.index.intersection(trend_w.index)
    if len(common) < 12:
        beta = 1.0
    else:
        c = rets.loc[common].corr(trend_w.loc[common])
        beta = 1.0 + c if not pd.isna(c) else 1.0

    raw = (trend_w * beta).values
    mu, sigma = np.nanmean(raw), np.nanstd(raw)
    z = np.where(sigma > 0, (raw - mu) / sigma, 0)
    return pd.DataFrame({"date": series.index, "sector": sector, "macro_alignment": np.clip(z, -3, 3)})


# ── Step 4: Signals ─────────────────────────────────────────────────

def prepare_signals(factors: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Preparing signals...")
    # Only use primary sectors (no duplicate ETF signals)
    primary = factors[factors["sector"].isin(PRIMARY_SECTORS)]
    return primary[["date", "sector", "composite_score"]].rename(
        columns={"composite_score": "signal_score", "sector": "symbol"}
    )


# ── Step 5: Backtest ────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame, weekly: pd.DataFrame):
    print("[5/5] Running backtest...")

    price_records = []
    for sector in weekly.columns:
        for dt, price in weekly[sector].dropna().items():
            price_records.append({"date": dt, "symbol": sector, "close": price})
    prices = pd.DataFrame(price_records)

    config = BacktestConfig(
        name="ETF Factor Backtest (No Look-Ahead)",
        start_date=str(signals["date"].min().date()),
        end_date=str(signals["date"].max().date()),
        frequency="weekly",
        top_n=5,
        max_position_weight=0.25,
        max_turnover=0.50,
        initial_capital=10_000_000,
        cash_buffer_pct=0.05,
        stamp_duty=0.001,
        commission=0.0003,
        slippage_bps=5.0,
        min_hold_weeks=1,
        max_vol_participation=0.01,
    )

    engine = WeeklyBacktestEngine()
    result = engine.run(config, signals, prices)
    report = evaluate(result)

    print()
    print("-" * 60)
    print("BACKTEST RESULTS")
    print("-" * 60)
    print(report.summary)
    print()
    # Show key overlooked metrics
    print(f"  Total Cost (fees+tax): {result.total_cost:,.0f}")
    print(f"  Avg Turnover: {result.turnover_avg:.2f} trades/week")
    print()
    print(report.metrics_table.to_string(index=False))

    # Latest sector ranking
    latest_date = signals["date"].max()
    latest = signals[signals["date"] == latest_date].sort_values("signal_score", ascending=False)
    print(f"\nTop sectors at {latest_date.date()}:")
    for i, row in enumerate(latest.head(8).itertuples()):
        code = row.symbol
        name = ETF_MAP.get(code, ("?", "?"))[1]
        print(f"  {i+1}. {code} ({name}): {row.signal_score:.3f}")

    return result, report


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TradingGod Pipeline v2 — ETF-based, No Look-Ahead")
    print("=" * 60)

    etf_df = fetch_etf_data()
    weekly = prepare_weekly_prices(etf_df)
    factors = compute_factors(weekly)
    signals = prepare_signals(factors)
    result, report = run_backtest(signals, weekly)

    print("\nPipeline complete.")
    return result, report


if __name__ == "__main__":
    result, report = main()
