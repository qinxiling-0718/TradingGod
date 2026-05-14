"""Pipeline v3 — Combined Expectation Gap + Momentum factors.

Path B approach:
  Static factor  → consensus EPS growth (cross-sectional: WHICH sectors)
  Dynamic factor → weekly momentum/trend deviation (time-series: WHEN)
  Combined       → static_score × (1 + momentum) → weekly re-ranking

This solves the "寒武纪第一是一个切面" problem — the ranking is dynamic,
not static. High-growth sectors only lead when their momentum confirms.

Usage:  uv run python scripts/run_pipeline_v3.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.store.duckdb_store import DuckDBStore
from backtest.engine import WeeklyBacktestEngine, BacktestConfig
from backtest.evaluation import evaluate

DB_PATH = "data/trading_god.duckdb"
store = DuckDBStore(DB_PATH)

# ── ETF mapping ────────────────────────────────────────────────────

ETF_MAP = {
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
PRIMARY_SECTORS = list(ETF_MAP.keys())


# ── Industry → Sector mapping ──────────────────────────────────────

INDUSTRY_TO_SECTOR = {
    "半导体": "sw_semiconductor", "芯片": "sw_semiconductor",
    "电子": "sw_electronics", "元器件": "sw_electronics",
    "光学": "sw_electronics", "PCB": "sw_electronics",
    "面板": "sw_electronics", "显示": "sw_electronics",
    "计算机": "sw_computer_equipment", "软件": "sw_computer_equipment",
    "IT": "sw_computer_equipment", "数据中心": "sw_computer_equipment",
    "通信": "sw_telecom_equipment", "光通信": "sw_telecom_equipment",
    "网络": "sw_telecom_equipment", "设备": "sw_telecom_equipment",
    "卫星": "sw_telecom_equipment",
    "传媒": "sw_media", "互联网": "sw_media", "平台": "sw_media",
    "金融": "sw_media", "办公": "sw_media",
    "医药": "sw_pharma",
    "军工": "sw_national_defense", "航天": "sw_national_defense",
    "有色": "sw_nonferrous",
    "汽车": "sw_auto", "驾驶": "sw_auto",
    "化工": "sw_chemical",
    "机械": "sw_mechanical_equipment", "制造": "sw_mechanical_equipment",
    "电气": "sw_electric_equipment", "新能源": "sw_electric_equipment",
    "电池": "sw_electric_equipment", "电源": "sw_electric_equipment",
}


# ── Step 1: Static Expectation Factors ──────────────────────────────

def compute_static_scores() -> pd.DataFrame:
    """Compute static consensus growth scores per stock, aggregate to sector."""
    print("[1/4] Static expectation gap scores (consensus EPS growth)...")

    fc = store.read_df("analyst_forecast")
    if fc.empty:
        print("  ERROR: No analyst_forecast data.")
        return pd.DataFrame()

    cols = list(fc.columns)
    col_year = cols[0]; col_n = cols[1]
    col_min = cols[2]; col_mean = cols[3]
    col_max = cols[4]; col_ind = cols[5]

    results = []
    for code in fc["ts_code"].unique():
        sf = fc[fc["ts_code"] == code].sort_values(col_year)
        if len(sf) < 2:
            continue
        latest, prior = sf.iloc[-1], sf.iloc[-2]
        try:
            eps_latest = float(latest[col_mean])
            eps_prior = float(prior[col_mean])
            eps_min = float(latest[col_min])
            eps_max = float(latest[col_max])
            n = int(latest[col_n])
            ind_eps = float(latest[col_ind]) if col_ind else eps_latest
        except (ValueError, TypeError, KeyError):
            continue
        if eps_prior <= 0:
            continue

        growth = (eps_latest / eps_prior) - 1
        disp = (eps_max - eps_min) / eps_latest if eps_latest > 0 else 1.0
        growth_sign = 1 if growth > 0 else -1
        disp_score = growth_sign * max(0, 1.0 - min(disp, 2.0))
        premium = (eps_latest - ind_eps) / ind_eps if ind_eps > 0 else 0.0

        name = sf["name"].iloc[0]
        label = sf["sector_label"].iloc[0] if "sector_label" in sf.columns else ""

        results.append({
            "ts_code": code, "name": name, "label": label,
            "eps_consensus": eps_latest, "n_analysts": n,
            "growth": growth, "dispersion_score": disp_score,
            "premium": premium, "dispersion_raw": disp,
        })

    df = pd.DataFrame(results)

    # Z-score static factors
    for col in ["growth", "dispersion_score", "premium"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[col + "_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df["static_score"] = (
        0.40 * df["growth_z"] +
        0.35 * df["dispersion_score_z"] +
        0.25 * df["premium_z"]
    )

    # Aggregate to sector
    df["sector"] = df.apply(_map_to_sector, axis=1)
    sector_static = df.groupby("sector")["static_score"].mean().to_dict()

    # Fill missing sectors with market average
    all_sectors = {s: sector_static.get(s, 0.0) for s in PRIMARY_SECTORS}

    print(f"  {len(df)} stocks → {len(set(df['sector']) & set(PRIMARY_SECTORS))} sectors mapped")
    top3 = sorted(all_sectors.items(), key=lambda x: -x[1])[:3]
    for s, v in top3:
        print(f"    {s}: static={v:.3f}")
    return df, all_sectors


def _map_to_sector(row):
    text = str(row.get("label", "")) + str(row.get("name", ""))
    for kw, sector in sorted(INDUSTRY_TO_SECTOR.items(), key=lambda x: -len(x[0])):
        if kw in text:
            return sector
    return "other"


# ── Step 2: Dynamic Momentum Factors (weekly) ──────────────────────

def compute_weekly_signals(
    weekly_prices: pd.DataFrame,
    static_scores: dict[str, float],
) -> pd.DataFrame:
    """Combine static expectation scores with weekly momentum.

    For each week × sector:
      dynamic = price momentum z-score (trend deviation + relative strength)
      combined = static_score × (1 + α × dynamic)
      → re-rank every week
    """
    print("[2/4] Computing weekly dynamic signals...")

    records = []
    for sector in weekly_prices.columns:
        close = weekly_prices[sector].dropna()
        if len(close) < 52:
            continue

        # Trend deviation: close vs 52-week EMA
        ema52 = close.ewm(span=52, adjust=False).mean()
        std52 = close.rolling(52).std()
        trend_z = ((close - ema52) / std52).fillna(0).clip(-3, 3)

        # 4-week momentum
        ret4 = close.pct_change(4).fillna(0)
        ret4_z = (ret4 - ret4.rolling(52).mean()) / ret4.rolling(52).std()
        ret4_z = ret4_z.fillna(0).clip(-3, 3)

        # Combined dynamic signal
        dynamic = 0.5 * trend_z + 0.5 * ret4_z

        for dt in close.index:
            records.append({
                "date": dt,
                "symbol": sector,
                "dynamic_z": dynamic.loc[dt] if dt in dynamic.index else 0.0,
                "price": close.loc[dt],
            })

    df = pd.DataFrame(records)
    static_val = df["symbol"].map(static_scores).fillna(0)

    # Combined: static foundation × dynamic adjustment
    alpha = 0.6  # how much momentum can adjust the static signal
    df["signal_score"] = static_val + alpha * df["dynamic_z"]

    print(f"  {len(df)} weekly signals, {df['symbol'].nunique()} sectors")
    return df


# ── Step 3: ETF Weekly Prices ──────────────────────────────────────

def prepare_weekly_prices() -> pd.DataFrame:
    print("[3/4] Preparing ETF weekly prices...")
    etf_df = store.read_df("etf_prices_daily")
    etf_df["date"] = pd.to_datetime(etf_df["date"])
    pivot = etf_df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    weekly = pivot.resample("W-FRI").last().dropna(how="all").ffill()
    result = {}
    for sector, (code, _) in ETF_MAP.items():
        if code in weekly.columns:
            result[sector] = weekly[code]
    df = pd.DataFrame(result)
    print(f"  {len(df)} weeks × {len(df.columns)} ETFs")
    return df


# ── Step 4: Backtest ────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame, weekly: pd.DataFrame):
    print("[4/4] Running backtest...")

    price_records = []
    for sector in weekly.columns:
        for dt, price in weekly[sector].dropna().items():
            price_records.append({"date": dt, "symbol": sector, "close": price})
    prices = pd.DataFrame(price_records)

    config = BacktestConfig(
        name="Hybrid: Static Expectation + Dynamic Momentum",
        start_date=str(weekly.index[0].date()),
        end_date=str(weekly.index[-1].date()),
        frequency="weekly",
        top_n=5,
        max_position_weight=0.25,
        max_turnover=0.50,
        initial_capital=10_000_000,
        cash_buffer_pct=0.05,
        stamp_duty=0.001,
        commission=0.0003,
        slippage_bps=5.0,
    )

    engine = WeeklyBacktestEngine()
    result = engine.run(config, signals, prices)
    report = evaluate(result)

    print()
    print("-" * 60)
    print("BACKTEST RESULTS")
    print("-" * 60)
    print(report.summary)
    print(f"  Total Cost: {result.total_cost:,.0f}")
    print()
    print(report.metrics_table.to_string(index=False))

    # Latest ranking
    latest = signals[signals["date"] == signals["date"].max()].sort_values(
        "signal_score", ascending=False
    )
    print(f"\nTop sectors (latest week):")
    for i, row in enumerate(latest.head(8).itertuples()):
        etf_name = ETF_MAP.get(row.symbol, ("?", "?"))[1]
        print(f"  {i+1}. {row.symbol} ({etf_name}): "
              f"signal={row.signal_score:.3f}")

    return result, report


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TradingGod V3 — Static Expectation × Dynamic Momentum")
    print("=" * 60)

    _, static_scores = compute_static_scores()
    if not static_scores:
        return None, None

    weekly = prepare_weekly_prices()
    signals = compute_weekly_signals(weekly, static_scores)
    result, report = run_backtest(signals, weekly)

    print("\nPipeline complete.")
    return result, report


if __name__ == "__main__":
    result, report = main()
