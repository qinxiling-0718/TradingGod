"""Real expectation gap factors using analyst consensus data.

Factors:
  1. Forecast Revision Momentum — direction of consensus EPS changes
  2. Analyst Dispersion — agreement level as signal quality weight
  3. Forecast-Actual Gap — implied growth vs last reported
  4. Valuation Gap — PE vs historical median (from sw_daily)

These replace the constructional momentum factors with data-driven
expectation gap signals.
"""

import numpy as np
import pandas as pd

from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"
store = DuckDBStore(DB_PATH)


def compute_expectation_factors() -> pd.DataFrame:
    """Compute expectation gap factors for all AI stocks.

    Returns DataFrame with columns:
        date, ts_code, symbol, name, industry,
        f_consensus_growth, f_dispersion, f_forecast_premium,
        f_valuation_gap, composite_score
    """
    if not store.table_exists("analyst_forecast"):
        print("  No analyst_forecast table — run ingest_tushare.py first")
        return pd.DataFrame()

    fc = store.read_df("analyst_forecast")
    if fc.empty:
        return pd.DataFrame()

    results = []

    # Process per stock
    for code in fc["ts_code"].unique():
        stock_fc = fc[fc["ts_code"] == code].copy()
        if stock_fc.empty:
            continue

        name = stock_fc["name"].iloc[0]
        industry = stock_fc["industry"].iloc[0]
        symbol = stock_fc["symbol"].iloc[0]

        factors = _compute_stock_factors(stock_fc, code, symbol, name, industry)
        if factors is not None:
            results.append(factors)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    # Cross-sectional z-score normalization
    for col in ["f_consensus_growth", "f_dispersion", "f_forecast_premium", "f_valuation_gap"]:
        if col in combined.columns:
            mu = combined[col].mean()
            sigma = combined[col].std()
            if sigma > 0:
                combined[col + "_z"] = (combined[col] - mu) / sigma
            else:
                combined[col + "_z"] = 0.0

    # Composite score: weighted combination of z-scored factors
    z_cols = [c for c in combined.columns if c.endswith("_z")]
    if z_cols:
        w = np.array([0.35, 0.25, 0.25, 0.15])  # growth, dispersion, premium, valuation
        combined["composite_score"] = sum(
            w[i] * combined[c] for i, c in enumerate(z_cols) if i < len(w)
        )

    return combined


def _compute_stock_factors(
    stock_fc: pd.DataFrame, code: str, symbol: str, name: str, industry: str
) -> pd.DataFrame | None:
    """Compute expectation gap factors for a single stock."""

    # 1. Consensus Growth — EPS growth rate implied by forecast
    col_year = [c for c in stock_fc.columns if "年" in c][0] if any("年" in c for c in stock_fc.columns) else stock_fc.columns[0]
    col_mean = [c for c in stock_fc.columns if "均" in c or "均值" in c][0] if any("均" in c in stock_fc.columns) else stock_fc.columns[3]
    col_min = [c for c in stock_fc.columns if "小" in c][0] if any("小" in c for c in stock_fc.columns) else stock_fc.columns[2]
    col_max = [c for c in stock_fc.columns if "大" in c][0] if any("大" in c for c in stock_fc.columns) else stock_fc.columns[4]
    col_ind = [c for c in stock_fc.columns if "行业" in c or "平均" in c][0] if any(c and ("行业" in c or "平均" in c) for c in stock_fc.columns) else None
    col_n = [c for c in stock_fc.columns if "人" in c or "数" in c][0] if any("人" in c for c in stock_fc.columns) else stock_fc.columns[1]

    stock_fc = stock_fc.sort_values(col_year)

    if len(stock_fc) < 2:
        return None

    # Latest and prior year
    latest = stock_fc.iloc[-1]
    prior = stock_fc.iloc[-2]

    eps_latest = float(latest[col_mean])
    eps_prior = float(prior[col_mean])
    eps_min = float(latest[col_min])
    eps_max = float(latest[col_max])
    n_analysts = int(latest[col_n])

    # Consensus growth rate
    consensus_growth = (eps_latest / eps_prior - 1) if eps_prior > 0 else 0.0

    # Dispersion signal
    abs_dispersion = (eps_max - eps_min) / eps_latest if eps_latest > 0 else 1.0
    # Low dispersion + positive growth = strong positive
    # Low dispersion + negative growth = strong negative
    # High dispersion = signal neutralized
    growth_sign = 1 if consensus_growth > 0 else -1
    dispersion_signal = growth_sign * max(0, 1.0 - abs_dispersion)

    # Forecast premium vs industry
    ind_eps = float(latest[col_ind]) if col_ind and col_ind in latest.index else eps_latest
    forecast_premium = (eps_latest - ind_eps) / ind_eps if ind_eps > 0 else 0.0

    # Valuation gap — compute if PE data available
    valuation_gap = _compute_valuation_gap(code)

    return pd.DataFrame([{
        "ts_code": code,
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "f_consensus_growth": consensus_growth,
        "f_dispersion": dispersion_signal,
        "f_forecast_premium": forecast_premium,
        "f_valuation_gap": valuation_gap,
        "n_analysts": n_analysts,
        "abs_dispersion": abs_dispersion,
        "eps_consensus": eps_latest,
    }])


def _compute_valuation_gap(ts_code: str) -> float:
    """Compute PE valuation gap from Tushare sw_daily data.

    Valuation gap = (current PE / historical PE median) - 1
    Negative = undervalued relative to history.
    """
    if not store.table_exists("sw_industry_pe_daily"):
        return 0.0

    try:
        pe_df = store.read_df("sw_industry_pe_daily")
        if pe_df.empty:
            return 0.0

        # Map stock to SW industry (simplified — use parent industry)
        # For individual stocks, we'd need Tushare's daily_basic for PE
        # For now, use the SW index PE as sector proxy
        pe_col = [c for c in pe_df.columns if "pe" in c.lower()][0] if any("pe" in c.lower() for c in pe_df.columns) else None
        if pe_col is None:
            return 0.0

        pe_series = pd.to_numeric(pe_df[pe_col], errors="coerce").dropna()
        if len(pe_series) < 20:
            return 0.0

        current_pe = pe_series.iloc[-1]
        median_pe = pe_series.median()
        if median_pe > 0:
            return (current_pe / median_pe) - 1
    except Exception:
        pass

    return 0.0


def compute_sector_aggregate(factors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stock-level factors to sector-level signals.

    Maps each stock to its ETF via industry → SW sector mapping.
    Returns sector-level average factor scores.
    """
    if factors.empty:
        return pd.DataFrame()

    # Industry → sector mapping (simplified)
    industry_to_sector = {
        "半导体": "sw_semiconductor",
        "电子": "sw_electronics",
        "元器件": "sw_electronics",
        "计算机": "sw_computer_equipment",
        "软件": "sw_software",
        "通信": "sw_telecom_equipment",
        "互联网": "sw_media",
        "传媒": "sw_media",
        "医药": "sw_pharma",
        "军工": "sw_national_defense",
        "有色": "sw_nonferrous",
        "汽车": "sw_auto",
        "化工": "sw_chemical",
        "机械": "sw_mechanical_equipment",
        "电气": "sw_electric_equipment",
    }

    def map_to_sector(ind):
        for kw, sector in industry_to_sector.items():
            if kw in str(ind):
                return sector
        return str(ind)

    factors = factors.copy()
    factors["sector"] = factors["industry"].apply(map_to_sector)

    # Average factor scores per sector
    sector_agg = factors.groupby("sector").agg(
        avg_consensus_growth=("f_consensus_growth_z", "mean"),
        avg_dispersion=("f_dispersion_z", "mean"),
        avg_premium=("f_forecast_premium_z", "mean"),
        avg_valuation=("f_valuation_gap_z", "mean"),
        composite_score=("composite_score", "mean"),
        stock_count=("ts_code", "count"),
    ).reset_index()

    return sector_agg
