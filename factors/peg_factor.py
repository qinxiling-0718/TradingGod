"""PEG Factor Engine — Forward PEG + ΔPEG + Hybrid Profit-Quality weighting.

Core innovation: Hybrid PEG — blends PE-based and PR-based signals
using an automatic profit-quality weight function (no manual phase rules).

  quality = min(1.0, net_margin × 10)
  hybrid_peg = quality × peg + (1 - quality) × prg

  Early-stage (margin < 5%):  mostly PRG  → revenue growth is the signal
  Mature (margin > 10%):      mostly PEG  → profit growth is the signal
  Transitional (5-10%):       blended    → smooth transition
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"
store = DuckDBStore(DB_PATH)


# ── Sector → ETF price mapping ─────────────────────────────────────

ETF_PRICE_TABLE = "etf_prices_daily"

# ── Industry → Sector mapping ──────────────────────────────────────

INDUSTRY_TO_SECTOR = {
    "半导体": "sw_semiconductor", "芯片": "sw_semiconductor",
    "电子": "sw_electronics", "元器件": "sw_electronics",
    "光学": "sw_electronics", "PCB": "sw_electronics",
    "面板": "sw_electronics", "显示": "sw_electronics",
    "计算机": "sw_computer_equipment", "软件": "sw_computer_equipment",
    "IT": "sw_computer_equipment", "数据": "sw_computer_equipment",
    "通信": "sw_telecom_equipment", "光通信": "sw_telecom_equipment",
    "光模块": "sw_telecom_equipment", "光器件": "sw_telecom_equipment",
    "网络": "sw_telecom_equipment", "卫星": "sw_telecom_equipment",
    "传媒": "sw_media", "互联网": "sw_media", "平台": "sw_media",
    "金融": "sw_media", "办公": "sw_media",
    "医药": "sw_pharma",
    "军工": "sw_national_defense", "航天": "sw_national_defense",
    "航空": "sw_national_defense",
    "有色": "sw_nonferrous", "稀土": "sw_nonferrous", "铜": "sw_nonferrous",
    "汽车": "sw_auto", "驾驶": "sw_auto",
    "化工": "sw_chemical", "材料": "sw_chemical",
    "机械": "sw_mechanical_equipment", "制造": "sw_mechanical_equipment",
    "电气": "sw_electric_equipment", "新能源": "sw_electric_equipment",
    "电池": "sw_electric_equipment", "光伏": "sw_electric_equipment",
}


def _map_sector(name: str, label: str = "") -> str:
    text = str(label) + str(name)
    for kw, sector in sorted(INDUSTRY_TO_SECTOR.items(), key=lambda x: -len(x[0])):
        if kw in text:
            return sector
    return "other"


# ── PEG Computation ─────────────────────────────────────────────────

@dataclass
class StockPEG:
    """PEG metrics for a single stock."""
    ts_code: str
    name: str
    sector: str
    consensus_eps: float        # Next-year consensus EPS
    consensus_growth: float     # Implied consensus EPS growth rate
    forward_pe: float           # Current price / consensus_eps
    peg: float                  # Forward PE / growth rate
    peg_z: float = 0.0          # Cross-sectional z-score
    delta_peg: float = 0.0      # ΔPEG (if we have snapshots)
    growth_revision: float = 0.0  # Δconsensus_growth (if snapshots)
    # Hybrid PEG fields
    revenue_growth: float = 0.0     # YoY revenue growth
    net_margin: float = 0.0         # Net profit margin
    profit_quality: float = 0.5     # Automatic weight: 0=pure PRG, 1=pure PEG
    hybrid_signal: float = 0.0      # quality × peg_signal + (1-quality) × prg_signal


def compute_peg_factors(use_revisions: bool = True) -> pd.DataFrame:
    """Compute PEG factors for all stocks with forecast data.

    Args:
        use_revisions: If True and 2+ snapshots exist, compute ΔPEG.

    Returns:
        DataFrame with stock-level PEG metrics.
    """
    if not store.table_exists("analyst_forecast"):
        print("  No analyst_forecast data")
        return pd.DataFrame()

    fc = store.read_df("analyst_forecast")
    if fc.empty:
        return pd.DataFrame()

    # Check for snapshots
    has_snapshots = store.table_exists("forecast_snapshots")
    n_snaps = 0
    if has_snapshots:
        n_snaps = store.read_df("forecast_snapshots")["snapshot_date"].nunique()

    cols = list(fc.columns)
    col_year = cols[0]
    col_mean = cols[3]
    col_min = cols[2]
    col_max = cols[4]

    results = []
    for code in fc["ts_code"].unique():
        sf = fc[fc["ts_code"] == code].sort_values(col_year)
        if len(sf) < 2:
            continue

        name = sf["name"].iloc[0]
        label = sf["sector_label"].iloc[0] if "sector_label" in sf.columns else ""
        sector = _map_sector(name, label)

        latest = sf.iloc[-1]
        prior = sf.iloc[-2]
        try:
            eps = float(latest[col_mean])
            eps_prior = float(prior[col_mean])
            eps_min = float(latest[col_min])
            eps_max = float(latest[col_max])
        except (ValueError, TypeError, KeyError):
            continue

        if eps_prior <= 0:
            continue

        # Consensus growth rate
        growth = (eps / eps_prior) - 1

        # Forward PE = sector PE (from Tushare)
        sector_pe = _get_sector_pe(sector)
        forward_pe = sector_pe

        # PEG
        peg = forward_pe / (growth * 100) if growth > 0 else float("inf")
        if peg <= 0 or peg > 100:
            peg = 100.0

        # Dispersion — consensus quality weight
        dispersion = (eps_max - eps_min) / eps if eps > 0 else 1.0
        quality = max(0, 1.0 - min(dispersion, 2.0))

        # ── Revenue & Profit Quality (hybrid PEG) ──
        symbol = sf["symbol"].iloc[0]
        fin = _get_financial_metrics(symbol)
        rev_growth = fin.get("revenue_growth")
        net_margin = fin.get("net_margin")
        profit_quality = _compute_profit_quality(net_margin)

        # PRG proxy: revenue growth alone (simplified PR-to-Growth)
        # Higher revenue growth = higher PRG signal
        prg_signal = (rev_growth if rev_growth else growth) - 0.15  # 15% baseline
        # Normalize: 0% growth = -1 signal, 50%+ = +1 signal
        prg_signal = max(-1.0, min(1.0, prg_signal * 3))

        # ΔPEG from snapshots (if available)
        delta_peg, growth_rev = 0.0, 0.0
        if use_revisions and has_snapshots and n_snaps >= 2:
            delta_peg, growth_rev = _compute_revisions(code, sf, cols)

        results.append({
            "ts_code": code, "name": name, "sector": sector,
            "consensus_eps": eps, "consensus_growth": growth,
            "forward_pe": forward_pe, "peg": peg,
            "dispersion": dispersion, "quality": quality,
            "delta_peg": delta_peg, "growth_revision": growth_rev,
            "revenue_growth": rev_growth or 0.0,
            "net_margin": net_margin or 0.0,
            "profit_quality": profit_quality,
            "prg_signal": prg_signal,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Filter out infinite PEG
    df = df[df["peg"] < 50].copy()

    # Cross-sectional z-scores
    for col in ["peg", "consensus_growth", "quality"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[col + "_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    # PEG signal: low PEG = good, high growth = good, high quality = good
    df["peg_signal"] = (
        -0.40 * df["peg_z"] +              # Lower PEG → higher signal
         0.40 * df["consensus_growth_z"] +  # Higher growth → higher signal
         0.20 * df["quality_z"]             # Lower dispersion → higher signal
    )

    # PRG z-score (revenue growth cross-sectional)
    mu_rg, sigma_rg = df["revenue_growth"].mean(), df["revenue_growth"].std()
    df["prg_z"] = (df["revenue_growth"] - mu_rg) / sigma_rg if sigma_rg > 0 else 0.0
    df["prg_signal"] = 0.7 * df["prg_z"].clip(-3, 3)

    # HYBRID signal: profit-quality blended PEG + PRG
    # quality → 1.0: almost pure PEG (mature companies)
    # quality → 0.0: almost pure PRG (early stage companies)
    df["quality_weight"] = df["profit_quality"]  # for display
    df["hybrid_signal"] = (
        df["profit_quality"] * df["peg_signal"] +
        (1 - df["profit_quality"]) * df["prg_signal"]
    )

    # Add revision bonus if available
    if n_snaps >= 2:
        mu_r, sigma_r = df["delta_peg"].mean(), df["delta_peg"].std()
        if sigma_r > 0:
            df["delta_peg_z"] = (df["delta_peg"] - mu_r) / sigma_r
            # ΔPEG < 0 → positive signal; ΔPEG > 0 → negative signal
            df["peg_signal"] += -0.30 * df["delta_peg_z"]
        mu_g, sigma_g = df["growth_revision"].mean(), df["growth_revision"].std()
        if sigma_g > 0:
            df["growth_revision_z"] = (df["growth_revision"] - mu_g) / sigma_g
            df["peg_signal"] += 0.30 * df["growth_revision_z"]

    return df


# ── Revenue & Profit Quality ─────────────────────────────────────────

# Cache for THS financial data (avoid repeated API calls)
_financial_cache: dict[str, dict] = {}


def _get_financial_metrics(symbol: str) -> dict:
    """Fetch latest annual revenue, net profit, and margin from THS.

    Returns dict with keys: revenue_growth, net_margin, revenue, net_profit.
    Cached in-memory to avoid repeated API calls.
    """
    if symbol in _financial_cache:
        return _financial_cache[symbol]

    try:
        import akshare as ak
        df = ak.stock_financial_abstract_ths(symbol=symbol)
        if df is None or df.empty:
            _financial_cache[symbol] = {}
            return {}

        # THS columns at fixed positions:
        # [0]=报告期, [1]=归母净利润, [2]=归母净利润同比, [5]=营业总收入,
        # [6]=营业总收入同比, [12]=销售净利率, [13]=销售毛利率
        date_col = df.columns[0]
        profit_growth_col = df.columns[2]
        rev_col = df.columns[5]
        rev_growth_col = df.columns[6]
        margin_col = df.columns[12]

        # Latest annual data (exclude quarterly 03-31/06-30/09-30)
        annual = df[~df[date_col].astype(str).str.contains("03-31|06-30|09-30", na=False)]
        if annual.empty:
            annual = df

        # THS data is sorted OLDEST-first. Scan from newest (last row) backwards.
        n = len(annual)
        rev_growth = None
        margin = None
        for idx in range(n - 1, -1, -1):  # newest → oldest
            row = annual.iloc[idx]
            rg = _parse_pct(str(row[rev_growth_col]))
            mg = _parse_pct(str(row[margin_col]))
            if rg is not None and rev_growth is None:
                rev_growth = rg
            if mg is not None and margin is None:
                margin = mg
            if rev_growth is not None and margin is not None:
                break

        # Fallback: use net profit growth for revenue growth
        if rev_growth is None:
            for idx in range(n - 1, -1, -1):
                pg = _parse_pct(str(annual.iloc[idx][profit_growth_col]))
                if pg is not None:
                    rev_growth = pg
                    break

        # Fallback: use most recent row (newest data) even if quarterly
        if rev_growth is None or margin is None:
            latest_any = df.iloc[-1]
            if rev_growth is None:
                rev_growth = _parse_pct(str(latest_any[rev_growth_col]))
                if rev_growth is None:
                    rev_growth = _parse_pct(str(latest_any[profit_growth_col]))
            if margin is None:
                margin = _parse_pct(str(latest_any[margin_col]))

        result = {
            "revenue_growth": rev_growth if rev_growth is not None else 0.0,
            "net_margin": margin if margin is not None else 0.0,
        }
        _financial_cache[symbol] = result
        return result

    except Exception:
        _financial_cache[symbol] = {}
        return {}


def _parse_pct(raw: str) -> float | None:
    """Parse percentage string like '12.02%', '-5.30%', 'False'."""
    if not raw or raw in ("False", "None", "nan", ""):
        return None
    try:
        clean = raw.replace("%", "").replace("+", "").strip()
        return float(clean) / 100.0
    except ValueError:
        return None


def _compute_profit_quality(net_margin: float | None) -> float:
    """Automatic quality weight: 0 = pure revenue (early stage), 1 = pure profit (mature).

    quality = clamp(net_margin × 10, 0, 1)

    net_margin = 0%   → quality = 0.0  → pure PRG (revenue growth is the ONLY signal)
    net_margin = 5%   → quality = 0.5  → equal blend
    net_margin = 10%  → quality = 1.0  → pure PEG (profit growth is the signal)
    net_margin = 15%  → quality = 1.0  → pure PEG (capped at 1.0)
    """
    if net_margin is None:
        return 0.5  # unknown → equal blend
    return max(0.0, min(1.0, net_margin * 10))


def _get_sector_pe(sector: str) -> float:
    """Get current sector PE from Tushare sw_industry_pe_daily data."""
    if not store.table_exists("sw_industry_pe_daily"):
        return 30.0  # fallback

    try:
        pe_df = store.read_df("sw_industry_pe_daily")
        if pe_df.empty:
            return 30.0
    except Exception:
        return 30.0

    # Map sector to SW index code
    SECTOR_TO_SW = {
        "sw_semiconductor": "801081.SI",
        "sw_electronics": "801080.SI",
        "sw_computer_equipment": "801101.SI",
        "sw_telecom_equipment": "801102.SI",
        "sw_media": "801760.SI",
        "sw_pharma": "801150.SI",
        "sw_national_defense": "801740.SI",
        "sw_nonferrous": "801050.SI",
        "sw_auto": "801110.SI",
        "sw_chemical": "801030.SI",
        "sw_mechanical_equipment": "801070.SI",
        "sw_electric_equipment": "801730.SI",
    }

    sw_code = SECTOR_TO_SW.get(sector)
    if sw_code is None:
        return 30.0

    # Tushare sw_daily stores PE in 'pe' column
    sec_pe = pe_df[pe_df["ts_code"] == sw_code]
    if sec_pe.empty:
        return 30.0

    pe_col = "pe" if "pe" in sec_pe.columns else None
    if pe_col is None:
        # Try to find PE column
        for c in sec_pe.columns:
            if "pe" in c.lower():
                pe_col = c
                break
    if pe_col is None:
        return 30.0

    pe_values = pd.to_numeric(sec_pe[pe_col], errors="coerce").dropna()
    if pe_values.empty:
        return 30.0

    # Use latest PE value
    latest_pe = float(pe_values.iloc[-1])
    return max(3.0, min(200.0, latest_pe))


def _compute_revisions(
    sf: pd.DataFrame, fc_cols: list[str]
) -> tuple[float, float]:
    """Compute ΔPEG from forecast snapshots.

    Returns (delta_peg, growth_revision).
    """
    try:
        snaps = store.read_df("forecast_snapshots")
        code = sf["ts_code"].iloc[0]
        stock_snaps = snaps[snaps["ts_code"] == code]

        if len(stock_snaps) < 2:
            return 0.0, 0.0

        stock_snaps = stock_snaps.sort_values("snapshot_date")
        oldest = stock_snaps.iloc[0]
        newest = stock_snaps.iloc[-1]

        col_mean = fc_cols[3]
        eps_old = float(oldest[col_mean])
        eps_new = float(newest[col_mean])

        if eps_old <= 0:
            return 0.0, 0.0

        growth_revision = (eps_new / eps_old) - 1

        # ΔPEG ≈ -Δgrowth (simplified: P/E doesn't change much over 4 weeks)
        delta_peg = -growth_revision

        return delta_peg, growth_revision
    except Exception:
        return 0.0, 0.0


# ── Sector Aggregation ──────────────────────────────────────────────

def aggregate_to_sectors(peg_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stock-level PEG signals to sector level."""
    if peg_df.empty:
        return pd.DataFrame()

    # Only keep mapped sectors
    peg_df = peg_df[peg_df["sector"] != "other"].copy()

    agg = peg_df.groupby("sector").agg(
        signal_score=("hybrid_signal", "mean"),
        peg_signal=("peg_signal", "mean"),
        prg_signal=("prg_signal", "mean"),
        stock_count=("ts_code", "count"),
        avg_peg=("peg", "mean"),
        avg_growth=("consensus_growth", "mean"),
        avg_quality=("quality", "mean"),
        avg_profit_quality=("profit_quality", "mean"),
        avg_revenue_growth=("revenue_growth", "mean"),
        mean_delta_peg=("delta_peg", "mean"),
        mean_growth_rev=("growth_revision", "mean"),
    ).reset_index()

    return agg


def print_factor_report(peg_df: pd.DataFrame):
    """Print a human-readable PEG factor report."""
    if peg_df.empty:
        return

    print("\n" + "=" * 60)
    print("PEG Factor Report")
    print("=" * 60)

    top = peg_df.nlargest(10, "hybrid_signal")
    print("\nTop 10 by Hybrid signal:")
    print(f"  {'Name':<10s} {'Sector':<22s} {'PEG':>5s} {'G%':>6s} "
          f"{'RevG%':>6s} {'Margin':>7s} {'Q':>5s} {'Hybrid':>7s} {'PEGsig':>7s}")
    for _, r in top.iterrows():
        print(f"  {r['name']:<10s} {r['sector']:<22s} {r['peg']:>5.1f} {r['consensus_growth']:>5.1%} "
              f"{r['revenue_growth']:>5.1%} {r['net_margin']:>6.1%} "
              f"{r['profit_quality']:>4.2f} {r['hybrid_signal']:>+7.3f} {r['peg_signal']:>+7.3f}")

    print(f"\n  N stocks: {len(peg_df)}")
    print(f"  Median PEG: {peg_df['peg'].median():.1f}")
    print(f"  Median Growth: {peg_df['consensus_growth'].median():.1%}")
    print(f"  Median Profit Quality: {peg_df['profit_quality'].median():.2f}")
    print(f"  Sectors: {peg_df['sector'].nunique()}")
