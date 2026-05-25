"""Dispersion Factor — consensus disagreement as opportunity fingerprint.

Philosophy:
    Extreme analyst dispersion is not "noise" — it is the MAP of where
    expectation gaps exist. When conservatism dominates, the gap between
    bulls and bears is wide. The moment that gap NARROWS with consensus
    moving UP is the moment the market starts pricing the gap.

The Four Quadrants:
                              Consensus Direction
                              Rising           Falling
    Dispersion  Narrowing     Q1: FORMING      Q3: DETERIORATING
    Trend       Widening      Q2: DEBATING      Q4: PANICKING

    Q1 = STRONGEST BUY:  conservatives capitulating, consensus forming upward
    Q2 = WATCH:           debate intensifying, bulls getting bolder but bears not convinced
    Q3 = SELL:            optimists retreating, consensus converging downward
    Q4 = STRONGEST SELL:  panic spreading, everyone running for the exit

Requires: 4+ forecast_snapshots (weekly snapshots of THS analyst consensus)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"
store = DuckDBStore(DB_PATH)


class CoveragePhase(str, Enum):
    """Analyst coverage lifecycle phase."""
    PRE_COVERAGE = "pre_coverage"         # 0 analysts — pure narrative, no data
    FIRST_TOUCH = "first_touch"           # 1-2 analysts — brave first movers
    EXPANDING = "expanding"               # 3-9 analysts — coverage accelerating
    ESTABLISHED = "established"           # 10+ analysts — mature coverage
    DECLINING = "declining"               # Coverage shrinking — analysts leaving


class DispersionQuadrant(str, Enum):
    CONSENSUS_FORMING = "consensus_forming"         # Q1: narrowing + rising
    DEBATE_INTENSIFYING = "debate_intensifying"      # Q2: widening + rising
    CONSENSUS_DETERIORATING = "consensus_deteriorating"  # Q3: narrowing + falling
    PANIC_SPREADING = "panic_spreading"              # Q4: widening + falling
    INSUFFICIENT_DATA = "insufficient_data"          # Not enough snapshots


@dataclass
class DispersionSignal:
    """Output of dispersion factor for a single stock."""

    ts_code: str
    name: str
    sector_label: str

    # Current state
    n_snapshots: int
    current_consensus: float        # Latest mean EPS estimate
    current_dispersion: float       # Latest (max-min)/mean
    current_n_analysts: int         # Latest analyst count

    # Trends (over lookback window)
    delta_consensus: float           # % change in consensus mean
    delta_dispersion: float          # change in dispersion ratio
    delta_analysts: int              # change in analyst count
    coverage_velocity: float = 0.0   # analysts added per snapshot (加速度)
    coverage_phase: CoveragePhase = CoveragePhase.PRE_COVERAGE

    # Signal
    quadrant: DispersionQuadrant = DispersionQuadrant.INSUFFICIENT_DATA
    signal_score: float = 0.0
    signal_confidence: float = 0.0

    # Interpretation
    narrative: str = ""


def compute_dispersion_signals(lookback_snapshots: int = 4) -> list[DispersionSignal]:
    """Compute dispersion signals for all stocks with sufficient snapshots.

    Args:
        lookback_snapshots: Number of snapshots to use for trend window.

    Returns:
        List of DispersionSignal, one per stock with sufficient data.
    """
    if not store.table_exists("forecast_snapshots"):
        return []

    snaps = store.read_df("forecast_snapshots")
    if snaps.empty:
        return []

    # THS columns are always at fixed positions in the snapshot table:
    #   [0] = 年度 (year)
    #   [1] = 预测人数 (analyst count)
    #   [2] = 最小值 (min EPS)
    #   [3] = 均值 (consensus mean EPS)
    #   [4] = 最大值 (max EPS)
    #   [5] = 行业平均数 (industry avg EPS)
    #   [6+] = metadata (snapshot_date, ts_code, symbol, name, ...)
    cols = list(snaps.columns)
    col_year = cols[0]
    col_n = cols[1]
    col_min = cols[2]
    col_mean = cols[3]
    col_max = cols[4]
    col_ind = cols[5]

    # For each (stock, forecast_year), track consensus over snapshots
    signals = []

    for code in snaps["ts_code"].unique():
        stock_snaps = snaps[snaps["ts_code"] == code].sort_values("snapshot_date")
        # Require at least N DISTINCT snapshot dates (not just N rows)
        unique_dates = stock_snaps["snapshot_date"].nunique()
        if unique_dates < lookback_snapshots:
            continue

        name = stock_snaps["name"].iloc[0]
        label = stock_snaps["sector_label"].iloc[0] if "sector_label" in stock_snaps.columns else ""

        # Use the latest forecast year (most forward-looking) across snapshots
        latest_year = str(stock_snaps[col_year].iloc[-1])
        year_snaps = stock_snaps[stock_snaps[col_year] == latest_year]
        if year_snaps["snapshot_date"].nunique() < lookback_snapshots:
            # Not enough unique dates for this year — try all years
            year_snaps = stock_snaps
            if year_snaps["snapshot_date"].nunique() < lookback_snapshots:
                continue

        year_snaps = year_snaps.sort_values("snapshot_date")

        # Compute consensus mean and dispersion at each snapshot
        means = []
        disps = []
        ns = []

        for _, row in year_snaps.iterrows():
            try:
                mean = float(row[col_mean])
                lo = float(row[col_min])
                hi = float(row[col_max])
                n = int(row[col_n])
            except (ValueError, TypeError, KeyError):
                continue

            if mean <= 0:
                continue

            disp = (hi - lo) / mean
            means.append(mean)
            disps.append(disp)
            ns.append(n)

        if len(means) < lookback_snapshots:
            continue

        # Trends: compare latest vs oldest in lookback window
        means_arr = np.array(means[-lookback_snapshots:])
        disps_arr = np.array(disps[-lookback_snapshots:])
        ns_arr = np.array(ns[-lookback_snapshots:])

        delta_mean = (means_arr[-1] - means_arr[0]) / means_arr[0] if means_arr[0] > 0 else 0.0
        delta_disp = disps_arr[-1] - disps_arr[0]
        delta_n = int(ns_arr[-1] - ns_arr[0])

        # Coverage velocity: avg new analysts per snapshot
        coverage_velocity = delta_n / max(lookback_snapshots - 1, 1)

        # Coverage phase classification
        current_n = int(ns_arr[-1])
        if current_n == 0:
            phase = CoveragePhase.PRE_COVERAGE
        elif current_n <= 2:
            phase = CoveragePhase.FIRST_TOUCH
        elif delta_n > 2:
            phase = CoveragePhase.EXPANDING      # Rapidly gaining coverage
        elif current_n >= 10:
            phase = CoveragePhase.ESTABLISHED
        elif delta_n < -2:
            phase = CoveragePhase.DECLINING
        else:
            phase = CoveragePhase.EXPANDING if current_n < 10 else CoveragePhase.ESTABLISHED

        # ── Quadrant classification ──
        mean_rising = delta_mean > 0.01     # 1%+ change = rising
        mean_falling = delta_mean < -0.01   # -1%+ change = falling
        disp_narrowing = delta_disp < -0.02  # 2pp+ drop = narrowing
        disp_widening = delta_disp > 0.02    # 2pp+ rise = widening

        if disp_narrowing and mean_rising:
            quadrant = DispersionQuadrant.CONSENSUS_FORMING
            # Strongest buy: consensus forming upward
            # Signal = convergence speed × consensus momentum
            signal_score = min(1.0, abs(delta_mean) * 5 + abs(delta_disp) * 3)
            signal_confidence = min(1.0, abs(delta_disp) * 4 + (1.0 - disps_arr[-1]))
            narrative = (
                f"Analyst consensus FORMING upward: mean +{delta_mean:.1%}, "
                f"dispersion {delta_disp:+.1%}. "
                f"Conservatives being proven wrong by data. "
                f"{'Coverage RISING — more analysts joining the bull case.' if delta_n > 0 else 'Coverage stable.'}"
            )

        elif disp_widening and mean_rising:
            quadrant = DispersionQuadrant.DEBATE_INTENSIFYING
            signal_score = delta_mean * 2  # Weak positive: bulls getting bolder but bears not convinced
            signal_confidence = 0.3 + min(0.4, abs(delta_mean) * 2)
            narrative = (
                f"Analyst debate INTENSIFYING: mean +{delta_mean:.1%}, "
                f"but dispersion WIDENING ({delta_disp:+.1%}). "
                f"Bulls raising estimates but bears not convinced — information asymmetry peak. "
                f"This is the SETUP before consensus forms."
            )

        elif disp_narrowing and mean_falling:
            quadrant = DispersionQuadrant.CONSENSUS_DETERIORATING
            signal_score = max(-1.0, delta_mean * 5 - abs(delta_disp) * 2)
            signal_confidence = min(1.0, abs(delta_mean) * 4)
            narrative = (
                f"Analyst consensus DETERIORATING: mean {delta_mean:.1%}, "
                f"dispersion narrowing ({delta_disp:+.1%}). "
                f"Optimists retreating — consensus converging downward. AVOID."
            )

        elif disp_widening and mean_falling:
            quadrant = DispersionQuadrant.PANIC_SPREADING
            signal_score = -min(1.0, abs(delta_mean) * 5 + abs(delta_disp) * 3)
            signal_confidence = min(1.0, abs(delta_disp) * 4)
            narrative = (
                f"PANIC spreading: mean {delta_mean:.1%}, "
                f"dispersion WIDENING ({delta_disp:+.1%}). "
                f"Everyone heading for the exit. STRONGEST SELL."
            )

        else:
            continue  # No clear quadrant — insufficient signal

        # Coverage acceleration bonus
        # When analysts are RAPIDLY joining + consensus is rising = strong positive
        if phase == CoveragePhase.EXPANDING and mean_rising:
            signal_score += 0.15 * min(1.0, coverage_velocity / 3.0)
            signal_confidence += 0.1
            narrative += (
                f" Coverage ACCELERATING (+{coverage_velocity:.1f}/snap). "
                f"Phase: {phase.value}. More analysts entering = attention compounding."
            )
        elif phase == CoveragePhase.FIRST_TOUCH and mean_rising:
            narrative += (
                f" EARLY STAGE: only {current_n} analysts. "
                f"Extreme information asymmetry — biggest alpha potential, highest risk."
            )

        # Cap scores
        signal_score = max(-1.0, min(1.0, signal_score))
        signal_confidence = max(0.0, min(1.0, signal_confidence))

        signals.append(DispersionSignal(
            ts_code=code,
            name=name,
            sector_label=label,
            n_snapshots=len(year_snaps),
            current_consensus=means_arr[-1],
            current_dispersion=disps_arr[-1],
            current_n_analysts=int(ns_arr[-1]),
            delta_consensus=delta_mean,
            delta_dispersion=delta_disp,
            delta_analysts=delta_n,
            coverage_velocity=coverage_velocity,
            coverage_phase=phase,
            quadrant=quadrant,
            signal_score=signal_score,
            signal_confidence=signal_confidence,
            narrative=narrative,
        ))

    return sorted(signals, key=lambda s: s.signal_score, reverse=True)


# ── Sector aggregation ──────────────────────────────────────────────

# Industry → Sector mapping (reuse from peg_factor)
INDUSTRY_TO_SECTOR = {
    "半导体": "sw_semiconductor", "芯片": "sw_semiconductor",
    "电子": "sw_electronics", "元器件": "sw_electronics",
    "光学": "sw_electronics", "PCB": "sw_electronics",
    "面板": "sw_electronics", "显示": "sw_electronics",
    "光模块": "sw_telecom_equipment", "光器件": "sw_telecom_equipment",
    "光通信": "sw_telecom_equipment",
    "计算机": "sw_computer_equipment", "软件": "sw_computer_equipment",
    "IT": "sw_computer_equipment", "数据": "sw_computer_equipment",
    "通信": "sw_telecom_equipment", "网络": "sw_telecom_equipment",
    "传媒": "sw_media", "互联网": "sw_media", "平台": "sw_media",
    "金融": "sw_media", "办公": "sw_media",
    "医药": "sw_pharma",
    "军工": "sw_national_defense", "航天": "sw_national_defense",
    "航空": "sw_national_defense",
    "有色": "sw_nonferrous", "稀土": "sw_nonferrous",
    "汽车": "sw_auto", "驾驶": "sw_auto",
    "化工": "sw_chemical", "材料": "sw_chemical",
    "机械": "sw_mechanical_equipment", "制造": "sw_mechanical_equipment",
    "电气": "sw_electric_equipment", "新能源": "sw_electric_equipment",
    "电池": "sw_electric_equipment", "光伏": "sw_electric_equipment",
    "存储": "sw_semiconductor",
}


def _map_to_sector(label: str, name: str = "") -> str:
    text = str(label) + str(name)
    for kw, sector in sorted(INDUSTRY_TO_SECTOR.items(), key=lambda x: -len(x[0])):
        if kw in text:
            return sector
    return "other"


def aggregate_dispersion_to_sectors(
    signals: list[DispersionSignal],
) -> pd.DataFrame:
    """Aggregate dispersion signals to sector level."""
    if not signals:
        return pd.DataFrame()

    rows = []
    for s in signals:
        sector = _map_to_sector(s.sector_label, s.name)
        rows.append({
            "sector": sector,
            "ts_code": s.ts_code,
            "name": s.name,
            "quadrant": s.quadrant.value,
            "signal_score": s.signal_score,
            "signal_confidence": s.signal_confidence,
            "delta_consensus": s.delta_consensus,
            "delta_dispersion": s.delta_dispersion,
            "delta_analysts": s.delta_analysts,
            "current_dispersion": s.current_dispersion,
            "current_consensus": s.current_consensus,
            "narrative": s.narrative,
        })

    df = pd.DataFrame(rows)

    # Sector average
    sector_agg = df.groupby("sector").agg(
        avg_signal=("signal_score", "mean"),
        avg_confidence=("signal_confidence", "mean"),
        stock_count=("ts_code", "count"),
        q1_count=("quadrant", lambda x: (x == "consensus_forming").sum()),
        q2_count=("quadrant", lambda x: (x == "debate_intensifying").sum()),
        q3_count=("quadrant", lambda x: (x == "consensus_deteriorating").sum()),
        q4_count=("quadrant", lambda x: (x == "panic_spreading").sum()),
        avg_delta_consensus=("delta_consensus", "mean"),
        avg_delta_dispersion=("delta_dispersion", "mean"),
    ).reset_index()

    return sector_agg


# ── Report ──────────────────────────────────────────────────────────

def print_dispersion_report(signals: list[DispersionSignal]):
    """Print a human-readable dispersion factor report."""
    if not signals:
        print("  No dispersion signals available — need 4+ forecast snapshots.")
        print(f"  Run collect_snapshots.py weekly to accumulate.")
        return

    q1 = [s for s in signals if s.quadrant == DispersionQuadrant.CONSENSUS_FORMING]
    q2 = [s for s in signals if s.quadrant == DispersionQuadrant.DEBATE_INTENSIFYING]
    q3 = [s for s in signals if s.quadrant == DispersionQuadrant.CONSENSUS_DETERIORATING]
    q4 = [s for s in signals if s.quadrant == DispersionQuadrant.PANIC_SPREADING]

    print(f"\n{'=' * 65}")
    print(f"  DISPERSION FACTOR REPORT")
    print(f"  {len(signals)} stocks with valid 4+ week revision sequences")
    print(f"{'=' * 65}")

    # Q1: The opportunity zone
    if q1:
        print(f"\n  [Q1] CONSENSUS FORMING ({len(q1)} stocks) — STRONGEST BUY")
        print(f"  " + "-" * 58)
        print(f"  {'Stock':<12s} {'Signal':>7s} {'dCons':>8s} {'dDisp':>8s} {'dN':>5s} {'Phase':>14s}")
        for s in sorted(q1, key=lambda x: -x.signal_score):
            print(f"  {s.name:<12s} {s.signal_score:>+6.3f} {s.delta_consensus:>+7.1%} "
                  f"{s.delta_dispersion:>+7.1%} {s.delta_analysts:>+4d} {s.coverage_phase.value:>14s}")
            print(f"    -> {s.narrative[:110]}")

    # Q2: The watch zone (debate before consensus)
    if q2:
        print(f"\n  [Q2] DEBATE INTENSIFYING ({len(q2)} stocks) — WATCH (setup before Q1)")
        print(f"  " + "-" * 58)
        for s in sorted(q2, key=lambda x: -x.signal_score):
            print(f"  {s.name:<12s} {s.signal_score:>+6.3f} {s.delta_consensus:>+7.1%} "
                  f"{s.delta_dispersion:>+7.1%} {s.delta_analysts:>+8d} {s.current_dispersion:>8.1%}")
            print(f"    -> {s.narrative[:100]}")

    # Q3: Deteriorating
    if q3:
        print(f"\n  [Q3] CONSENSUS DETERIORATING ({len(q3)} stocks) — SELL")
        for s in sorted(q3, key=lambda x: x.signal_score):
            print(f"  {s.name:<12s} {s.signal_score:>+6.3f} {s.delta_consensus:>+7.1%} "
                  f"{s.delta_dispersion:>+7.1%} {s.delta_analysts:>+8d}")

    # Q4: Panic
    if q4:
        print(f"\n  [Q4] PANIC SPREADING ({len(q4)} stocks) — STRONGEST SELL")
        for s in sorted(q4, key=lambda x: x.signal_score):
            print(f"  {s.name:<12s} {s.signal_score:>+6.3f} {s.delta_consensus:>+7.1%} "
                  f"{s.delta_dispersion:>+7.1%} {s.delta_analysts:>+8d}")

    # Sector summary
    print(f"\n  SECTOR SUMMARY:")
    sector_agg = aggregate_dispersion_to_sectors(signals)
    if not sector_agg.empty:
        for _, r in sector_agg.sort_values("avg_signal", ascending=False).iterrows():
            if r["sector"] == "other":
                continue
            print(f"  {r['sector']:<30s} signal={r['avg_signal']:+.3f} "
                  f"Q1={int(r['q1_count'])} Q2={int(r['q2_count'])} "
                  f"Q3={int(r['q3_count'])} Q4={int(r['q4_count'])} "
                  f"({int(r['stock_count'])} stocks)")


if __name__ == "__main__":
    signals = compute_dispersion_signals(lookback_snapshots=4)
    print_dispersion_report(signals)
