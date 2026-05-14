"""Head-to-head stock comparison — PEG + expectation gap.

Usage:  uv run python scripts/compare_stocks.py
"""

import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore
from factors.metadata_annotator import _detect_peg_trap, PegTrapType

store = DuckDBStore("data/trading_god.duckdb")

STOCKS = [
    ("603986", "兆易创新", "存储芯片/NOR Flash/MCU", "sw_semiconductor"),
    ("688041", "海光信息", "服务器CPU/GPU国产替代", "sw_semiconductor"),
]

# Get PE data once
pe_df = store.read_df("sw_industry_pe_daily")
semi_pe = pe_df[pe_df["ts_code"] == "801081.SI"]
pe_col = "pe" if "pe" in semi_pe.columns else [c for c in semi_pe.columns if "pe" in c.lower()][0]
pe_vals = pd.to_numeric(semi_pe[pe_col], errors="coerce").dropna()
pe_now = float(pe_vals.iloc[-1])
pe_med = float(pe_vals.median())
pe_p25 = float(pe_vals.quantile(0.25))
pe_p75 = float(pe_vals.quantile(0.75))


def analyze_one(code, name, biz, sector):
    print(f"\n{'=' * 62}")
    print(f"  {name} ({code}.SH) — {biz}")
    print(f"{'=' * 62}")

    # Fetch forecast
    time.sleep(2)
    try:
        fc = ak.stock_profit_forecast_ths(symbol=code)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if fc is None or fc.empty:
        print("  No forecast data")
        return

    cols = list(fc.columns)
    fc = fc.sort_values(cols[0])

    # 1. Consensus
    print("\n  [1] Consensus EPS")
    print("  " + "-" * 52)
    growth_rates = []
    for i, (_, r) in enumerate(fc.iterrows()):
        y, n, lo, mean, hi, ind = r.iloc[0], int(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4]), float(r.iloc[5])
        disp = (hi - lo) / mean if mean > 0 else 0
        premium = (mean - ind) / ind if ind > 0 else 0
        if i > 0:
            prev_mean = float(fc.iloc[i - 1].iloc[3])
            yr_growth = (mean / prev_mean) - 1
            growth_rates.append(yr_growth)
            grow_str = f"YoY={yr_growth:.1%}"
        else:
            grow_str = ""
        print(f"  {y}: EPS={mean:.2f} [{lo:.2f}-{hi:.2f}], n={n:>2d}, "
              f"disp={disp:.1%}, prem={premium:.1%}, {grow_str}")

    latest = fc.iloc[-1]
    prior = fc.iloc[-2]
    eps_next = float(latest.iloc[3])
    eps_curr = float(prior.iloc[3])
    growth = (eps_next / eps_curr) - 1
    n_analysts = int(latest.iloc[1])
    disp = (float(latest.iloc[4]) - float(latest.iloc[2])) / eps_next

    # Growth trend
    if len(growth_rates) >= 2:
        g1, g2 = growth_rates[-1], growth_rates[-2]
        if g1 > g2 + 0.05:
            print(f"  >> Growth ACCELERATING: {g2:.1%} -> {g1:.1%}")
        elif g1 < g2 - 0.05:
            print(f"  >> Growth DECELERATING: {g2:.1%} -> {g1:.1%}")
        else:
            print(f"  >> Growth STABLE: ~{g1:.1%}")

    # 2. PEG
    print(f"\n  [2] PEG Valuation (Semiconductor PE={pe_now:.1f}, median={pe_med:.1f})")
    print("  " + "-" * 52)
    peg = pe_now / (growth * 100) if growth > 0 else 999

    pe_pos = (pe_now - pe_vals.min()) / (pe_vals.max() - pe_vals.min()) * 100
    print(f"  Forward EPS: {eps_next:.2f} | Current EPS: {eps_curr:.2f}")
    print(f"  Implied Growth: {growth:.1%}")
    print(f"  PEG = {pe_now:.1f} / {growth*100:.1f} = {peg:.2f}")
    print(f"  Sector PE at {pe_pos:.0f}th percentile (lower = cheaper)")

    if peg < 0.8:
        print(f"  >>> STRONG BUY: PEG={peg:.2f} < 0.8 — market underpricing growth")
    elif peg < 1.2:
        print(f"  >>> BUY: PEG={peg:.2f} reasonable — margin of safety exists")
    elif peg < 2.0:
        print(f"  >>> HOLD: PEG={peg:.2f} fair — growth fully priced")
    else:
        print(f"  >>> CAUTION: PEG={peg:.2f} > 2.0 — aggressive expectations priced in")

    # Scenarios
    for pct, label in [(25, "bear"), (50, "median"), (75, "bull")]:
        pe_s = float(pe_vals.quantile(pct / 100))
        peg_s = pe_s / (growth * 100) if growth > 0 else 999
        print(f"    {label:>8s} PE={pe_s:.1f}: PEG={peg_s:.2f}")

    # 3. Consensus quality
    print(f"\n  [3] Consensus Quality")
    print("  " + "-" * 52)

    # Dispersion analysis across years
    disps = []
    for _, r in fc.iterrows():
        lo, mean, hi = float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4])
        d = (hi - lo) / mean if mean > 0 else 0
        disps.append(d)

    if len(disps) >= 2:
        d_trend = "NARROWING" if disps[-1] < disps[-2] - 0.05 else ("WIDENING" if disps[-1] > disps[-2] + 0.05 else "STABLE")
    else:
        d_trend = "N/A"

    # Analyst count trend
    n_list = [int(r.iloc[1]) for _, r in fc.iterrows()]
    if len(n_list) >= 2:
        n_trend = "RISING" if n_list[-1] > n_list[-2] else ("FALLING" if n_list[-1] < n_list[-2] else "STABLE")
    else:
        n_trend = "N/A"

    print(f"  Analyst coverage: {n_analysts} (trend: {n_trend})")
    print(f"  Consensus dispersion: {disp:.1%} (trend: {d_trend})")
    print(f"  Industry premium: {(eps_next - float(latest.iloc[5])) / float(latest.iloc[5]):.1%}")

    if disp > 1.0:
        print(f"  >> EXTREME DISPERSION: analyst estimates span >100% of mean")
        print(f"  >> Consensus may be MEANINGLESS — extreme disagreement = information asymmetry")
    elif disp > 0.5:
        print(f"  >> HIGH DISPERSION: significant disagreement among analysts")
    elif disp < 0.20:
        print(f"  >> TIGHT consensus — high conviction")
    else:
        print(f"  >> MODERATE dispersion — healthy debate")

    # 4. PEG Trap
    print(f"\n  [4] PEG Trap Analysis")
    print("  " + "-" * 52)
    trap = _detect_peg_trap(sector, peg, growth, signal=0.5, dispersion=disp)

    if trap.trap_type != PegTrapType.NONE:
        print(f"  [!] {trap.headline}")
        print(f"  SCENARIO A: {trap.scenario_a[:120]}...")
        print(f"  SCENARIO B: {trap.scenario_b[:120]}...")
        print(f"  Model lean: {trap.model_lean:.0%} toward Scenario B")
        print(f"\n  EVIDENCE CHECKLIST:")
        for j, check in enumerate(trap.evidence_checklist):
            print(f"    [{j+1}] {check}")
    else:
        print(f"  No PEG trap detected — valuation within normal range.")

    return {
        "name": name,
        "code": code,
        "peg": peg,
        "growth": growth,
        "eps_next": eps_next,
        "n_analysts": n_analysts,
        "dispersion": disp,
        "trap_type": trap.trap_type.value,
    }


# ── Run ────────────────────────────────────────────────────────────

print("=" * 62)
print("  Stock Comparison: 兆易创新 vs 海光信息")
print(f"  Semiconductor PE: {pe_now:.1f} (median: {pe_med:.1f})")
print("=" * 62)

results = []
for code, name, biz, sector in STOCKS:
    r = analyze_one(code, name, biz, sector)
    if r:
        results.append(r)

# ── Head-to-head ───────────────────────────────────────────────────

if len(results) == 2:
    print(f"\n\n{'=' * 62}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 62}")
    print(f"  {'Metric':<25s} {'兆易创新':>16s} {'海光信息':>16s}")
    print("  " + "-" * 58)
    for label, key, fmt in [
        ("PEG", "peg", ".2f"),
        ("Implied Growth", "growth", ".1%"),
        ("Forward Consensus EPS", "eps_next", ".2f"),
        ("Analyst Coverage", "n_analysts", "d"),
        ("Consensus Dispersion", "dispersion", ".1%"),
    ]:
        v1 = results[0][key]
        v2 = results[1][key]
        if fmt == "d":
            s1, s2 = f"{int(v1)}", f"{int(v2)}"
        elif fmt == ".1%":
            s1, s2 = f"{v1:.1%}", f"{v2:.1%}"
        else:
            s1, s2 = f"{v1:.2f}", f"{v2:.2f}"
        print(f"  {label:<25s} {s1:>16s} {s2:>16s}")

    # Verdict
    print(f"\n  KEY DIFFERENCES:")
    print(f"  - Growth: 兆易创新 {results[0]['growth']:.1%} vs 海光信息 {results[1]['growth']:.1%}")
    print(f"  - Dispersion: 兆易创新 {results[0]['dispersion']:.0%} vs 海光信息 {results[1]['dispersion']:.0%}")
    print(f"  - Consensus quality: {'兆易创新 has EXTREME disagreement — higher risk' if results[0]['dispersion'] > 1.0 else '兆易创新 has moderate consensus'}")

print()
