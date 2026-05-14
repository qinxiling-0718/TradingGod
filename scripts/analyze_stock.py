"""Single stock deep analysis — expectation gap report.

Usage:  uv run python scripts/analyze_stock.py <stock_code>
Example: uv run python scripts/analyze_stock.py 002463
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


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "002463"
    analyze(code)


def analyze(code: str):
    # ── Load data ──
    fc = store.read_df("analyst_forecast")
    stock_fc = fc[fc["symbol"] == code].sort_values(fc.columns[0])
    name = stock_fc["name"].iloc[0] if not stock_fc.empty else code

    # Financial actuals
    time.sleep(2)
    try:
        fin = ak.stock_financial_abstract_ths(symbol=code)
    except Exception:
        fin = None

    # PE data
    pe_df = store.read_df("sw_industry_pe_daily")
    elec_pe = pe_df[pe_df["ts_code"] == "801080.SI"]

    # ── Print report ──
    print("=" * 62)
    print(f"  {name} ({code}.SZ) — AI PCB Expectation Gap Analysis")
    print("=" * 62)

    # 1. Analyst Forecast
    print("\n  [1] Consensus EPS Forecast (THS)")
    print("  " + "-" * 52)
    if stock_fc.empty:
        print("  No forecast data available.")
        return

    cols = list(stock_fc.columns)
    growth_rates = []
    for i, (_, r) in enumerate(stock_fc.iterrows()):
        y, n, lo, mean, hi, ind = r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3], r.iloc[4], r.iloc[5]
        disp = (hi - lo) / mean if mean > 0 else 0
        if i > 0:
            prev_mean = float(stock_fc.iloc[i - 1].iloc[3])
            yr_growth = (float(mean) / prev_mean) - 1
            growth_rates.append(yr_growth)
            grow_str = f"YoY={yr_growth:.1%}"
        else:
            grow_str = ""
        beats_ind = " > industry" if float(mean) > float(ind) else " < industry"
        print(f"  {y}: consensus={float(mean):.2f}, range=[{lo}-{hi}], "
              f"n={int(n):>2d}, disp={disp:.1%}, {grow_str}{beats_ind}")

    if len(growth_rates) >= 2:
        g1, g2 = growth_rates[-1], growth_rates[-2]
        if g1 > g2:
            print(f"  >> Growth ACCELERATING: {g2:.1%} -> {g1:.1%}")
        elif g1 < g2:
            print(f"  >> Growth DECELERATING: {g2:.1%} -> {g1:.1%}")
        else:
            print(f"  >> Growth STABLE: {g1:.1%}")

    # 2. Financial history
    print("\n  [2] Recent Financial Performance (THS)")
    print("  " + "-" * 52)
    if fin is not None and not fin.empty:
        fin_cols = list(fin.columns)
        date_col = fin_cols[0]
        eps_col = next((c for c in fin_cols if "每股收益" in c), fin_cols[7])
        rev_col = next((c for c in fin_cols if "营业总收入" in c), fin_cols[5])
        margin_col = next((c for c in fin_cols if "毛利率" in c), None)
        roe_col = next((c for c in fin_cols if "净资产收益率" in c and "摊薄" not in c), None)
        growth_col = next((c for c in fin_cols if "同比增长" in c and "净利润" in c and "扣非" not in c), None)

        # Show last 6 annual + most recent quarter
        annual = fin[~fin[date_col].astype(str).str.contains("03-31|06-30|09-30", na=False)]
        recent_annual = annual.head(4)
        tables = []
        for _, r in recent_annual.iterrows():
            d = str(r[date_col])[:10]
            eps = r[eps_col]
            rev = r[rev_col] if pd.notna(r[rev_col]) else "N/A"
            margin = r[margin_col] if margin_col and pd.notna(r[margin_col]) else "N/A"
            roe = r[roe_col] if roe_col and pd.notna(r[roe_col]) else "N/A"
            grow = r[growth_col] if growth_col and pd.notna(r[growth_col]) else "N/A"
            tables.append((d, eps, rev, margin, roe, grow))

        if tables:
            print(f"  {'Period':<14s} {'EPS':>8s} {'Revenue':>14s} {'Margin':>8s} {'ROE':>8s} {'YoY Profit':>12s}")
            for d, eps, rev, margin, roe, grow in tables:
                print(f"  {d:<14s} {str(eps):>8s} {str(rev):>14s} {str(margin):>8s} {str(roe):>8s} {str(grow):>12s}")
    else:
        print("  No financial data available.")

    # 3. PEG Analysis
    print("\n  [3] PEG Valuation Analysis")
    print("  " + "-" * 52)

    if not elec_pe.empty:
        pe_col = "pe" if "pe" in elec_pe.columns else [c for c in elec_pe.columns if "pe" in c.lower()][0]
        pe_vals = pd.to_numeric(elec_pe[pe_col], errors="coerce").dropna()
        pe_now = float(pe_vals.iloc[-1])
        pe_med = float(pe_vals.median())
        pe_p25 = float(pe_vals.quantile(0.25))
        pe_p75 = float(pe_vals.quantile(0.75))
        pe_min = float(pe_vals.min())
        pe_max = float(pe_vals.max())

        print(f"  Electronics Sector PE:")
        print(f"    Current: {pe_now:.1f}  |  Median: {pe_med:.1f}")
        print(f"    Range:   [{pe_min:.1f}, {pe_max:.1f}]")
        print(f"    IQR:     [{pe_p25:.1f}, {pe_p75:.1f}]")
        pe_position = (pe_now - pe_min) / (pe_max - pe_min) * 100 if pe_max > pe_min else 50
        print(f"    Position: {pe_position:.0f}th percentile (lower = cheaper)")

        # PEG for each forecast year
        print()
        latest = stock_fc.iloc[-1]
        prior = stock_fc.iloc[-2]
        eps_next = float(latest.iloc[3])
        eps_curr = float(prior.iloc[3])
        growth = (eps_next / eps_curr) - 1
        peg = pe_now / (growth * 100) if growth > 0 else 999

        print(f"  PEG Calculation:")
        print(f"    Consensus EPS (next year): {eps_next}")
        print(f"    Consensus EPS (current):   {eps_curr}")
        print(f"    Implied Growth Rate:       {growth:.1%}")
        print(f"    PEG = {pe_now:.1f} / {growth * 100:.1f} = {peg:.2f}")

        print()
        if peg < 0.8:
            print(f"    >>> STRONG BUY: PEG={peg:.2f} < 0.8")
            print(f"        Market is significantly underpricing growth.")
            print(f"        Expected gap is POSITIVE and WIDE.")
        elif peg < 1.2:
            print(f"    >>> BUY: PEG={peg:.2f} in 0.8~1.2")
            print(f"        Growth reasonably priced with some margin of safety.")
        elif peg < 2.0:
            print(f"    >>> HOLD: PEG={peg:.2f} in 1.2~2.0")
            print(f"        Growth fully priced — need acceleration for upside.")
        else:
            print(f"    >>> CAUTION: PEG={peg:.2f} > 2.0")
            print(f"        Market has already priced in aggressive growth expectations.")

        # Historical PEG context
        for pct, label in [(25, "bear case"), (50, "median"), (75, "bull case")]:
            pe_scenario = float(pe_vals.quantile(pct / 100))
            peg_scenario = pe_scenario / (growth * 100) if growth > 0 else 999
            print(f"    {label:>12s} (PE={pe_scenario:.1f}): PEG={peg_scenario:.2f}")

    # 4. Consensus Quality
    print("\n  [4] Consensus Quality & Analyst Behavior")
    print("  " + "-" * 52)

    for _, r in stock_fc.iterrows():
        y = r.iloc[0]
        n = int(r.iloc[1])
        lo, mean, hi = float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4])
        ind = float(r.iloc[5])
        disp = (hi - lo) / mean if mean > 0 else 0

        # Premium to industry
        premium = (mean - ind) / ind if ind > 0 else 0

        print(f"  {y}: {n} analysts covering, dispersion={disp:.1%}")
        print(f"       Consensus EPS={mean}, Industry Avg={ind}, Premium={premium:.1%}")

        if disp < 0.20:
            print(f"       >> TIGHT consensus — high conviction among analysts")
        elif disp < 0.50:
            print(f"       >> MODERATE dispersion — some debate but direction clear")
        else:
            print(f"       >> WIDE dispersion — significant disagreement, higher risk")

    # 5. Snapshots status
    print("\n  [5] Revision Sequence Status")
    print("  " + "-" * 52)
    sn = store.read_df("forecast_snapshots")
    if not sn.empty:
        stock_sn = sn[sn["symbol"] == code]
        n_snaps = len(stock_sn)
        print(f"  Snapshots accumulated: {n_snaps}")
        if n_snaps >= 4:
            print(f"  >> READY: Delta-PEG computation available")
        elif n_snaps >= 2:
            remaining = 4 - n_snaps
            print(f"  >> {remaining} more weeks until Delta-PEG activates")
        else:
            print(f"  >> First snapshot — baseline established")
            print(f"  >> 3 more weekly snapshots needed for revision tracking")
    else:
        print(f"  No snapshots yet — run collect_snapshots.py")

    # 6. Summary
    print()
    print("=" * 62)
    print("  EXPECTATION GAP SUMMARY")
    print("=" * 62)

    eps_last_actual = "N/A"
    if fin is not None and not fin.empty:
        eps_last_actual = str(fin.iloc[0][eps_col]) if eps_col in fin.columns else "N/A"

    print(f"""
  Stock:          {name} ({code}.SZ)
  Sector:         Electronics (PCB)
  Consensus EPS:  {eps_next} (forward year)
  Implied Growth: {growth:.1%}
  Sector PE:      {pe_now:.1f}
  PEG:            {peg:.2f}
  Analyst Count:  {int(stock_fc.iloc[-1].iloc[1])}
  Dispersion:     {(float(stock_fc.iloc[-1].iloc[4]) - float(stock_fc.iloc[-1].iloc[2])) / float(stock_fc.iloc[-1].iloc[3]):.1%}
  Industry Prem:  {(float(stock_fc.iloc[-1].iloc[3]) - float(stock_fc.iloc[-1].iloc[5])) / float(stock_fc.iloc[-1].iloc[5]):.1%}
  Last EPS:       {eps_last_actual}
  Revision Data:  {n_snaps} snapshots (need 4+)

  JUDGMENT:
    PEG = {peg:.2f} is {"LOW — market may be underpricing" if peg < 0.8 else "REASONABLE" if peg < 1.5 else "HIGH — growth already priced in"}
    the {growth:.1%} consensus growth rate.

    {"Growth ACCELERATING — EPS momentum is building." if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[-2] else "Growth stable." if len(growth_rates) >= 2 else ""}

    {"Analyst consensus is TIGHT ({:.0%} dispersion) — high conviction signal.".format(disp) if disp < 0.20 else "Moderate analyst dispersion ({:.0%}) — some uncertainty remains.".format(disp)}
""")

    # ── PEG Trap Analysis ──
    trap = _detect_peg_trap("sw_electronics", peg, growth, signal=0.5, dispersion=disp)
    if trap.trap_type != PegTrapType.NONE:
        print(f"""  [!] PEG TRAP: {trap.headline}

    SCENARIO A: {trap.scenario_a}

    SCENARIO B: {trap.scenario_b}

    EVIDENCE NEEDED (top 4 checks):
""")
        for j, check in enumerate(trap.evidence_checklist[:4]):
            print(f"      [{j+1}] {check}")
        print()

    print(f"""  BLIND SPOTS (model cannot see):
    - PCB is a cyclical manufacturing business — is AI demand structural enough to override the PCB cycle?
    - Customer concentration: key clients include Huawei, ZTE, and server OEMs
    - Raw material cost (copper, CCL) can compress margins unexpectedly
    - Geopolitical risk: PCB exports subject to trade policy changes

  QUESTIONS for human judgment:
    - Can AI server PCB demand sustain 40%+ growth for 3 consecutive years?
    - Is the current sector PE of {pe_now:.1f} (vs median {pe_med:.1f}) cheap because the market is right about a PCB cycle downturn, or cheap because the market is underestimating AI structural demand?
    - What do supply chain checks say about 沪电's order book for 2H 2026?
""")

    print("=" * 62)


if __name__ == "__main__":
    main()
