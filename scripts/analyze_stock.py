"""Single or multi-stock deep analysis — PEG + PRG + dispersion + trap.

Usage:
  uv run python scripts/analyze_stock.py 002463              # single
  uv run python scripts/analyze_stock.py 002463 300502       # multi
  uv run python scripts/analyze_stock.py 002463 300502 688041 # compare N stocks
"""

import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import akshare as ak
import pandas as pd
import numpy as np
from data.store.duckdb_store import DuckDBStore
from factors.metadata_annotator import _detect_peg_trap, PegTrapType
from factors.dispersion_factor import compute_dispersion_signals, DispersionQuadrant

store = DuckDBStore("data/trading_god.duckdb")

# ── Helpers ────────────────────────────────────────────────────────

def parse_pct(raw):
    if not raw or raw in ('False','None','nan',''): return None
    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
    except: return None

SW_PE_MAP = {
    "半导体": "801081.SI", "电子": "801080.SI", "元器件": "801080.SI",
    "计算机": "801101.SI", "IT设备": "801101.SI", "软件": "801101.SI",
    "通信设备": "801102.SI", "通信": "801102.SI",
    "传媒": "801760.SI", "医药": "801150.SI", "军工": "801740.SI",
    "有色": "801050.SI", "汽车": "801110.SI", "化工": "801030.SI",
    "机械": "801070.SI", "电气": "801730.SI", "综合": "801080.SI",
}

def get_pe(sw_code):
    pe_df = store.read_df("sw_industry_pe_daily")
    sec = pe_df[pe_df["ts_code"] == sw_code]
    if sec.empty: return 0, 0
    pec = "pe" if "pe" in sec.columns else [c for c in sec.columns if "pe" in c.lower()][0]
    pv = pd.to_numeric(sec[pec], errors="coerce").dropna()
    return float(pv.iloc[-1]), float(pv.median())

def get_sector_pe(industry, code):
    if code.startswith("688"): return get_pe("801081.SI")
    sw = SW_PE_MAP.get(industry)
    if not sw: return get_pe("801080.SI")
    return get_pe(sw)

# ── Stock Analysis ─────────────────────────────────────────────────

def analyze(code):
    """Fetch and analyze a single stock. Returns dict of metrics."""
    time.sleep(1.5)

    # Forecast
    try:
        fc = ak.stock_profit_forecast_ths(symbol=code)
        if fc is None or fc.empty: return None
        fc = fc.sort_values(fc.columns[0])
    except Exception:
        return None

    latest = fc.iloc[-1]; prior = fc.iloc[-2]
    eps_next = float(latest.iloc[3]); eps_curr = float(prior.iloc[3])
    eps_growth = (eps_next / eps_curr - 1) if eps_curr > 0 else 0
    n_analysts = int(latest.iloc[1])
    eps_disp = (float(latest.iloc[4]) - float(latest.iloc[2])) / eps_next if eps_next > 0 else 0

    # Growth trend
    growth_rates = []
    for i in range(1, len(fc)):
        prev = float(fc.iloc[i-1].iloc[3])
        curr = float(fc.iloc[i].iloc[3])
        if prev > 0: growth_rates.append(curr / prev - 1)
    if len(growth_rates) >= 2:
        g1, g2 = growth_rates[-1], growth_rates[-2]
        trend = "ACCEL" if g1 > g2 + 0.03 else ("DECEL" if g1 < g2 - 0.03 else "STABLE")
    else:
        trend = "N/A"

    # Financials
    time.sleep(1.5)
    try:
        fin = ak.stock_financial_abstract_ths(symbol=code)
        dc=fin.columns[0]; rgc=fin.columns[6]; mgc=fin.columns[12]; pgc=fin.columns[2]
        annual=fin[~fin[dc].astype(str).str.contains("03-31|06-30|09-30",na=False)]
        na=len(annual)
        rev_g, margin = None, None
        for idx in range(na-1, -1, -1):
            r=annual.iloc[idx]
            if rev_g is None: rev_g=parse_pct(str(r[rgc]))
            if margin is None: margin=parse_pct(str(r[mgc]))
            if rev_g is not None and margin is not None: break
        rev_g=rev_g or 0.0; margin=margin or 0.0
        profit_q=min(1.0, max(0.0, margin*10))

        # Latest quarters
        quarters=[]
        for idx in range(max(0,len(fin)-3), len(fin)):
            r=fin.iloc[idx]
            d=str(r[dc])[:10]
            quarters.append({
                "date":d,
                "rev_g":parse_pct(str(r[rgc])) or 0,
                "profit_g":parse_pct(str(r[pgc])) or 0,
                "margin":parse_pct(str(r[mgc])) or 0,
            })
    except Exception:
        rev_g=margin=profit_q=0.0; quarters=[]

    # Industry info
    industry = "半导体" if code.startswith("688") else "电子"
    try:
        ts = __import__('tushare').pro_api()
        df = store.read_df("ai_stock_universe") if store.table_exists("ai_stock_universe") else pd.DataFrame()
    except:
        df = pd.DataFrame()
    if not df.empty:
        row = df[df["ts_code"].str.contains(code)]
        if not row.empty: industry = row["industry"].iloc[0]

    # PE / PEG
    pe_now, pe_med = get_sector_pe(industry, code)
    peg_val = pe_now / (eps_growth * 100) if eps_growth > 0 else 999

    # PRG
    prg_raw = (rev_g - 0.15) * 3
    prg_sig = max(-1.0, min(1.0, prg_raw))
    style = "PRG主导" if profit_q < 0.5 else ("混合" if profit_q < 0.8 else "PEG主导")

    # PEG trap
    trap_type = _detect_peg_trap("sw_electronics", peg_val, eps_growth, 0.5, eps_disp).trap_type.value

    # Dispersion signal
    disp_signals = compute_dispersion_signals(lookback_snapshots=4)
    disp_quad = "N/A"
    disp_dcons = 0.0
    for s in disp_signals:
        if code in s.ts_code:
            disp_quad = s.quadrant.value
            disp_dcons = s.delta_consensus
            break

    return {
        "code": code, "eps_next": eps_next, "eps_curr": eps_curr,
        "eps_growth": eps_growth, "n_analysts": n_analysts,
        "eps_disp": eps_disp, "trend": trend,
        "rev_g": rev_g, "margin": margin, "profit_q": profit_q,
        "pe_now": pe_now, "pe_med": pe_med, "peg": peg_val,
        "prg_sig": prg_sig, "style": style,
        "trap": trap_type, "disp_quad": disp_quad, "disp_dcons": disp_dcons,
        "quarters": quarters, "industry": industry,
    }

# ── Display ────────────────────────────────────────────────────────

HEADER_FMT = "{:<12s} {:>6s} {:>7s} {:>7s} {:>7s} {:>5s} {:>8s} {:>7s} {:>8s} {:>8s} {:>12s}"
ROW_FMT = "{:<12s} {:>5.1f} {:>6.1%} {:>6.1%} {:>6.1%} {:>4.2f} {:>6.1f} {:>+6.2f} {:>+7.3f} {:>8s} {:>12s}"

def print_header():
    print("\n" + HEADER_FMT.format(
        "Stock", "PEG", "G%", "RevG%", "Margin", "Q", "PE", "PRG", "PEGsig", "Style", "Disp/Q"
    ))
    print("-" * 88)

def print_row(d):
    print(ROW_FMT.format(
        d["code"], d["peg"], d["eps_growth"], d["rev_g"], d["margin"],
        d["profit_q"], d["pe_now"], d["prg_sig"], 0.0, d["style"],
        f"{d['disp_quad']}/{d['eps_disp']:.0%}"
    ))

def print_detail(d, name_lookup):
    label = name_lookup.get(d["code"], d["code"])
    print(f"\n{'='*62}")
    print(f"  {label} ({d['code']}) — {d['industry']}")
    print(f"{'='*62}")
    print(f"  PEG={d['peg']:.2f} | Growth={d['eps_growth']:.1%} | "
          f"Rev.G={d['rev_g']:.1%} | Margin={d['margin']:.1%} | Q={d['profit_q']:.2f}")
    print(f"  PE={d['pe_now']:.1f}(中位{d['pe_med']:.1f}) | PRG={d['prg_sig']:+.2f} | "
          f"Style={d['style']} | Analysts={d['n_analysts']} | Disp={d['eps_disp']:.0%}")
    print(f"  Disp.Quadrant={d['disp_quad']} | dCons={d['disp_dcons']:.1%} | "
          f"Trap={d['trap']} | Growth={d['trend']}")
    if d["quarters"]:
        print(f"  Recent quarters:")
        for q in d["quarters"]:
            print(f"    {q['date']}: Rev.G={q['rev_g']:.1%} "
                  f"Profit.G={q['profit_g']:.1%} Margin={q['margin']:.1%}")

# ── Main ───────────────────────────────────────────────────────────

def main():
    codes = sys.argv[1:] if len(sys.argv) > 1 else []
    if not codes:
        print("Usage: uv run python scripts/analyze_stock.py <code1> [code2] ...")
        print("Example: uv run python scripts/analyze_stock.py 002463 300502")
        return

    # Name lookup
    name_lookup = {}
    try:
        fc = store.read_df("analyst_forecast")
        for _, r in fc[["ts_code","name"]].drop_duplicates().iterrows():
            name_lookup[r["ts_code"].split(".")[0]] = r["name"]
    except: pass

    results = []
    for code in codes:
        code = code.strip().replace(".SH","").replace(".SZ","").replace(".sh","").replace(".sz","")
        print(f"\n... Analyzing {code} ...", end=" ", flush=True)
        r = analyze(code)
        if r:
            results.append(r)
            name = name_lookup.get(code, code)
            print(f"OK ({name})")
        else:
            print("NO DATA")

    if not results:
        print("\nNo valid data for any stock.")
        return

    # Summary table
    print("\n" + "=" * 88)
    print("  COMPARISON TABLE")
    print("=" * 88)
    print_header()
    for d in sorted(results, key=lambda x: x["peg"]):
        print_row(d)

    # Detail per stock
    for d in results:
        print_detail(d, name_lookup)

if __name__ == "__main__":
    main()
