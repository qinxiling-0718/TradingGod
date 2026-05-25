"""Batch analysis for 3 stocks."""
import sys, time
sys.path.insert(0, '.')
import akshare as ak
import pandas as pd
import numpy as np
from data.store.duckdb_store import DuckDBStore
from factors.metadata_annotator import _detect_peg_trap, PegTrapType

store = DuckDBStore("data/trading_god.duckdb")
pe_df = store.read_df("sw_industry_pe_daily")

STOCKS = [
    ("603629", "利通电子", "元器件"),
    ("600584", "长电科技", "半导体"),
    ("300857", "协创数据", "IT设备"),
]

def p(raw):
    if not raw or raw in ('False','None','nan',''): return None
    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
    except: return None

def get_sector_pe(sw_code):
    sec = pe_df[pe_df["ts_code"] == sw_code]
    pec = "pe" if "pe" in sec.columns else [c for c in sec.columns if "pe" in c.lower()][0]
    pv = pd.to_numeric(sec[pec], errors="coerce").dropna()
    return float(pv.iloc[-1]), float(pv.median()), pv

for code, name, industry in STOCKS:
    print(f"\n{'='*64}")
    print(f"  {name} ({code}.{'SH' if code.startswith('6') else 'SZ'}) — {industry}")
    print(f"{'='*64}")

    # ── Forecast ──
    time.sleep(2)
    try:
        fc = ak.stock_profit_forecast_ths(symbol=code)
        if fc is None or fc.empty:
            print(f"  No forecast data\n")
            continue
        fc = fc.sort_values(fc.columns[0])
    except Exception as e:
        print(f"  Forecast ERROR: {e}\n")
        continue

    print(f"\n  [分析师一致预期]")
    growth_rates = []
    for i, (_, r) in enumerate(fc.iterrows()):
        y, n, lo, mean, hi, ind = r.iloc[0], int(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4]), float(r.iloc[5])
        disp = (hi-lo)/mean if mean>0 else 0
        if i>0:
            prev = float(fc.iloc[i-1].iloc[3])
            g = (mean/prev - 1)
            growth_rates.append(g)
            grow_str = f"YoY={g:.1%}"
        else:
            grow_str = ""
        print(f"  {y}: EPS={mean:.2f} [{lo:.2f}-{hi:.2f}], n={n:>2d}, disp={disp:.1%} {grow_str}")

    latest = fc.iloc[-1]; prior = fc.iloc[-2]
    eps_next = float(latest.iloc[3]); eps_curr = float(prior.iloc[3])
    growth = (eps_next/eps_curr - 1)
    n_now = int(latest.iloc[1])
    disp_now = (float(latest.iloc[4])-float(latest.iloc[2]))/eps_next
    if len(growth_rates) >= 2:
        g1, g2 = growth_rates[-1], growth_rates[-2]
        trend = "ACCELERATING" if g1>g2+0.03 else ("DECELERATING" if g1<g2-0.03 else "STABLE")
        print(f"  >> Growth {trend}: {g2:.1%} -> {g1:.1%}")

    # ── Financials ──
    time.sleep(2)
    try:
        fin = ak.stock_financial_abstract_ths(symbol=code)
        dc=fin.columns[0]; rc=fin.columns[5]; rgc=fin.columns[6]; mgc=fin.columns[12]
        annual = fin[~fin[dc].astype(str).str.contains("03-31|06-30|09-30",na=False)]
        n_a = len(annual)
        # Latest 3 annual + latest quarter
        rev_g, margin = None, None
        for idx in range(n_a-1, -1, -1):
            r = annual.iloc[idx]
            if rev_g is None: rev_g = p(str(r[rgc]))
            if margin is None: margin = p(str(r[mgc]))
            if rev_g is not None and margin is not None: break
        rev_g = rev_g or 0.0; margin = margin or 0.0
        profit_q = max(0.0, min(1.0, margin*10))
        print(f"\n  [财务] Rev.G={rev_g:.1%}, Margin={margin:.1%}, Q={profit_q:.2f}")
        for idx in range(max(0,n_a-3), n_a):
            r = annual.iloc[idx]
            print(f"    {str(r[dc])[:10]}: Rev.G={str(r[rgc]):>10s}, Margin={str(r[mgc]):>10s}")
    except:
        rev_g, margin, profit_q = 0.0, 0.0, 0.5
        print(f"\n  [财务] ERROR fetching financials")

    # ── PEG ──
    # Map industry to SW code
    sw_map = {"半导体":"801081.SI","元器件":"801080.SI","IT设备":"801101.SI"}
    sw_code = sw_map.get(industry, "801080.SI")
    pe_now, pe_med, pv = get_sector_pe(sw_code)
    peg_val = pe_now/(growth*100) if growth>0 else 999
    print(f"\n  [估值] PE={pe_now:.1f}(中位{pe_med:.1f}), PEG={peg_val:.2f}")

    # ── Hybrid ──
    prg_raw = (rev_g-0.15)*3
    prg_sig = max(-1.0, min(1.0, prg_raw))
    style = "PRG主导" if profit_q<0.5 else ("混合" if profit_q<0.8 else "PEG主导")
    print(f"  [混合] {style}, PRG={prg_sig:+.2f}")

    # ── Trap ──
    trap = _detect_peg_trap(sw_code, peg_val, growth, 0.5, disp_now)
    trap_msg = trap.headline if trap.trap_type != PegTrapType.NONE else "无陷阱"
    print(f"  [PEG陷阱] {trap_msg}")

    # ── Agent ──
    verdict = ""
    if peg_val < 0.8 and growth > 0.2:
        verdict = "强买入：低PEG+高增长，预期差显著"
    elif peg_val < 1.2 and growth > 0.15:
        verdict = "偏买入：PEG合理偏低，有安全边际"
    elif peg_val < 2.0:
        verdict = "持有：增长已被合理定价"
    else:
        verdict = "谨慎：PEG偏高，预期过于乐观或增速不足"

    analyst_verdict = ""
    if n_now >= 15 and disp_now < 0.3:
        analyst_verdict = "高共识：多位分析师+低分歧→预期可靠"
    elif n_now >= 10:
        analyst_verdict = f"中共识：{n_now}分析师+分歧{disp_now:.0%}→预期有参考价值"
    elif n_now >= 3:
        analyst_verdict = f"低覆盖：仅{n_now}分析师→信息不对称大，预期差可能被低估"
    else:
        analyst_verdict = f"极低覆盖：{n_now}分析师→几乎无市场共识，最大预期差"

    print(f"\n  [Agent判断]")
    print(f"  {verdict}")
    print(f"  {analyst_verdict}")
    print(f"  核心观察: EPS增速={growth:.1%}, 营收增速={rev_g:.1%}, 利润率={margin:.1%}")
    if profit_q < 0.5:
        print(f"  注意: 利润率极薄(Q={profit_q:.2f})→营收增速权重更高→关注营收势头")
    if disp_now > 0.5:
        print(f"  注意: 分析师分歧>50%→共识可靠性低→分歧因子值得关注")
    if n_now < 8:
        print(f"  注意: 分析师覆盖不足→存在信息盲区→基本面验证需依赖自有调研")

print(f"\n{'='*64}")
print(f"  Batch analysis complete.")
print(f"{'='*64}")
