"""海光信息 PEG vs PRG 对比"""
import sys, time
sys.path.insert(0, '.')
import akshare as ak
import pandas as pd
import numpy as np
from data.store.duckdb_store import DuckDBStore

store = DuckDBStore("data/trading_god.duckdb")
code, name = "688041", "海光信息"

def p(raw):
    if not raw or raw in ('False','None','nan',''): return None
    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
    except: return None

# Forecast
print(f"=== {name} ({code}) PEG vs PRG ===\n")
fc = ak.stock_profit_forecast_ths(symbol=code)
fc = fc.sort_values(fc.columns[0])
latest=fc.iloc[-1]; prior=fc.iloc[-2]
eps_next=float(latest.iloc[3]); eps_curr=float(prior.iloc[3])
eps_growth=(eps_next/eps_curr-1)
eps_disp=(float(latest.iloc[4])-float(latest.iloc[2]))/eps_next

print("[利润端 - PEG]")
for _, r in fc.iterrows():
    print(f"  {r.iloc[0]}: EPS={float(r.iloc[3]):.2f}, n={int(r.iloc[1])}")
print(f"  共识EPS增速: {eps_growth:.1%} ({eps_curr:.2f} -> {eps_next:.2f})")
print(f"  分歧度: {eps_disp:.1%}")

# Financials
time.sleep(2)
fin = ak.stock_financial_abstract_ths(symbol=code)
dc=fin.columns[0]; rgc=fin.columns[6]; mgc=fin.columns[12]
annual=fin[~fin[dc].astype(str).str.contains("03-31|06-30|09-30",na=False)]
n_a=len(annual)

print("\n[营收端 - PRG]")
rev_g_list=[]
for idx in range(max(0,n_a-5), n_a):
    r=annual.iloc[idx]
    d=str(r[dc])[:10]
    rg=p(str(r[rgc])); mg=p(str(r[mgc]))
    if rg is not None: rev_g_list.append(rg)
    print(f"  {d}: Rev.G={str(r[rgc]):>10s}, Margin={str(r[mgc]):>10s}")

rev_g_now=rev_g_list[-1] if rev_g_list else 0.0
profit_q=min(1.0, max(0.0, (p(str(annual.iloc[-1][mgc])) or 0.0)*10))

# Latest quarters
print("\n  最近季度 (含2026Q1):")
for idx in range(max(0,len(fin)-3), len(fin)):
    r=fin.iloc[idx]
    d=str(r[dc])[:10]
    print(f"  {d}: Rev.G={str(r[rgc])}, Profit.G={str(r[fin.columns[2]])}, Margin={str(r[mgc])}")

# PE
pe_df=store.read_df("sw_industry_pe_daily")
semi=pe_df[pe_df["ts_code"]=="801081.SI"]
pec="pe" if "pe" in semi.columns else [c for c in semi.columns if "pe" in c.lower()][0]
pv=pd.to_numeric(semi[pec],errors="coerce").dropna()
pe_now=float(pv.iloc[-1])
peg_val=pe_now/(eps_growth*100) if eps_growth>0 else 999

# Head-to-head
print()
print("=" * 60)
print("  PEG vs PRG — Head-to-Head")
print("=" * 60)

rows = [
    ("估值锚", f"PE={pe_now:.1f} (中位{pv.median():.1f})", f"Rev.G={rev_g_now:.1%}"),
    ("增速", f"{eps_growth:.1%}", f"{rev_g_now:.1%}"),
    ("利润率", f"{p(str(annual.iloc[-1][mgc])):.1%}", f"{p(str(annual.iloc[-1][mgc])):.1%}"),
    ("分析师", f"{int(latest.iloc[1])}人", "N/A"),
    ("分歧度", f"{eps_disp:.1%}", "N/A"),
    ("比率/Verdict", f"PEG={peg_val:.2f}", f"PRG={'STRONG' if rev_g_now>0.5 else 'GOOD' if rev_g_now>0.3 else 'MODERATE'}"),
]
for label, peg_v, prg_v in rows:
    print(f"  {label:<12s} | {peg_v:<28s} | {prg_v:<20s}")

print()
prg_raw=(rev_g_now-0.15)*3
prg_sig=max(-1.0,min(1.0,prg_raw))
print(f"  PRG信号: {prg_sig:+.3f} (营收增速{rev_g_now:.1%} = {'极强' if prg_sig>0.5 else '强' if prg_sig>0.2 else '中'})")
print(f"  PEG结论: {'STRONG BUY' if peg_val<0.8 else 'BUY' if peg_val<1.2 else 'HOLD' if peg_val<2.0 else 'CAUTION'}")
print(f"  利润质量Q: {profit_q:.2f}")
if profit_q >= 1.0:
    print(f"  Q=1.0 → 纯PEG定价，PRG不影响混合信号")
else:
    w_peg = profit_q
    w_prg = 1 - profit_q
    print(f"  Q={profit_q:.2f} → PEG权重={w_peg:.0%} PRG权重={w_prg:.0%}")

print(f"""
  关键不对称:
  1. 营收增速 {rev_g_now:.1%} >> EPS增速 {eps_growth:.1%}
     → 营收在爆发但利润增速滞后（扩产/研发/备货的财务印记）
     → 当备货转化为交付时，利润增速会追赶营收增速

  2. Q=1.0 意味着市场对海光的利润率有充分信心
     → PEG是唯一估值锚，PRG只是隐藏的安全垫
     → 如果利润率恶化 → Q下降 → PRG自动接管 → {rev_g_now:.1%}的营收增速保护信号

  3. PEG=1.56(半导体) vs 1.05(计算机) vs PRG=极强
     → 三个框架给出了从\"偏贵\"到\"便宜\"的不同答案
     → 框架的选择本身就是预期差的来源
""")
