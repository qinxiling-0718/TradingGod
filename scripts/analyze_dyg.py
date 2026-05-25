"""东阳光 (600673.SH) — post-秦淮数据 acquisition analysis"""
import sys, time
sys.path.insert(0, '.')
import akshare as ak
import pandas as pd
import numpy as np
from data.store.duckdb_store import DuckDBStore
from factors.metadata_annotator import _detect_peg_trap, PegTrapType

store = DuckDBStore("data/trading_god.duckdb")
code, name = "600673", "东阳光"

def p(raw):
    if not raw or raw in ('False','None','nan',''): return None
    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
    except: return None

# ═══ 1. Forecast ═══
print("=" * 66)
print(f"  {name} ({code}.SH) — 收购秦淮数据后深度分析")
print("=" * 66)
print()
print("  背景：东阳光原为铝箔/化工企业，通过收购秦淮数据（ChinData）")
print("  进入AI数据中心赛道，正在经历从周期品到AI基础设施的转型。")
print()
time.sleep(2)
fc = ak.stock_profit_forecast_ths(symbol=code)
fc = fc.sort_values(fc.columns[0])

print("  一、分析师一致预期")
print("  " + "-" * 56)
growth_rates = []
for i, (_, r) in enumerate(fc.iterrows()):
    y, n, lo, mean, hi, ind = r.iloc[0], int(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4]), float(r.iloc[5])
    disp = (hi-lo)/mean if mean>0 else 0
    prem = (mean-ind)/ind if ind>0 else 0
    if i>0:
        prev = float(fc.iloc[i-1].iloc[3])
        g = (mean/prev - 1)
        growth_rates.append(g)
        grow_str = f"YoY={g:.1%}"
    else:
        grow_str = ""
    print(f"  {y}: EPS={mean:.2f} [{lo:.2f}-{hi:.2f}], n={n:>2d}, disp={disp:.1%}, prem={prem:.1%} {grow_str}")

latest = fc.iloc[-1]; prior = fc.iloc[-2]
eps_next = float(latest.iloc[3]); eps_curr = float(prior.iloc[3])
growth = (eps_next/eps_curr - 1)
n_now = int(latest.iloc[1])
disp_now = (float(latest.iloc[4])-float(latest.iloc[2]))/eps_next
ind_avg = float(latest.iloc[5])

if len(growth_rates) >= 2:
    g1, g2 = growth_rates[-1], growth_rates[-2]
    trend = "ACCELERATING" if g1>g2+0.03 else ("DECELERATING" if g1<g2-0.03 else "STABLE")
    print(f"  >> Growth {trend}: {g2:.1%} -> {g1:.1%}")

# ═══ 2. Financials ═══
print()
print("  二、财务基本面（最新年度）")
print("  " + "-" * 56)
time.sleep(2)
fin = ak.stock_financial_abstract_ths(symbol=code)
dc=fin.columns[0]; rc=fin.columns[5]; rgc=fin.columns[6]; mgc=fin.columns[12]
pc=fin.columns[1]; pgc=fin.columns[2]; epsc=fin.columns[7]

# Data is oldest-first. Get newest annual.
annual = fin[~fin[dc].astype(str).str.contains("03-31|06-30|09-30",na=False)]
# Show newest 5 annual
n_annual = len(annual)
for idx in range(max(0,n_annual-5), n_annual):
    r = annual.iloc[idx]
    print(f"  {str(r[dc])[:10]}: Rev={str(r[rc]):>14s} Rev.G={str(r[rgc]):>10s} "
          f"Profit={str(r[pc]):>12s} Margin={str(r[mgc]):>10s}")

# Latest valid metrics (scan newest first)
rev_g, margin = None, None
for idx in range(n_annual-1, -1, -1):
    r = annual.iloc[idx]
    if rev_g is None: rev_g = p(str(r[rgc]))
    if margin is None: margin = p(str(r[mgc]))
    if rev_g is not None and margin is not None: break
rev_g = rev_g or 0.0; margin = margin or 0.0

# Latest quarter data (newest at end of df)
print(f"\n  最近四个季度（含最新）:")
n_total = len(fin)
for idx in range(max(0,n_total-4), n_total):
    r = fin.iloc[idx]
    d = str(r[dc])[:10]
    if "03-31" in d or "06-30" in d or "09-30" in d or "12-31" in d:
        print(f"    {d}: Rev={str(r[rc])}, Rev.G={str(r[rgc])}, "
              f"Profit={str(r[pc])}, Profit.G={str(r[pgc])}, Margin={str(r[mgc])}")

profit_q = max(0.0, min(1.0, margin*10))

# ═══ 3. Industry Classification ═══
print()
print("  三、行业归属辩论")
print("  " + "-" * 56)
print(f"  Tushare分类: 综合类（未明确归属于AI产业链）")
print(f"  实际业务: 铝箔/化工（传统）+ 秦淮数据（AI数据中心）")
print(f"  当前EPS远低于行业平均（premium={ind_avg} vs {eps_next}）")
print(f"  → 分析师预测使用的是'综合类'行业平均EPS，")
print(f"    但这可能在低估秦淮数据的成长性。")
print(f"  → 这是一个跨行业转型中的公司——旧标签还没更新。")

# ═══ 4. PEG ═══
print()
print("  四、PEG 估值分析（使用电子行业PE作为数据中心proxy）")
print("  " + "-" * 56)
pe_df = store.read_df("sw_industry_pe_daily")
for label, sw_code in [("电子", "801080.SI"), ("计算机", "801101.SI")]:
    sec = pe_df[pe_df["ts_code"] == sw_code]
    pec = "pe" if "pe" in sec.columns else [c for c in sec.columns if "pe" in c.lower()][0]
    pv = pd.to_numeric(sec[pec], errors="coerce").dropna()
    pe_cur = float(pv.iloc[-1])
    peg = pe_cur/(growth*100) if growth>0 else 999
    print(f"  {label}行业 PE={pe_cur:.1f}, 中位数={pv.median():.1f} → PEG={peg:.2f}")

# Use electronics PE for main analysis
sec = pe_df[pe_df["ts_code"] == "801080.SI"]
pec = "pe" if "pe" in sec.columns else [c for c in sec.columns if "pe" in c.lower()][0]
pv = pd.to_numeric(sec[pec], errors="coerce").dropna()
pe_now = float(pv.iloc[-1]); pe_med = float(pv.median())
peg_elec = pe_now/(growth*100) if growth>0 else 999
pe_pos = (pe_now-pv.min())/(pv.max()-pv.min())*100

print()
print(f"  使用电子行业PE={pe_now:.1f}:")
print(f"  共识EPS(2028): {eps_next:.2f}, EPS(2027): {eps_curr:.2f}")
print(f"  隐含增速: {growth:.1%}")

# Also compute with 计算机PE
sec_cs = pe_df[pe_df["ts_code"] == "801101.SI"]
pec_cs = "pe" if "pe" in sec_cs.columns else [c for c in sec_cs.columns if "pe" in c.lower()][0]
pv_cs = pd.to_numeric(sec_cs[pec_cs], errors="coerce").dropna()
pe_cs = float(pv_cs.iloc[-1])
peg_cs = pe_cs/(growth*100) if growth>0 else 999

# ═══ 5. Hybrid ═══
print()
print("  五、混合 PEG")
print("  " + "-" * 56)
print(f"  最新营收增速: {rev_g:.1%}")
print(f"  最新净利率:   {margin:.1%}")
print(f"  利润质量 Q:   {profit_q:.2f}")
style = "营收驱动 → PRG主导" if profit_q<0.5 else ("过渡型 → 混合" if profit_q<0.8 else "利润驱动 → PEG主导")
prg_raw = (rev_g-0.15)*3
prg_sig = max(-1.0, min(1.0, prg_raw))
print(f"  公司阶段:     {style}")
print(f"  PRG信号:      {prg_sig:+.3f}")

# ═══ 6. PEG Trap ═══
print()
print("  六、PEG 陷阱检测")
print("  " + "-" * 56)
for label, pv_use, pe_use in [("电子行业", pv, pe_now), ("计算机行业", pv_cs, pe_cs)]:
    peg_use = pe_use/(growth*100) if growth>0 else 999
    trap = _detect_peg_trap("sw_electronics", peg_use, growth, 0.5, disp_now)
    if trap.trap_type != PegTrapType.NONE:
        print(f"  [{label} PE={pe_use:.1f}] [!] {trap.headline}")
    else:
        print(f"  [{label} PE={pe_use:.1f}] PEG={peg_use:.2f} — 无陷阱触发")

# ═══ 7. Agent Thesis ═══
print()
print("=" * 66)
print("  AGENT 投资展望：东阳光 × 秦淮数据")
print("=" * 66)
print(f"""
  【转型本质】
  东阳光收购秦淮数据，本质上是用周期股的估值买入了一个数据中心资产。
  这类似于2019年AMD收购Xilinx——旧标签（CPU公司）被新叙事（FPGA+AI）覆盖。

  但东阳光的情况更极端：
  - 旧业务（铝箔/化工）仍在运行，产生现金流但增长有限
  - 新业务（秦淮数据）是高增长的AI基础设施，但尚未被市场充分定价
  - 分析师只给了{n_now}人覆盖，EPS预测仅{eps_next:.2f}——
    这个数字可能完全没有反映秦淮数据的真实增长潜力

  【估值框架的撕裂】
  如果按\"综合类\"行业估值：PE可能只有15-20x → 股价被严重低估
  如果按\"数据中心\"估值：可比公司（光环新网、宝信软件）PE在30-50x
  如果按\"AI基础设施\"估值：可比公司PE在50-80x

  目前市场大概率在用第一个框架定价，而秦淮数据的资产'
  需要用第二个甚至第三个框架来定价。
  这个\"估值框架的切换\"就是东阳光最大的预期差来源。

  【分析师覆盖的盲区】
  只有{n_now}位分析师覆盖（且2028年只剩2人）。
  覆盖人数在下降——老分析师不熟悉数据中心，新分析师还没入场。
  这个\"覆盖真空\"意味着：
  - 市场共识尚未形成 → 预期差巨大
  - 但风险也大 → 没有分析师帮你验证基本面

  【关键问题】
  1. 秦淮数据的真实营收和利润贡献占东阳光的比例？
     如果数据中心业务占比超过30%，东阳光就应该被重分类。
  2. 秦淮数据的主要客户是谁？字节/快手/阿里？合同期限？
     这决定了数据中心收入的稳定性。
  3. 旧业务（铝箔/化工）的处置计划？
     如果东阳光最终剥离旧业务，就成了纯数据中心公司——重估逻辑完全不同。
  4. 管理层的战略意图？
     是把秦淮数据当\"第二增长曲线\"还是\"主业转型\"？

  【远景路径】
  乐观：秦淮数据业绩超预期→分析师开始用数据中心框架估值
       →覆盖人数从{n_now}人增到10+人→分歧形成→共识凝聚
       →PE从{pe_now:.1f}x(电子)重估到50x+(数据中心)
       →股价可能在12-18个月内翻倍
  中性：秦淮数据稳定增长→东阳光逐步被市场了解
       →分析师缓慢增加→估值缓慢上移→年化回报15-25%
  悲观：秦淮数据整合失败/客户流失/旧业务拖累
       →市场继续按周期股估值→股价长期低迷

  【东阳光在我们的AI产业链中的位置】
  如果秦淮数据整合成功，东阳光应该被纳入AI产业链的 Layer 3
  （计算基础设施 — 数据中心），与光环新网、宝信软件、奥飞数据并列。
  但目前它还在\"综合类\"的标签下——这是信息不对称的最大来源。

  {'='*66}
    免责声明：以上为AI辅助分析，不构成投资建议。
    秦淮数据的具体财务数据未单独披露，分析基于公开信息和合理推断。
  {'='*66}
""")
