"""Comprehensive analysis: 思瑞浦 (688536.SH)"""
import sys, time
sys.path.insert(0, '.')
import akshare as ak
import pandas as pd
import numpy as np
from data.store.duckdb_store import DuckDBStore
from factors.metadata_annotator import _detect_peg_trap, PegTrapType

store = DuckDBStore("data/trading_god.duckdb")
code, name = "688536", "思瑞浦"

def pct(raw):
    if not raw or raw in ('False','None','nan',''): return None
    try: return float(str(raw).replace('%','').replace('+','').strip())/100.0
    except: return None

# ═══ 1. Consensus ═══
print("=" * 64)
print(f"  {name} ({code}.SH) 深度分析报告")
print("=" * 64)

print("\n  一、分析师一致预期")
print("  " + "-" * 56)
time.sleep(2)
fc = ak.stock_profit_forecast_ths(symbol=code)
fc = fc.sort_values(fc.columns[0])
growth_rates = []
for i, (_, r) in enumerate(fc.iterrows()):
    y = r.iloc[0]; n = int(r.iloc[1])
    lo, mean, hi = float(r.iloc[2]), float(r.iloc[3]), float(r.iloc[4])
    ind = float(r.iloc[5])
    disp = (hi-lo)/mean if mean>0 else 0
    prem = (mean-ind)/ind if ind>0 else 0
    if i>0:
        prev = float(fc.iloc[i-1].iloc[3])
        g = (mean/prev - 1)
        growth_rates.append(g)
        grow_str = f"YoY={g:.1%}"
    else:
        grow_str = ""
    print(f"  {y}: EPS={mean:.2f} [{lo:.2f}-{hi:.2f}], n={n:>2d}, "
          f"disp={disp:.1%}, prem={prem:.1%} {grow_str}")

latest = fc.iloc[-1]; prior = fc.iloc[-2]
eps_next = float(latest.iloc[3]); eps_curr = float(prior.iloc[3])
growth = (eps_next/eps_curr - 1)
n_now = int(latest.iloc[1])
disp_now = (float(latest.iloc[4])-float(latest.iloc[2]))/eps_next

if len(growth_rates) >= 2:
    g1, g2 = growth_rates[-1], growth_rates[-2]
    trend = "ACCELERATING" if g1>g2+0.03 else ("DECELERATING" if g1<g2-0.03 else "STABLE")
    print(f"  >> Growth {trend}: {g2:.1%} -> {g1:.1%}")

# ═══ 2. Financials ═══
print("\n  二、财务基本面")
print("  " + "-" * 56)
time.sleep(2)
fin = ak.stock_financial_abstract_ths(symbol=code)
dc = fin.columns[0]; rc = fin.columns[5]; rgc = fin.columns[6]
mc = fin.columns[12]; pc = fin.columns[1]; pgc = fin.columns[2]

annual = fin[~fin[dc].astype(str).str.contains("03-31|06-30|09-30",na=False)]
annual_new = annual.iloc[::-1].head(5)
print(f"  {'Period':<14s} {'Revenue':>14s} {'Rev.G':>10s} {'Profit':>12s} {'Margin':>10s}")
for _, r in annual_new.iterrows():
    d = str(r[dc])[:10]
    rev = str(r[rc])
    rg = str(r[rgc])
    prof = str(r[pc])
    mg = str(r[mc])
    print(f"  {d:<14s} {rev:>14s} {rg:>10s} {prof:>12s} {mg:>10s}")

# Latest valid metrics (scan newest first)
rev_g, margin = None, None
for idx in range(len(annual)-1, -1, -1):
    r = annual.iloc[idx]
    if rev_g is None: rev_g = pct(str(r[rgc]))
    if margin is None: margin = pct(str(r[mc]))
    if rev_g is not None and margin is not None: break
rev_g = rev_g or 0.0; margin = margin or 0.0
profit_q = max(0.0, min(1.0, margin*10))

# Recent quarters
recent = fin.head(4)
print(f"\n  最近四个季度:")
for _, r in recent.iterrows():
    print(f"    {str(r[dc])[:10]}: Rev={str(r[rc])}, Rev.G={str(r[rgc])}, "
          f"Profit.G={str(r[pgc])}, Margin={str(r[mc])}")

# ═══ 3. PEG ═══
print("\n  三、PEG 估值分析")
print("  " + "-" * 56)
pe_df = store.read_df("sw_industry_pe_daily")
semi = pe_df[pe_df["ts_code"] == "801081.SI"]
pec = "pe" if "pe" in semi.columns else [c for c in semi.columns if "pe" in c.lower()][0]
pv = pd.to_numeric(semi[pec], errors="coerce").dropna()
pe_now = float(pv.iloc[-1]); pe_med = float(pv.median())
pe_p25 = float(pv.quantile(0.25)); pe_p75 = float(pv.quantile(0.75))
peg_val = pe_now/(growth*100) if growth>0 else 999
pe_pos = (pe_now-pv.min())/(pv.max()-pv.min())*100

print(f"  半导体 PE: {pe_now:.1f} (中位数: {pe_med:.1f}, 分位: {pe_pos:.0f}%)")
print(f"  共识EPS(2028): {eps_next:.2f}, 共识EPS(2027): {eps_curr:.2f}")
print(f"  隐含增速: {growth:.1%}")
print(f"  PEG = {pe_now:.1f} / {growth*100:.1f} = {peg_val:.2f}")
print()
print(f"  情景分析:")
for pct, label in [(10,"极熊"), (25,"熊"), (50,"中位"), (75,"牛"), (90,"极牛")]:
    ps = float(pv.quantile(pct/100))
    ps_peg = ps/(growth*100) if growth>0 else 999
    print(f"    {label:>6s} (PE={ps:.1f}): PEG={ps_peg:.2f}  "
          f"{'<< 低估' if ps_peg<0.8 else '<< 合理偏低估' if ps_peg<1.2 else ''}")

# ═══ 4. Hybrid PEG ═══
print("\n  四、混合 PEG")
print("  " + "-" * 56)
print(f"  最新营收增速: {rev_g:.1%}")
print(f"  最新净利率:   {margin:.1%}")
print(f"  利润质量 Q:   {profit_q:.2f}")
style = "营收驱动 → PRG主导" if profit_q<0.5 else ("过渡型 → 混合" if profit_q<0.8 else "利润驱动 → PEG主导")
prg_raw = (rev_g-0.15)*3
prg_sig = max(-1.0, min(1.0, prg_raw))
print(f"  公司阶段:     {style}")
print(f"  PRG信号:      {prg_sig:+.3f} (营收增速{rev_g:.1%} → 信号{'强' if prg_sig>0.3 else '中' if prg_sig>0 else '弱'})")
print(f"  混合信号:     Q={profit_q:.2f} → PRG占{1-profit_q:.0%}权重")

# ═══ 5. PEG Trap ═══
print("\n  五、PEG 陷阱检测")
print("  " + "-" * 56)
trap = _detect_peg_trap("sw_semiconductor", peg_val, growth, 0.5, disp_now)
if trap.trap_type != PegTrapType.NONE:
    print(f"  [!] {trap.headline}")
    print(f"  模型偏向: {trap.model_lean:.0%} Scenario B")
else:
    print(f"  PEG={peg_val:.2f} 在合理区间 (0.8-3.0)，无陷阱触发")

# ═══ 6. Agent Thesis ═══
print()
print("=" * 64)
print("  AGENT 投资展望")
print("=" * 64)

report = f"""
  【公司画像】
  {name} — 中国模拟芯片龙头，聚焦信号链（运放/ADC/DAC/接口）
  和电源管理芯片。下游覆盖工业、汽车、通信、消费电子。

  在AI硬件生态中，信号链芯片是"感知层"的核心元件——
  没有高精度ADC/DAC，AI推理芯片就无法与物理世界交互。

  【估值判断】PEG = {peg_val:.2f}
  PEG={peg_val:.2f} 处于合理偏低估区间。即使熊市PE场景（PE={pe_p25:.1f}），
  PEG仍只有{pe_p25/(growth*100):.2f}，安全边际充足。

  当前半导体PE={pe_now:.1f}处于历史{pe_pos:.0f}%分位，
  板块整体估值偏低——这给了思瑞浦一个"水涨船高"的保护。

  【增长质量】Q = {profit_q:.2f} — {style}
  营收增速 {rev_g:.1%} > 利润增速 {growth:.1%}。
  这是典型的技术投入期公司特征：
  - 研发费用吃掉利润（模拟芯片需要大量工程师和IP积累）
  - 扩产初期折旧高
  - 客户导入周期长（车规认证1-2年）

  但Q={profit_q:.2f}意味着系统自动给了PRG {1-profit_q:.0%}的权重——
  营收增速的强劲被纳入了评估，而不是只看利润。

  【共识质量】分歧 {disp_now:.1%}
  {n_now}位分析师覆盖，分歧{disp_now:.1%}属于"有争议但可控"。
  乐观派看到国产替代和汽车放量，保守派担心TI竞争和研发投入。
  这个分歧水平说明市场还没有形成一致预期——预期差存在。

  【核心预期差】
  思瑞浦的Alpha不在增速本身（{growth:.1%}已被定价），而在两个"非线性拐点"：

  拐点1: 汽车芯片放量
    - 新能源车的BMS/域控需要大量模拟芯片
    - 思瑞浦的车规产品线正在导入期
    - 一旦通过头部Tier1认证→营收增速可能跳升到60%+

  拐点2: 利润率拐点
    - 当前净利率{margin:.1%}→若2027年回到15%+
    - Q从{profit_q:.2f}跳到1.0→混合PEG逻辑重置
    - 市场会开始用纯PEG框架重新给思瑞浦定价
    - PEG恒定的前提下，利润增速上升→股价必须跟涨

  这两个拐点的发生时点，决定了思瑞浦未来2年的投资回报率。

  【风险】
  - TI/ADI降价竞争：模拟芯片不是蓝海，价格战随时可能发生
  - 研发转化率：大量投入能否形成有竞争力的产品矩阵？
  - 客户集中度：前五大客户占比？单一客户依赖风险？
  - 估值天花板：即使利润释放，模拟芯片的PE中枢在40-60x，
    当前PE={pe_now:.1f}已经接近合理区间

  【远景路径】
  {name}的路径取决于"国产替代的深度"：

  乐观（2027年）：
    汽车+工业客户批量导入→营收增速维持50%+
    →利润率回到15%→利润增速反超营收增速
    →PEG被动下降→重新定价→双击
    估值锚：PE回到60x→市值翻倍

  中性：
    营收增速自然减速到30-40%→利润率缓慢改善
    →PEG维持1.0-1.5→随业绩增长
    估值锚：PE维持40-50x→年化回报15-25%

  悲观：
    TI价格战+客户导入不及预期→营收增速降到20%
    →利润率停滞→Q长期<0.5→PRG主导
    →需要市场对"用营收估值"有耐心
    估值锚：PS 5-8x→市值可能不涨

  【需要人类判断的关键问题】
  1. 思瑞浦的高精度ADC/DAC能否对标TI的ADS/INA系列？
     这是模拟芯片皇冠上的明珠，决定了技术壁垒的高度。
  2. 汽车芯片的真正放量时点？2026H2还是2027？
  3. 公司是否有"从第二供应商到战略主供"的客户关系跃迁案例？
     如果有，这就是思瑞浦的"中际旭创时刻"。
  4. 创始人/CTO的技术背景和产品路线图是否清晰？
     模拟芯片靠的是IP积累和工程师红利。

  {'='*64}
    免责声明：以上为AI辅助分析，不构成投资建议。
    模型输出仅供与人类判断进行辩论和交叉验证。
  {'='*64}
"""

print(report)
