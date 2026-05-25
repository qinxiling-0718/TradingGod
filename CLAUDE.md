# TradingGod — AI 驱动的景气度投资分析 Agent

## 项目身份

TradingGod 是一个**封装了"预期差 + 景气度"分析方法论的 AI Agent**，不是一个传统的量化交易软件。

- **投资方向**：AI 产业链（限定赛道）
- **市场**：仅聚焦 A 股
- **核心哲学**：**预期差做动，景气度做基**
- **回测频率**：周频为主（不同因子可有不同的最优观测频率）
- **数据源**：AKShare（主，免费）+ Tushare（补充，SW行业PE）+ THS（分析师预测）
- **交互形态**：Agent + Skills + Scripts 三层协作
- **当前阶段**：因子验证期 — PEG + 修正序列 + 元数据标注 + 回测闭环可运行

### 我们构建的不是软件，是 Agent

```
┌──────────────────────────────────────────────────────┐
│  Agent 层  →  分析报告 + 盲点标注 + 问题供人辩论        │
│  Skills 层 →  PEG因子 + 产业链推演 + 元数据标注        │
│  Scripts 层 → 数据采集 + 因子计算 + 回测              │
└──────────────────────────────────────────────────────┘
```

| | 传统量化软件 | TradingGod Agent |
|---|---|---|
| **因子** | 固定公式 + 固定权重 | 可注册因子 + 动态影响力（区间约束） |
| **决策** | 自动化信号 → 执行 | 模型输出 + 标注 + 人类辩论 |
| **输出** | Dashboard/报告 | 分析过程 + 置信度 + 盲点 + "不知道什么" |
| **核心** | 代码/数据/算力 | **分析方法论**——这是代码化不出来的壁垒 |

### 技术壁垒三层模型

```
┌─────────────────────────────────────────────┐
│  第三层：Agent 壁垒（最有价值）                │
│  方法论的封装：不是给答案，而是给出分析过程、    │
│  置信度、风险边界、和"目前不知道什么"          │
├─────────────────────────────────────────────┤
│  第二层：方法壁垒                             │
│  什么因子在什么阶段有效、产业链传导验证方法、    │
│  交叉验证框架、因子生命周期管理               │
├─────────────────────────────────────────────┤
│  第一层：数据壁垒                             │
│  长时间积累的清洗后高频跟踪数据 + 一致预期历史   │
│  数据源大家都能买，但清洗对齐后的因子级数据是壁垒 │
└─────────────────────────────────────────────┘
```

---

## 投资哲学（双引擎模型）

### 核心公式

```
预期差 = 现实（或高频跟踪值） - 市场一致预期
Alpha  = f(景气度方向 × 预期差变化)
```

### 引擎一：景气度 = 基座 = 频率（"基"）

景气度回答："这个赛道在走上坡路还是下坡路？"

**核心洞察：景气度本质上是"频率"概念**：

- **太高频**（日频）：噪音淹没信号
- **太低频**（季度/年度）：等确认拐点，市场早定价完了
- **刚刚好**（周频）：在"市场开始意识到但还没充分定价"的窗口内捕捉

不同因子的**最优观测频率不同**：

| 因子类型 | 变化速度 | 最优观测频率 | 市场定价速度 |
|---------|---------|------------|------------|
| 宏观 PMI 预期差 | 月度 | 月频/周频 | 慢（1-3 月） |
| 行业盈利预期差 | 季度财务 + 周度高频率验证 | 周频 | 中等（2-4 周） |
| 资金流向预期差 | 日频 | 周频聚合 | 快（1-5 天） |
| 估值匹配预期差 | 慢变 | 月频 | 慢 |

**频率不是全局参数，而是每个因子的属性。**

### 引擎二：预期差 = 动能（"动"）

预期差回答："市场有没有充分定价这个趋势？"

超额收益不是来自"景气度上行"本身，而是来自"**市场尚未充分定价的景气度上行**"。

**如果市场不笨，就没有预期差了。** 预期差存在的条件是：
- 信息不对称：通过高频跟踪数据比市场更早感知到变化
- 认知惯性：市场对新信息的定价需要时间
- 情绪偏差：短期噪音让市场暂时偏离基本面趋势

### 修正序列 — 预期差的时间维度

预期差不是静态快照，而是**预期从 t₀ 到 t₁ 的移动过程**：

```
t₀: 市场预期某行业 2026 EPS = 40
t₁: 北美云厂 CapEx 超预期 → 高频数据显示订单爆满
t₂: 分析师开始上调预期 → 40 → 45 → 52
t₃: 市场还没有完全定价 52 → 预期差存在
t₄: 市场充分定价 → 预期差消失
```

**我们积累的就是 t₀→t₄ 的完整修正时间序列。** 这是真正的数据壁垒——不是买来的数据，是自己积累的。

### 双引擎决策矩阵

| | 景气上行 | 景气下行 |
|---|---|---|
| **预期差扩大** | **重仓（最佳买点）** | 反弹交易（短促，轻仓） |
| **预期差收敛** | 持有/逐步减仓 | 清仓/回避 |

### 周期 vs 结构 — 模型无法判断，但必须标注

模型无法区分"20% 增长是结构性还是周期性"，在数学上完全相同。但投资含义完全不同。这是 Agent 层最重要的标注之一——**模型不知道的事必须明确告诉人类**。

---

## 核心设计哲学

### 动态影响力 + 模糊空间

**权重可以在区间内浮动，但不能超出边界。**

```
传统：综合得分 = Σ(固定权重_i × 因子得分_i)
我们：综合得分 = Σ(权重函数_i(市场状态) × 因子得分_i)
      其中 权重函数_i ∈ [w_min, w_max]  ← 模糊空间约束
```

模糊空间的好处：给适应性、防过拟合、允许犯错、可审计。

### 元因子框架 (Meta-Factor)

元因子是决定"某个因子在当前环境下该不该被信任"的条件变量：

| 元因子 | 作用 | 影响逻辑 |
|--------|------|---------|
| 波动率状态 | 调节资金流因子的权重 | 高波动 → 资金流噪音大 → 衰减 |
| 周期位置 | 调节宏观因子的权重 | 拐点附近 → 宏观因子权重放大 |
| 因子共识度 | 调节所有因子的权重 | 多因子方向一致 → 共振放大 |
| 数据新鲜度 | 调节各因子的置信度 | 数据越旧 → 置信度衰减 |

### PEG — AI 赛道最核心的估值-预期复合因子

```
PEG = Sector PE / Consensus Growth Rate

ΔPEG < 0 + ΔG > 0  → 增长加速、估值未跟上 → 预期差扩大 → 买入
ΔPEG > 0 + ΔG → 0  → 估值在提前透支增长 → 预期差消失 → 卖出
```

PEG 比 PE 更适合 AI 赛道因为：AI 股票的 PE 差异极大（20x-500x），但 PEG 把增长纳入分母后具有可比性。中际旭创 PE 可能很高但 PEG=0.6——因为 48% 的增速让高 PE 变得便宜。

### 分歧因子 — 保守主义偏见 = 机会的指纹

**极端分析师分歧不是噪音，而是尚未被定价的预期差地图。**

当研究圈存在系统性的保守主义偏见（如"代工厂不值钱""存储是周期品""光模块不可持续"），分析师共识会分裂成两个阵营。分歧越大，两套框架的对抗越激烈——其中一套将被证明是错的，而证明的那一刻就是 Alpha。

**四大象限**：

```
                        共识均值 Δ(↑)
                         │
          Q1: FORMING    │    Q2: DEBATING
          分歧收窄+均值上移│    分歧扩大+均值上移
          保守派在认输    │    多空在激化
          ★ 最强买入      │    ◆ 观察（Q1的前夜）
          ────────────────┼────────────────
          分歧收窄+均值下移│    分歧扩大+均值下移
          乐观派在撤退    │    恐慌在扩散
          ▼ 卖出          │    ★★ 最强卖出
          Q3: DETERIORATING│   Q4: PANICKING
                         │
                        共识均值 Δ(↓)
```

- **Q1 (CONSENSUS FORMING)**：分歧收窄 + 均值上移 = 保守派在认输、数据在证实乐观派 → 最强买入
- **Q2 (DEBATE INTENSIFYING)**：分歧扩大 + 均值上移 = 乐观派更乐观但保守派更保守 → 这是 Q1 的前夜
- **Q3 (CONSENSUS DETERIORATING)**：分歧收窄 + 均值下移 = 乐观派在撤退 → 卖出
- **Q4 (PANIC SPREADING)**：分歧扩大 + 均值下移 = 所有人跑路 → 最强卖出

分歧因子和 PEG 因子互补：
- PEG 告诉你"定价偏离了多少"（静态）
- 分歧因子告诉你"共识在往哪个方向走"（动态）
- PEG=0.68 + Q1(FORMING) = 最安全的重仓时刻
- PEG=3.30 + Q4(PANICKING) = 远离

分歧因子需要 **4 周预测快照修正序列**积累后激活。

---

## 系统架构

### Agent + Skills + Scripts 三层协作

```
┌──────────────────────────────────────────────────┐
│              Agent 层（分析方法论的载体）            │
│  generate_report.py → PEG排名 + 标注 + 问题         │
│  供人类辩论: "这个信号在周期/结构维度上成立了多少？   │
│               "模型标注的盲点是否有新的信息填补？"     │
├──────────────────────────────────────────────────┤
│              Skills 层（可组合的分析工具）           │
│  peg_factor.py          → PEG计算 + ΔPEG动态修正   │
│  dispersion_factor.py   → 分歧因子 + 四象限信号     │
│  metadata_annotator.py  → 盲点 + 风险 + PEG陷阱    │
│  deduction.py           → AI产业链推演              │
│  influence.py           → 动态影响力+模糊空间       │
│  prosperity.py          → 景气度聚合评分            │
├──────────────────────────────────────────────────┤
│              Scripts 层（数据 + 计算 + 回测）       │
│  ingest_*.py            → 数据管线                 │
│  collect_snapshots.py   → 每周预测快照采集          │
│  run_pipeline_v3.py     → 因子计算 + 回测           │
│  generate_report.py     → 分析报告（Agent输出）     │
│  analyze_stock.py        → 个股深度分析 + PEG陷阱    │
│  compare_stocks.py       → 多股对比 + 头对头比较     │
├──────────────────────────────────────────────────┤
│              Data 层                             │
│  DuckDB (trading_god.duckdb)                     │
│  AKShare → SW指数/ETF/宏观                        │
│  Tushare → SW行业PE                                │
│  THS     → 分析师盈利预测                          │
└──────────────────────────────────────────────────┘
```

---

## 程序用法

### 环境

```bash
uv sync                          # 安装依赖
export TUSHARE_TOKEN="your_token" # 设置 Tushare Token
```

### 数据管线（一次性/按需）

```bash
# 拉取 SW 行业指数 + 宏观数据
uv run python scripts/ingest_data.py

# 拉取 Tushare/THS 分析师预测数据
uv run python scripts/ingest_tushare.py

# 检查数据质量
uv run python scripts/check_data.py
```

### 每周工作流

```bash
# 1. 每周五收盘后采集预测快照（修正序列积累）
uv run python scripts/collect_snapshots.py

# 2. 生成分析报告（PEG排名 + 标注 + 盲点）
uv run python scripts/generate_report.py

# 3. 个股/多股深度分析
uv run python scripts/analyze_stock.py 002463                    # 单只
uv run python scripts/analyze_stock.py 002463 300502 688041      # 多股对比

# 4. （可选）跑回测验证
uv run python scripts/run_pipeline_v3.py
```

### 脚本说明

| 脚本 | 用途 | 频率 |
|------|------|------|
| `ingest_data.py` | SW 行业指数 + 宏观数据 → DuckDB | 首次 + 按需更新 |
| `ingest_tushare.py` | Tushare/THS 分析师预测数据 → DuckDB | 首次 + 按需更新 |
| `collect_snapshots.py` | 采集 100+ AI 股票的 THS 预测快照 | **每周五** |
| `check_data.py` | 数据完整性检查 | 按需 |
| `run_pipeline.py` | v2 管线：ETF + 动量因子回测 | 按需 |
| `run_pipeline_v3.py` | v3 管线：PEG因子 + 回测 | 按需 |
| `generate_report.py` | Agent 输出：PEG排名 + 标注 + 盲点 | 每周 |

### 当前数据资产

| 表 | 行数 | 内容 |
|---|------|------|
| `sw_index_daily` | 84,428 | 17 个 SW 行业日线（1999-2026） |
| `etf_prices_daily` | 20,455 | 12 只行业 ETF 日线 |
| `analyst_forecast` | 114 | 38 只核心 AI 股 + 3 年预测 |
| `forecast_snapshots` | 640 | 1 次快照 × 107 只股票（积累中） |
| `sw_industry_pe_daily` | 16,347 | 5 个 SW 行业 PE/PB 日数据 |
| `macro_*` | 1,398 | PMI/CPI/PPI/M2/社融 |
| `stock_list` | 5,515 | A 股全量代码 |

---

## 当前因子体系

### 分歧因子（动态）— `factors/dispersion_factor.py`

基于预测修正序列的四象限信号（需 4+ 周快照积累后激活）。

```
对于每只股票，追踪 4 次快照的：
  Δconsensus  = 共识均值的%变化
  Δdispersion = 分歧度的绝对变化
  Δanalysts   = 覆盖分析师人数变化

四象限分类 → signal_score [-1.0 to +1.0] + confidence
```

### PEG 因子（核心）— `factors/peg_factor.py`

```
PEG = Sector PE / Consensus Growth Rate

静态信号（当前可用）：
  peg_signal = -0.40 × PEG_z + 0.40 × Growth_z + 0.20 × Quality_z
               ↑ 低PEG加分       ↑ 高增长加分      ↑ 低分歧加分

动态信号（4周修正序列积累后激活）：
  peg_signal += -0.30 × ΔPEG_z + 0.30 × ΔGrowth_revision_z
                ↑ PEG缩小加分      ↑ 增速上修加分
```

### 混合 PEG（PRG 自动接管）— `factors/peg_factor.py`

```
Q = min(1.0, 净利率 × 10)    ← 利润质量权重（全自动，无主观判断）

Hybrid_Signal = Q × PEG_Signal + (1-Q) × PRG_Signal

Q = 1.0（净利率>10%）：纯 PEG，成熟公司
Q = 0.5（净利率=5%）： PEG和PRG各半
Q = 0.0（无利润）：   纯 PRG，营收增速是唯一信号
```

PRG 信号基于营收增速：`PRG = clamp((Rev.G - 15%) × 3, -1, 1)`。当公司利润率薄时自动调节——营收爆发型的早期公司不会被 PEG 误判。

### 当前 PEG 排名（2026-05-13）

| # | 行业 | PEG | 增速 | 信号 | 增长性质 |
|---|------|-----|------|------|---------|
| 1 | 计算机设备 | 1.1 | 37.9% | +0.59 | 结构性 |
| 2 | 通信设备 | 0.9 | 38.2% | +0.55 | 结构性 |
| 3 | 电子 | 1.1 | 35.4% | +0.40 | 结构性 |
| 4 | 新能源 | 1.4 | 20.7% | +0.00 | 周期性 |
| 5 | 汽车 | 1.8 | 16.8% | -0.21 | 周期性 |
| 6 | 半导体 | 3.3 | 26.7% | -0.27 | 结构性/已定价 |
| 7 | 传媒 | 3.4 | 19.7% | -0.55 | 周期性 |
| 8 | 国防军工 | 2.9 | 10.5% | -0.97 | 周期性 |

### 元数据标注 — `factors/metadata_annotator.py`

每个行业输出带 6 组标注：
- **模型观察**：数据驱动的客观陈述
- **增长性质**：结构性 / 周期性 / 混合 / 不确定
- **叙事对齐**：领先 / 确认 / 发散 / 滞后
- **盲点**：模型看不到什么（地缘风险、客户集中度、政策、周期判断）
- **风险标记**：高 PEG、高分歧、低增长等警示
- **人类判断问题**：需要人回答的关键问题

### 其他因子（辅助）

| 因子 | 模块 | 说明 |
|------|------|------|
| 趋势偏离 | v2 pipeline | 价格 vs 52周 EMA |
| 行业相对强度 | v2 pipeline | 截面 z-score 排名 |
| 宏观匹配度 | v2 pipeline | PMI 趋势 × 行业 beta |
| 产业链推演 | `deduction.py` | AI 产业链上下游信号传播 |

---

## 模块详解

### 数据层 (`data/`)

#### AKShare 适配器 — `data/sources/akshare_adapter.py`
主力免费数据源：行业板块行情、宏观指标、估值、资金流向、财务摘要。内置 SW 行业名称映射。

#### Tushare 适配器 — `data/sources/tushare_adapter.py`
补充数据源：SW 行业 PE/PB（`sw_daily`）、财报、分析师预测（需高权限）。

#### DuckDB 存储 — `data/store/duckdb_store.py`
零配置单文件分析型数据库。`write_df` / `read_df` / `query` 三个核心方法。

### 因子引擎 (`factors/`)

| 模块 | 说明 |
|------|------|
| `registry.py` | 因子注册框架 + 六大维度定义 |
| `expectation_gap.py` | 预期差计算管道（raw→zscore→momentum→signal） |
| `expectation_factors.py` | 基于分析师预测的预期差因子（横截面） |
| `peg_factor.py` | **PEG 因子**：静态排名 + ΔPEG 动态修正 |
| `dispersion_factor.py` | **分歧因子**：四象限共识追踪（需4+周快照） |
| `influence.py` | 动态影响力函数 + 模糊空间 + 元因子 |
| `metadata_annotator.py` | **元数据标注**：盲点/风险/PEG陷阱/问题生成 |

### 分析引擎 (`analysis/`)

| 模块 | 说明 |
|------|------|
| `prosperity.py` | 景气度评分（6维聚合→行业排名） |
| `deduction.py` | AI 产业链推演（12节点×4层×上下游传播） |

### 回测引擎 (`backtest/`)

| 模块 | 说明 |
|------|------|
| `engine.py` | 周频回测：无前视偏差、ETF 实盘、A 股真实成本、现金约束 |
| `evaluation.py` | 绩效评估：Sharpe/回撤/IC-IR/归因报告 |

---

## 因子校准哲学

### 核心原则

1. **模糊空间优先**：定义因子的有效权重区间 `[w_min, w_max]`，不追求精确最优权重
2. **IC 只是参考，不是真理**：IC 高可能是过拟合，IC 低可能是市场在过渡期。不要用 IC 驱动因子调整
3. **修正序列是校准源**：当分析师共识上修时信号应为正，下修时信号应为负——这个方向正确性比 IC 数值重要
4. **PEG 的"G"质量比 PEG 数值重要**：38% 的 AI 结构性增速 vs 38% 的化工供给侧反弹——PEG 相同，含义完全不同
5. **允许犯错，追踪犯错**：记录每次因子信号与后续实际走势的偏差，但不立即调整因子。偏差积累到形成模式时再调整

### 校准节奏

```
每周：  collect_snapshots.py → 积累修正序列
       generate_report.py → 阅读标注，记录人类判断

每月：  检查 ΔPEG 方向 vs 后续行业收益方向
       如果系统性偏差 → 讨论因子定义是否需要调整
       如果是随机噪音 → 不做调整

每季度：回顾因子权重区间是否合理
        回顾元因子敏感度是否需要修正
```

### 当前已知的因子缺陷

| 缺陷 | 影响 | 应对 |
|------|------|------|
| 无法判断周期/结构 | 周期性高增长被误判为预期差 | 人工标注 growth_nature |
| 静态快照无时间序列 | 无法计算 ΔPEG | 积累 4 周后自动解决 |
| 行业映射不完整 | 部分股票落入 "other" | 扩展 INDUSTRY_TO_SECTOR |
| 外部冲击不可预测 | 贸易战/战争/政策突变 | 标注 blind_spots 提醒 |

---

## 配置 (`config/settings.yaml`)

```yaml
data:
  primary_source: akshare
  db_path: "data/trading_god.duckdb"

factors:
  windows: {short: 12, medium: 26, long: 52}
  expectation_gap:
    signal_threshold: 1.0
    decay_factor: 0.85

backtest:
  default_frequency: "weekly"
  benchmark: "000300"
  initial_capital: 10000000
  constraints:
    max_position_pct: 0.15
    max_sector_pct: 0.30
    max_turnover: 0.50

analysis:
  deduction:
    chain_depth: 3
    confidence_boost: 1.5
```

---

## 开发约定

- 所有核心逻辑通过 dataclass 定义数据结构
- 计算模块无副作用：输入 DataFrame/Series → 输出结果对象
- 参数从 YAML 加载，Token 从环境变量加载
- **模糊空间约束**：权重/阈值先定义区间而非精确值

### 运行方式

```bash
uv run python script_name.py    # 运行脚本
uv run jupyter notebook         # 交互式分析
uv run pytest                   # 测试
```

---

## 项目文件树

```
TradingGod/
├── CLAUDE.md                       # 本文件
├── pyproject.toml                  # 项目配置 & 依赖
├── config/settings.yaml            # 全局参数
│
├── data/
│   ├── sources/
│   │   ├── base.py                 # 数据源抽象基类
│   │   ├── akshare_adapter.py      # AKShare 适配器
│   │   └── tushare_adapter.py      # Tushare 适配器
│   └── store/
│       └── duckdb_store.py         # DuckDB 本地存储
│
├── factors/                        # 因子引擎
│   ├── registry.py                 # 因子注册 + 维度管理
│   ├── expectation_gap.py          # 预期差计算器（原始管道）
│   ├── expectation_factors.py      # 分析师预期差因子（横截面）
│   ├── peg_factor.py               # PEG因子（静态+动态修正）
│   ├── dispersion_factor.py        # 分歧因子（四象限共识追踪）
│   ├── influence.py                # 动态影响力 + 模糊空间 + 元因子
│   └── metadata_annotator.py       # 元数据标注（Agent上下文）
│
├── analysis/
│   ├── prosperity.py               # 景气度评分聚合
│   └── deduction.py                # AI产业链逻辑推演
│
├── backtest/
│   ├── engine.py                   # 周频回测引擎（ETF实盘+真实成本）
│   └── evaluation.py               # 绩效评估与归因
│
├── scripts/
│   ├── ingest_data.py              # SW指数 + 宏观数据
│   ├── ingest_tushare.py           # Tushare/THS分析师预测
│   ├── collect_snapshots.py        # 每周预测快照采集
│   ├── check_data.py               # 数据质量检查
│   ├── run_pipeline.py             # v2管线（ETF+动量回测）
│   ├── run_pipeline_v3.py          # v3管线（PEG因子+回测）
│   ├── generate_report.py          # 分析报告（Agent输出）
│   ├── analyze_stock.py            # 个股深度分析 + PEG陷阱检测
│   └── compare_stocks.py           # 多股对比 + 头对头比较
│
└── notebooks/                      # Jupyter Notebook（分析实验）
```

---

## 优化路线图

### 短期（当前~4周）

1. **积累修正序列** — 每周跑 `collect_snapshots.py`，2 周后分歧因子激活（快照 2/4）
2. **激活分歧因子** — 4 次快照后，四象限共识追踪自动启动
3. **扩展股票覆盖** — 从 100 只扩展到 150-200 只（重点覆盖中市值 AI 标的）
4. **完善行业映射** — 减少 "other" 分类，提高行业分类准确率

### 中期（1-3 月）

5. **产业链传导验证** — 用真实数据验证 deduction.py 中 AI 链的传播逻辑
6. **引入高频跟踪因子** — 光模块出货量、台积电月度营收等周频数据
7. **因子共识共振** — 当 PEG + 动量 + 宏观三个因子方向一致时放大信号
8. **截面 IC 稳定性监控** — 追踪每周 IC，建立 IC 的统计分布

### 长期（3-6 月）

9. **Tushare 高权限** — 通过积分/付费获取 `forecast`/`income` API，直接用一致预期数据
10. **Web 界面** — Agent 对话式分析 UI
11. **因子生命周期管理** — 自动检测因子失效、自动标记需要重新校准的因子
12. **扩展到非 AI 赛道** — 新能源、消费、医药等轮动逻辑复用

---

## 验证状态

| 检查项 | 状态 |
|--------|------|
| Python 3.14.5 + uv 0.11.12 | OK |
| 全部依赖安装 | OK |
| SW 行业指数数据 | OK (84K rows) |
| ETF 价格数据 | OK (20K rows) |
| 宏观数据 | OK (6 组) |
| Tushare SW PE 数据 | OK (16K rows) |
| THS 分析师预测 | OK (107 stocks) |
| 预测快照系统 | OK (4 snapshots, activated) |
| PEG 因子（静态） | OK |
| 混合 PEG（PRG自动接管）| OK |
| PEG 陷阱检测 | OK (LOW/HIGH PEG trap) |
| 分歧因子（四象限） | OK (activated, 11 stocks with 4+ weeks) |
| 元数据标注 | OK |
| 个股深度分析 | OK (`analyze_stock.py`) |
| 多股对比分析 | OK (`analyze_stock.py` multi) |
| 分析报告生成 | OK |
| 回测引擎（无前视偏差） | OK |
| 修正序列（ΔPEG + 分歧） | OK |
