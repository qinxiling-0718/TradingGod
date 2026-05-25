# TradingGod — AI 景气度投资分析系统

基于"预期差 + 景气度"双引擎的 AI 产业链 A 股分析工具。

## 快速开始

```bash
git clone <repo-url>
cd TradingGod
uv sync
export TUSHARE_TOKEN="your_token"
```

## 每周工作流

每周五收盘后，按顺序执行：

### 1. 采集预测快照（积累修正序列）

```bash
uv run python scripts/collect_snapshots.py
```

采集 100+ AI 产业链股票的 THS 分析师预测数据。累积 4 周后，分歧因子和 ΔPEG 自动激活。

### 2. 生成分析报告

```bash
uv run python scripts/generate_report.py
```

输出 PEG 排名 + 元数据标注 + PEG 陷阱检测 + 盲点 + 人类判断问题。

### 3. 个股深度分析

```bash
# 单只股票
uv run python scripts/analyze_stock.py 002463

# 多只股票对比
uv run python scripts/analyze_stock.py 002463 300502 688041 601138
```

输出 PEG、PRG、利润质量 Q、分歧象限、PEG 陷阱、最近季度财务趋势。

### 4. 回测验证（可选）

```bash
uv run python scripts/run_pipeline_v3.py
```

## 数据管线（首次使用）

```bash
# SW 行业指数 + 宏观数据
uv run python scripts/ingest_data.py

# Tushare/THS 分析师预测
uv run python scripts/ingest_tushare.py

# 数据质量检查
uv run python scripts/check_data.py
```

## 核心因子

| 因子 | 说明 | 状态 |
|------|------|------|
| **PEG** | 行业PE / 共识增速。低PEG+高增速=预期差 | 激活 |
| **PRG** | 基于营收增速的估值信号。利润率薄时自动接管 | 激活 |
| **混合PEG** | Q × PEG + (1-Q) × PRG。Q=min(1.0, 净利率×10) | 激活 |
| **PEG陷阱** | LOW PEG 是价值陷阱还是结构性窗口？自动标注 | 激活 |
| **分歧因子** | 四象限共识追踪。Q1=保守派认输=最强买入 | 需4+快照 |
| **ΔPEG** | PEG的时间序列变化。ΔPEG<0 + ΔG>0 = 买入 | 需4+快照 |

## 分析输出解读

```
Stock      PEG    G%     RevG%   Margin  Q     PE     PRG    Style
沪电股份   0.68   49.6%  42.0%   20.2%   1.00  33.7   +0.81  PEG主导
工业富联   1.03   26.0%  48.2%   3.9%    0.39  26.9   +1.00  PRG主导
寒武纪     0.93   59.2%  1392%   -4854%  0.00  52.0   +1.00  PRG主导
```

- **PEG < 0.8**：市场可能低估了增速
- **PEG > 3.0**：增长已被充分甚至过度定价
- **Q = 1.0**：纯 PEG 估值（成熟公司）
- **Q < 0.5**：PRG 主导（早期/薄利公司）

## 项目结构

```
TradingGod/
├── README.md                    # 本文件
├── CLAUDE.md                    # 设计哲学（面向AI上下文）
├── config/settings.yaml         # 全局参数
├── data/
│   ├── sources/                 # AKShare/Tushare 数据适配器
│   └── store/                   # DuckDB 本地存储
├── factors/                     # 因子引擎
│   ├── peg_factor.py           # PEG + 混合PEG
│   ├── dispersion_factor.py    # 分歧因子（四象限）
│   ├── influence.py            # 动态影响力 + 模糊空间
│   └── metadata_annotator.py   # 元数据标注 + PEG陷阱
├── analysis/                    # 分析引擎
│   ├── prosperity.py           # 景气度评分
│   └── deduction.py            # AI产业链推演
├── backtest/                    # 回测引擎
│   ├── engine.py               # 周频回测（ETF实盘+真实成本）
│   └── evaluation.py           # 绩效评估
└── scripts/                     # 可执行脚本
    ├── collect_snapshots.py    # 每周快照采集
    ├── generate_report.py      # 分析报告
    ├── analyze_stock.py        # 个股/多股分析
    ├── ingest_data.py          # SW指数+宏观
    └── ingest_tushare.py       # 分析师预测
```
