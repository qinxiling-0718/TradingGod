# TradingGod — AI 景气度投资分析系统

基于"预期差 + 景气度"双引擎的 A 股 AI 产业链分析工具。

## 快速开始

```bash
git clone <repo-url> && cd TradingGod
uv sync
export TUSHARE_TOKEN="your_token"
```

## 每周工作流

```bash
# 1. 采集预测快照（积累修正序列，4周后分歧因子激活）
uv run python scripts/collect_snapshots.py

# 2. 生成行业分析报告（PEG排名 + 标注 + 盲点 + 陷阱）
uv run python scripts/generate_report.py

# 3. 个股深度分析
uv run python scripts/analyze_stock.py 002463                    # 单只
uv run python scripts/analyze_stock.py 002463 300502 688041      # 多股对比
```

## 首次使用

```bash
uv run python scripts/ingest_data.py        # SW指数 + 宏观
uv run python scripts/ingest_tushare.py     # 分析师预测
uv run python scripts/check_data.py         # 数据检查
```

## 核心因子

| 因子 | 公式 | 状态 |
|------|------|------|
| **PEG** | 行业PE / 共识增速 | ✓ |
| **PRG** | 营收增速信号 `(Rev.G-15%)×3` | ✓ |
| **混合PEG** | `Q×PEG + (1-Q)×PRG`, `Q=min(1.0,净利率×10)` | ✓ |
| **PEG陷阱** | 低PEG是价值陷阱还是结构性窗口？自动标注 | ✓ |
| **分歧因子** | 四象限共识追踪（Q1=强买入） | ✓ (需4+快照) |
| **质量门动量** | `sigmoid(static×3)` 衰减弱基本面的动量 | ✓ |

## 读取指标

```
Stock      PEG    G%     RevG%   Margin  Q     PE     PRG    Style
沪电股份   0.68   49.6%  42.0%   20.2%   1.00  33.7   +0.81  PEG主导
工业富联   1.03   26.0%  48.2%   3.9%    0.39  26.9   +1.00  PRG主导
寒武纪     0.93   59.2%  —       —       0.00  52.0   +1.00  PRG主导

PEG < 0.8: 低估 | PEG > 3.0: 充分定价 | Q=1.0: 纯PEG | Q<0.5: PRG主导
```

## 回测

```bash
uv run python scripts/run_pipeline_v4.py
```

v4 个股回测（37-100只AI股，PEG+PRG+质量门动量）：年化~47%，Sharpe ~1.06。

## 项目结构

```
TradingGod/
├── README.md                 # 本文件
├── CLAUDE.md                 # 设计哲学（AI上下文）
├── config/settings.yaml
├── data/sources/             # AKShare/Tushare适配器
├── data/store/               # DuckDB存储
├── factors/                  # 因子引擎（PEG/PRG/分歧/标注）
├── analysis/                 # 景气度 + 产业链推演
├── backtest/                 # 周频回测引擎
└── scripts/                  # 可执行脚本
```

## 分析输出解释

每份报告输出包含5组标注：
- **模型观察**：数据驱动的客观陈述
- **增长性质**：结构性/周期性/混合
- **叙事对齐**：数据领先/滞后于市场叙事
- **盲点标注**：模型无法判断的（周期vs结构、外部冲击）
- **人类判断问题**：需要人回答的关键问题
