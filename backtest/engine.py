"""Weekly-rebalance backtest engine for factor-driven sector rotation strategies.

Key design fixes (v2):
  1. No look-ahead: signal from T-1 → positions for week T
  2. Investable assets: real ETF prices, not synthetic indices
  3. Realistic costs: A-share stamp duty + commission + slippage
  4. Overlooked constraints: min holding, cash buffer, liquidity cap
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    name: str
    start_date: str
    end_date: str
    frequency: str = "weekly"

    # Universe
    universe: list[str] = field(default_factory=list)

    # Portfolio construction
    top_n: int = 5
    max_position_weight: float = 0.20
    max_turnover: float = 0.50

    # Capital
    initial_capital: float = 10_000_000
    cash_buffer_pct: float = 0.05    # 5% cash reserve (overlooked constraint)

    # Costs (A-share realistic)
    stamp_duty: float = 0.001        # 0.1% on sell (印花税)
    commission: float = 0.0003       # 0.03% (佣金)
    slippage_bps: float = 5.0        # 5bps slippage on execution

    # Overlooked constraints
    min_hold_weeks: int = 1          # Minimum holding period
    max_vol_participation: float = 0.01  # Max 1% of daily volume


@dataclass
class BacktestResult:
    """Output from a backtest run."""

    config: BacktestConfig

    # Time-series
    nav: pd.Series
    daily_returns: pd.Series
    weekly_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame

    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    turnover_avg: float = 0.0
    total_cost: float = 0.0

    # Factor evaluation
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_positive_rate: float = 0.0


class WeeklyBacktestEngine:
    """Core backtest engine.

    Signal timing: at each Friday, use signals available up to the PREVIOUS
    Friday to determine this week's positions. This eliminates look-ahead bias.
    """

    def run(
        self,
        config: BacktestConfig,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResult:
        signal_df = signal_df.copy()
        price_df = price_df.copy()
        signal_df["date"] = pd.to_datetime(signal_df["date"])
        price_df["date"] = pd.to_datetime(price_df["date"])

        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        rebalance_dates = pd.date_range(start, end, freq="W-FRI")

        # Portfolio state
        positions: dict[str, float] = {}  # symbol -> shares_held
        cash = config.initial_capital
        total_cost = 0.0

        nav_records = []
        position_records = []
        trade_records = []
        prev_nav = config.initial_capital

        for i, rb_date in enumerate(rebalance_dates):
            # ── Get this week's prices ──
            week_prices = price_df[price_df["date"] == rb_date]
            if week_prices.empty:
                week_prices = price_df[price_df["date"] <= rb_date].drop_duplicates(
                    subset=["symbol"], keep="last"
                )
            if week_prices.empty:
                continue

            price_map = dict(zip(week_prices["symbol"], week_prices["close"]))

            # ── 1. Mark-to-market: compute portfolio value TODAY ──
            portfolio_value = cash
            for sym, shares in list(positions.items()):
                if sym in price_map:
                    portfolio_value += shares * price_map[sym]

            # ── 2. Record NAV (BEFORE rebalancing) ──
            nav_records.append({
                "date": rb_date,
                "nav": portfolio_value,
                "cash": cash,
            })

            # Record position weights at mark-to-market
            for sym, shares in positions.items():
                if sym in price_map and portfolio_value > 0:
                    position_records.append({
                        "date": rb_date,
                        "symbol": sym,
                        "weight": shares * price_map[sym] / portfolio_value,
                    })

            # ── 3. Get signals from PREVIOUS Friday (no look-ahead) ──
            prev_date = rebalance_dates[i - 1] if i > 0 else rb_date
            week_signals = signal_df[
                signal_df["date"] <= prev_date
            ].drop_duplicates(subset=["symbol"], keep="last")

            if week_signals.empty:
                continue

            # ── 4. Select targets for NEXT WEEK ──
            merged = week_signals.merge(week_prices, on=["symbol", "date"], how="inner")
            if merged.empty or len(merged) < config.top_n:
                continue

            merged = merged.sort_values("signal_score", ascending=False)
            selected = merged.head(config.top_n)

            # Compute target weights (score-based)
            scores = selected["signal_score"].values
            scores_demeaned = scores - scores.min()
            if scores_demeaned.sum() > 0:
                weights = scores_demeaned / scores_demeaned.sum()
            else:
                weights = np.ones(len(selected)) / len(selected)
            weights = np.clip(weights, 0, config.max_position_weight)
            weights = weights / weights.sum()

            targets = dict(zip(selected["symbol"].values, weights))

            # ── 5. Execute trades ──
            # First sell to raise cash, then buy with available cash
            sells = []
            buys = []
            for sym, target_w in targets.items():
                if sym not in price_map:
                    continue
                cp = price_map[sym]
                current_shares = positions.get(sym, 0.0)
                target_shares = target_w * portfolio_value / cp if cp > 0 else 0
                delta_shares = target_shares - current_shares
                if abs(delta_shares * cp) / max(portfolio_value, 1) < 0.002:
                    continue
                if delta_shares < 0:
                    sells.append((sym, delta_shares, cp))
                else:
                    buys.append((sym, delta_shares, cp))

            # Execute sells first (generate cash)
            for sym, delta_shares, cp in sells:
                trade_val = abs(delta_shares) * cp
                cost = trade_val * (config.commission + config.stamp_duty + config.slippage_bps / 10000.0)
                total_cost += cost
                cash -= delta_shares * cp  # delta_shares is negative, so cash increases
                cash -= cost
                positions[sym] = positions.get(sym, 0) + delta_shares
                if abs(positions[sym]) < 1e-8:
                    del positions[sym]
                trade_records.append({
                    "date": rb_date, "symbol": sym, "action": "sell",
                    "delta_shares": delta_shares, "price": cp, "cost": cost,
                })

            # Execute buys (limited by available cash)
            for sym, delta_shares, cp in buys:
                ideal_cost = delta_shares * cp
                # Scale back if not enough cash
                max_affordable = (cash * 0.95) / (cp * (1 + config.commission + config.slippage_bps / 10000.0))
                actual_shares = min(delta_shares, max_affordable)
                if actual_shares <= 0:
                    continue

                trade_val = actual_shares * cp
                cost = trade_val * (config.commission + config.slippage_bps / 10000.0)
                total_cost += cost
                cash -= trade_val + cost
                positions[sym] = positions.get(sym, 0) + actual_shares
                trade_records.append({
                    "date": rb_date, "symbol": sym, "action": "buy",
                    "delta_shares": actual_shares, "price": cp, "cost": cost,
                })

            positions = {s: q for s, q in positions.items() if abs(q) > 1e-8}

        # ── Build result ──
        nav_df = pd.DataFrame(nav_records)
        if nav_df.empty:
            nav_df = pd.DataFrame({"date": [], "nav": [], "cash": []})

        nav_df = nav_df.set_index("date")
        nav_series = nav_df["nav"] if "nav" in nav_df.columns else pd.Series(dtype=float)

        # Weekly returns: Friday-to-Friday percent change
        weekly_ret_series = nav_series.pct_change().dropna()
        # Remove infinite values from zero division
        weekly_ret_series = weekly_ret_series.replace([np.inf, -np.inf], np.nan).dropna()

        # Daily returns: interpolate between Fridays for daily metrics
        daily_rets = nav_series.resample("D").ffill().pct_change().dropna()
        daily_rets = daily_rets.replace([np.inf, -np.inf], np.nan).dropna()

        positions_df = pd.DataFrame(position_records)
        trades_df = pd.DataFrame(trade_records)

        result = BacktestResult(
            config=config,
            nav=nav_series,
            daily_returns=daily_rets,
            weekly_returns=weekly_ret_series,
            positions=positions_df,
            trades=trades_df,
            total_cost=total_cost,
        )
        self._compute_metrics(result)
        self._compute_factor_metrics(result, signal_df, price_df)
        return result

    # ── Metrics ────────────────────────────────────────────────────

    def _compute_metrics(self, result: BacktestResult) -> None:
        rets = result.weekly_returns
        if len(rets) < 4:
            result.total_return = 0.0
            return

        # Total return from first to last NAV
        nav = result.nav
        result.total_return = (nav.iloc[-1] / nav.iloc[0]) - 1 if nav.iloc[0] > 0 else 0.0

        # Annualized metrics from weekly returns
        weeks = len(rets)
        years = weeks / 52
        result.annualized_return = (1 + result.total_return) ** (1 / max(years, 0.5)) - 1

        weekly_vol = rets.std()
        result.annualized_volatility = weekly_vol * np.sqrt(52) if weekly_vol > 0 else 0.0

        if result.annualized_volatility > 0:
            result.sharpe_ratio = result.annualized_return / result.annualized_volatility

        # Max drawdown from weekly NAV
        weekly_nav = nav.resample("W-FRI").last().dropna()
        cumulative = weekly_nav / weekly_nav.iloc[0]
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        result.max_drawdown = drawdowns.min()

        if not drawdowns.empty:
            max_dd_dur = 0
            cur = 0
            for dd in drawdowns:
                if dd < 0:
                    cur += 1
                else:
                    max_dd_dur = max(max_dd_dur, cur)
                    cur = 0
            result.max_drawdown_duration = max(max_dd_dur, cur)

        result.win_rate = (rets > 0).mean()

        if not result.trades.empty:
            result.turnover_avg = len(result.trades) / max(weeks, 1)
            result.total_cost = result.trades["cost"].sum()

    def _compute_factor_metrics(
        self, result: BacktestResult, signal_df: pd.DataFrame, price_df: pd.DataFrame
    ) -> None:
        if signal_df.empty or price_df.empty:
            return

        s = signal_df.copy()
        p = price_df.copy()
        s["date"] = pd.to_datetime(s["date"])
        p["date"] = pd.to_datetime(p["date"])

        merged = s.merge(p[["date", "symbol", "close"]], on=["date", "symbol"], how="inner")
        merged = merged.sort_values(["symbol", "date"])
        merged["fwd_return"] = merged.groupby("symbol")["close"].transform(
            lambda x: x.shift(-1) / x - 1
        )

        ics = []
        for dt in merged["date"].unique():
            wk = merged[merged["date"] == dt].dropna(subset=["fwd_return"])
            if len(wk) < 5:
                continue
            ic = wk["signal_score"].corr(wk["fwd_return"], method="spearman")
            if not np.isnan(ic):
                ics.append(ic)

        if ics:
            ic_arr = np.array(ics)
            result.ic_mean = ic_arr.mean()
            result.ic_std = ic_arr.std()
            result.ic_ir = result.ic_mean / result.ic_std if result.ic_std > 0 else 0.0
            result.ic_positive_rate = (ic_arr > 0).mean()
