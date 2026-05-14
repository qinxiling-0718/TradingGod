"""Backtest evaluation and reporting utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult


@dataclass
class EvaluationReport:
    """Structured backtest evaluation report."""

    result: BacktestResult
    summary: str
    metrics_table: pd.DataFrame
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    top_drawdowns: pd.DataFrame


def evaluate(result: BacktestResult) -> EvaluationReport:
    """Produce a full evaluation report from backtest results."""
    metrics = {
        "Total Return": f"{result.total_return:.2%}",
        "Annualized Return": f"{result.annualized_return:.2%}",
        "Annualized Volatility": f"{result.annualized_volatility:.2%}",
        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
        "Max Drawdown": f"{result.max_drawdown:.2%}",
        "Max DD Duration (days)": result.max_drawdown_duration,
        "Win Rate (weekly)": f"{result.win_rate:.2%}",
        "Average Turnover": f"{result.turnover_avg:.2%}",
        "Factor IC Mean": f"{result.ic_mean:.4f}",
        "Factor IC IR": f"{result.ic_ir:.2f}",
        "IC Positive Rate": f"{result.ic_positive_rate:.2%}",
    }

    metrics_table = pd.DataFrame(
        metrics.items(), columns=["Metric", "Value"]
    )

    # Monthly returns
    monthly = result.weekly_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )

    # Yearly returns
    yearly = result.weekly_returns.resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    )

    # Top 5 drawdowns
    cumulative = (1 + result.daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max

    dd_periods = _find_drawdown_periods(drawdowns)
    dd_df = pd.DataFrame(dd_periods).head(5)

    # Summary text
    summary = _build_summary(result)

    return EvaluationReport(
        result=result,
        summary=summary,
        metrics_table=metrics_table,
        monthly_returns=monthly,
        yearly_returns=yearly,
        top_drawdowns=dd_df,
    )


def _find_drawdown_periods(drawdowns: pd.Series) -> list[dict]:
    """Identify distinct drawdown periods."""
    periods = []
    in_dd = False
    dd_start = None
    dd_max = 0.0

    for dt, dd in drawdowns.items():
        if dd < -0.01 and not in_dd:
            in_dd = True
            dd_start = dt
            dd_max = dd
        elif in_dd:
            if dd < dd_max:
                dd_max = dd
            if dd >= -0.005:  # Recovery threshold
                periods.append({
                    "start": dd_start,
                    "end": dt,
                    "max_drawdown": f"{dd_max:.2%}",
                    "duration_days": (dt - dd_start).days,
                })
                in_dd = False

    if in_dd:
        periods.append({
            "start": dd_start,
            "end": drawdowns.index[-1],
            "max_drawdown": f"{dd_max:.2%}",
            "duration_days": (drawdowns.index[-1] - dd_start).days,
        })

    return sorted(periods, key=lambda x: x["max_drawdown"])


def _build_summary(result: BacktestResult) -> str:
    """Generate a human-readable summary of backtest results."""
    return (
        f"Backtest: {result.config.name}\n"
        f"Period: {result.config.start_date} → {result.config.end_date}\n"
        f"Return: {result.total_return:.2%} | Sharpe: {result.sharpe_ratio:.2f} | "
        f"Max DD: {result.max_drawdown:.2%}\n"
        f"IC Mean: {result.ic_mean:.4f} | IC IR: {result.ic_ir:.2f}\n"
    )
