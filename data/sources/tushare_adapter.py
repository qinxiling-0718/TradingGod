"""Tushare data adapter — supplementary source for financial & analyst data."""

import os
from datetime import date, datetime
from typing import Optional

import pandas as pd

from data.sources.base import BaseAdapter, FetchResult


class TushareAdapter(BaseAdapter):
    """Tushare adapter focused on financial reports and analyst expectations.

    Requires TUSHARE_TOKEN environment variable or token passed to constructor.
    Free tier is sufficient for MVP.
    """

    def __init__(self, token: Optional[str] = None):
        self._token = token or os.getenv("TUSHARE_TOKEN", "")
        self._pro = None

    @property
    def pro(self):
        if self._pro is None:
            import tushare as ts

            ts.set_token(self._token)
            self._pro = ts.pro_api()
        return self._pro

    def name(self) -> str:
        return "tushare"

    def is_ready(self) -> bool:
        """Check if Tushare is configured with a valid token."""
        return bool(self._token)

    # ── Market Data ──────────────────────────────────────────────────

    def fetch_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "daily",
    ) -> FetchResult:
        frames = []
        for sym in symbols:
            try:
                df = self.pro.daily(
                    ts_code=sym,
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
                if df is not None and not df.empty:
                    df["symbol"] = sym
                    frames.append(df)
            except Exception as e:
                print(f"[Tushare] fetch_market_data failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"freq": freq},
        )

    # ── Macro Data ───────────────────────────────────────────────────

    MACRO_INDICATOR_MAP = {
        "pmi": "china_pmi",
        "cpi": "china_cpi",
        "ppi": "china_ppi",
        "money_supply_m2": "china_money_supply",
        "social_financing": "china_social_financing",
    }

    def fetch_macro_data(
        self,
        indicators: list[str],
        start_date: str = "20150101",
        end_date: Optional[str] = None,
    ) -> FetchResult:
        if end_date is None:
            end_date = datetime.today().strftime("%Y%m%d")

        results = {}
        for ind in indicators:
            func_name = self.MACRO_INDICATOR_MAP.get(ind)
            if func_name is None:
                print(f"[Tushare] Unknown macro indicator: {ind}")
                continue

            try:
                func = getattr(self.pro, func_name)
                df = func(
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
                results[ind] = df
            except Exception as e:
                print(f"[Tushare] fetch_macro_data failed for {ind}: {e}")

        return FetchResult(
            data=pd.DataFrame(),
            source=self.name(),
            fetched_at=date.today(),
            metadata={"results": results},
        )

    # ── Industry Data ────────────────────────────────────────────────

    def fetch_industry_data(
        self,
        industry_codes: list[str],
        start_date: str = "20150101",
        end_date: Optional[str] = None,
    ) -> FetchResult:
        return FetchResult(
            data=pd.DataFrame(),
            source=self.name(),
            fetched_at=date.today(),
            metadata={"note": "Use sw_daily or index_classify for industry data"},
        )

    # ── Financial Reports (Tushare's strength) ───────────────────────

    def fetch_financial_reports(
        self,
        symbols: list[str],
        report_type: str = "quarterly",
    ) -> FetchResult:
        """Fetch income statement, balance sheet, cash flow data from Tushare.

        Tushare has superior fundamental data vs AKShare:
        - income: revenue, net profit, EPS, operating profit
        - balancesheet: total assets, equity, liabilities
        - cashflow: operating cash flow
        """
        frames = []
        tables = ["income", "balancesheet", "cashflow"]

        for sym in symbols:
            for table in tables:
                try:
                    func = getattr(self.pro, table)
                    df = func(
                        ts_code=sym,
                        period=self._get_period(report_type),
                    )
                    if df is not None and not df.empty:
                        df["symbol"] = sym
                        df["table"] = table
                        frames.append(df)
                except Exception as e:
                    print(f"[Tushare] fetch_{table} failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"report_type": report_type},
        )

    def _get_period(self, report_type: str) -> str:
        """Map report type to Tushare period filter."""
        mapping = {
            "quarterly": "",
            "annual": "",
        }
        return mapping.get(report_type, "")

    # ── Analyst Expectations (Tushare's key advantage) ───────────────

    def fetch_analyst_expectations(
        self,
        symbols: list[str],
    ) -> FetchResult:
        """Fetch analyst consensus earnings forecasts.

        This is the CORE data for expectation gap calculation.
        Tushare provides:
        - profit forecast tables (fina_indicator with forecast fields)
        - analyst ratings
        """
        frames = []
        for sym in symbols:
            try:
                df = self.pro.forecast(ts_code=sym)
                if df is not None and not df.empty:
                    df["symbol"] = sym
                    frames.append(df)
            except Exception as e:
                print(f"[Tushare] fetch_forecast failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={},
        )
