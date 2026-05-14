"""Base adapter class for data sources. All adapters inherit from this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class FetchResult:
    """Standardized result from any data adapter."""
    data: pd.DataFrame
    source: str
    fetched_at: date
    metadata: dict


class BaseAdapter(ABC):
    """Abstract base for all data source adapters."""

    @abstractmethod
    def name(self) -> str:
        """Return the adapter name (e.g. 'akshare', 'tushare')."""
        ...

    @abstractmethod
    def fetch_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "daily",
    ) -> FetchResult:
        """Fetch OHLCV market data for given symbols."""
        ...

    @abstractmethod
    def fetch_macro_data(
        self,
        indicators: list[str],
        start_date: str,
        end_date: str,
    ) -> FetchResult:
        """Fetch macroeconomic indicators."""
        ...

    @abstractmethod
    def fetch_industry_data(
        self,
        industry_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> FetchResult:
        """Fetch industry-level data (revenue, profit, valuation)."""
        ...

    @abstractmethod
    def fetch_financial_reports(
        self,
        symbols: list[str],
        report_type: str = "quarterly",
    ) -> FetchResult:
        """Fetch financial report data."""
        ...

    @abstractmethod
    def fetch_analyst_expectations(
        self,
        symbols: list[str],
    ) -> FetchResult:
        """Fetch analyst consensus expectations (for expectation gap calc)."""
        ...
