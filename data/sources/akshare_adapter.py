"""AKShare data adapter — primary free data source for A-share market."""

from datetime import date, datetime
from typing import Optional

import akshare as ak
import pandas as pd

from data.sources.base import BaseAdapter, FetchResult


class AKShareAdapter(BaseAdapter):
    """AKShare adapter for A-share market data, macro, industry, and financials."""

    def name(self) -> str:
        return "akshare"

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
                df = ak.stock_zh_a_hist(
                    symbol=sym,
                    period="daily" if freq == "daily" else "weekly",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",  # 前复权
                )
                df["symbol"] = sym
                frames.append(df)
            except Exception as e:
                print(f"[AKShare] fetch_market_data failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"freq": freq, "symbol_count": len(frames)},
        )

    # ── Macro Data ───────────────────────────────────────────────────

    # Map our internal indicator names to AKShare function calls
    MACRO_INDICATOR_MAP = {
        "pmi_manufacturing": "macro_china_pmi",
        "pmi_non_manufacturing": "macro_china_non_man_pmi",
        "cpi": "macro_china_cpi_monthly",
        "ppi": "macro_china_ppi",
        "social_financing": "macro_china_shrzgm",
        "money_supply_m2": "macro_china_money_supply",
        "industrial_value_added": "macro_china_industrial_production",
        "fixed_asset_investment": "macro_china_fixed_asset_investment",
        "total_retail_sales": "macro_china_consumer_goods_retail",
        "industrial_profit": "macro_china_industrial_profit",
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
                print(f"[AKShare] Unknown macro indicator: {ind}")
                continue

            try:
                func = getattr(ak, func_name)
                df = func()
                results[ind] = df
            except Exception as e:
                print(f"[AKShare] fetch_macro_data failed for {ind}: {e}")

        return FetchResult(
            data=pd.DataFrame(),  # Multiple DFs stored in metadata
            source=self.name(),
            fetched_at=date.today(),
            metadata={"results": results, "indicators": list(results.keys())},
        )

    # ── Industry Data ────────────────────────────────────────────────

    # SW industry classification codes
    INDUSTRY_BOARD_MAP = {
        "sw_agriculture": "农林牧渔",
        "sw_mining": "采掘",
        "sw_chemical": "化工",
        "sw_steel": "钢铁",
        "sw_nonferrous": "有色金属",
        "sw_electronics": "电子",
        "sw_home_appliances": "家用电器",
        "sw_food_beverage": "食品饮料",
        "sw_textile_apparel": "纺织服装",
        "sw_light_manufacturing": "轻工制造",
        "sw_pharma": "医药生物",
        "sw_utilities": "公用事业",
        "sw_transport": "交通运输",
        "sw_real_estate": "房地产",
        "sw_commercial_trade": "商业贸易",
        "sw_leisure_services": "休闲服务",
        "sw_finance": "银行",
        "sw_nonbank_finance": "非银金融",
        "sw_auto": "汽车",
        "sw_mechanical_equipment": "机械设备",
        "sw_national_defense": "国防军工",
        "sw_computer": "计算机",
        "sw_media": "传媒",
        "sw_telecom": "通信",
        "sw_electric_equipment": "电气设备",
        "sw_building_materials": "建筑材料",
        "sw_building_decoration": "建筑装饰",
    }

    def fetch_industry_data(
        self,
        industry_codes: list[str],
        start_date: str = "20150101",
        end_date: Optional[str] = None,
    ) -> FetchResult:
        if end_date is None:
            end_date = datetime.today().strftime("%Y%m%d")

        frames = []
        for code in industry_codes:
            board_name = self.INDUSTRY_BOARD_MAP.get(code)
            if board_name is None:
                print(f"[AKShare] Unknown industry: {code}")
                continue

            try:
                df = ak.stock_board_industry_hist_em(
                    symbol=board_name,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
                df["industry_code"] = code
                frames.append(df)
            except Exception as e:
                print(f"[AKShare] fetch_industry_data failed for {code}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"industry_count": len(frames)},
        )

    # ── Financial Reports ────────────────────────────────────────────

    def fetch_financial_reports(
        self,
        symbols: list[str],
        report_type: str = "quarterly",
    ) -> FetchResult:
        frames = []
        for sym in symbols:
            try:
                df = ak.stock_financial_abstract_ths(symbol=sym)
                if df is not None and not df.empty:
                    df["symbol"] = sym
                    frames.append(df)
            except Exception as e:
                print(f"[AKShare] fetch_financial_reports failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"report_type": report_type, "symbol_count": len(frames)},
        )

    # ── Analyst Expectations ─────────────────────────────────────────

    def fetch_analyst_expectations(
        self,
        symbols: list[str],
    ) -> FetchResult:
        """Fetch analyst consensus ratings and target prices.

        Note: AKShare's analyst data is limited. For detailed consensus
        earnings estimates, Tushare's 'income_proj' API is recommended.
        """
        frames = []
        for sym in symbols:
            try:
                df = ak.stock_analyst_rank_ths(symbol=sym)
                if df is not None and not df.empty:
                    df["symbol"] = sym
                    frames.append(df)
            except Exception as e:
                print(f"[AKShare] fetch_analyst_expectations failed for {sym}: {e}")

        if not frames:
            return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

        combined = pd.concat(frames, ignore_index=True)
        return FetchResult(
            data=combined,
            source=self.name(),
            fetched_at=date.today(),
            metadata={"symbol_count": len(frames)},
        )

    # ── Industry PE/PB (valuation) ───────────────────────────────────

    def fetch_industry_valuation(self, date_str: Optional[str] = None) -> FetchResult:
        """Fetch industry-level PE/PB for all SW industries."""
        try:
            df = ak.stock_board_industry_pe_em()
            if df is not None and not df.empty:
                return FetchResult(
                    data=df,
                    source=self.name(),
                    fetched_at=date.today(),
                    metadata={},
                )
        except Exception as e:
            print(f"[AKShare] fetch_industry_valuation failed: {e}")

        return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

    # ── Money Flow ───────────────────────────────────────────────────

    def fetch_money_flow(
        self, start_date: str, end_date: str
    ) -> FetchResult:
        """Fetch north-bound capital flow (北向资金)."""
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            if df is not None and not df.empty:
                return FetchResult(
                    data=df,
                    source=self.name(),
                    fetched_at=date.today(),
                    metadata={},
                )
        except Exception as e:
            print(f"[AKShare] fetch_money_flow failed: {e}")

        return FetchResult(pd.DataFrame(), self.name(), date.today(), {})

    # ── Stock List ───────────────────────────────────────────────────

    def fetch_stock_list(self) -> list[dict]:
        """Fetch full A-share stock list with basic info."""
        try:
            df = ak.stock_info_a_code_name()
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"[AKShare] fetch_stock_list failed: {e}")
            return []
