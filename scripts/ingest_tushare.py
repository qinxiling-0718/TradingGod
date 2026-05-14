"""Tushare + THS data ingestion — focused on core AI stocks.

Fetches analyst consensus EPS from THS for top ~40 AI supply chain stocks.
This is the core data for expectation gap computation.

Usage:  uv run python scripts/ingest_tushare.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore

TUSHARE_TOKEN = "51c1318e74cde3b12e99eced7b536b2983faa25866308634823f7726"
DB_PATH = "data/trading_god.duckdb"
CALL_DELAY = 1.5

# Top ~40 AI supply chain stocks by relevance
# Format: (ts_code, name, label)
AI_STOCKS = [
    # ── Semiconductor Equipment ──
    ("002371.SZ", "北方华创", "半导体设备"),
    ("688012.SH", "中微公司", "半导体设备"),
    ("688082.SH", "盛美上海", "半导体设备"),
    # ── Semiconductor Fab/Design ──
    ("688981.SH", "中芯国际", "晶圆代工"),
    ("688256.SH", "寒武纪", "AI芯片"),
    ("603986.SH", "兆易创新", "存储芯片"),
    ("002049.SZ", "紫光国微", "FPGA芯片"),
    ("688008.SH", "澜起科技", "内存接口"),
    ("603501.SH", "韦尔股份", "CIS芯片"),
    # ── Optical Modules ──
    ("300502.SZ", "新易盛", "光模块"),
    ("300308.SZ", "中际旭创", "光模块"),
    ("300394.SZ", "天孚通信", "光器件"),
    # ── PCB/Assembly ──
    ("002463.SZ", "沪电股份", "PCB"),
    ("002916.SZ", "深南电路", "PCB"),
    ("300476.SZ", "胜宏科技", "PCB"),
    # ── AI Server/Computing ──
    ("000977.SZ", "浪潮信息", "AI服务器"),
    ("002415.SZ", "海康威视", "AI视觉"),
    ("002236.SZ", "大华股份", "AI视觉"),
    # ── AI Software/Platform ──
    ("002230.SZ", "科大讯飞", "AI平台"),
    ("688111.SH", "金山办公", "AI应用"),
    ("300033.SZ", "同花顺", "AI金融"),
    ("688369.SH", "致远互联", "协同办公"),
    # ── Data Center/Power ──
    ("002335.SZ", "科华数据", "数据中心"),
    ("300383.SZ", "光环新网", "数据中心"),
    ("600845.SH", "宝信软件", "工业互联网"),
    # ── Auto/Autonomous Driving ──
    ("002920.SZ", "德赛西威", "自动驾驶"),
    ("300750.SZ", "宁德时代", "动力电池"),
    ("601689.SH", "拓普集团", "汽车电子"),
    # ── More Semiconductors ──
    ("603160.SH", "汇顶科技", "指纹芯片"),
    ("300458.SZ", "全志科技", "SoC芯片"),
    ("300623.SZ", "捷捷微电", "功率半导体"),
    ("600460.SH", "士兰微", "功率IDM"),
    # ── Telecom Equipment ──
    ("002396.SZ", "星网锐捷", "网络设备"),
    ("600498.SH", "烽火通信", "光通信"),
    ("300628.SZ", "亿联网络", "通信终端"),
    # ── Display/Optoelectronic ──
    ("000725.SZ", "京东方A", "面板"),
    ("002456.SZ", "欧菲光", "光学模组"),
    ("300408.SZ", "三环集团", "电子陶瓷"),
    # ── Defense/Tech ──
    ("002025.SZ", "航天电器", "航天连接器"),
    ("600118.SH", "中国卫星", "卫星应用"),
]

store = DuckDBStore(DB_PATH)


def main():
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    print("=" * 60)
    print("Tushare + THS AI Stock Data Ingestion")
    print(f"Target: {len(AI_STOCKS)} core AI stocks")
    print("=" * 60)

    fetch_forecasts()
    fetch_sw_pe(pro)

    print("\n" + "=" * 60)
    print_summary()


def fetch_forecasts():
    """Fetch analyst consensus EPS forecasts from THS."""
    print(f"\n[1/2] THS profit forecasts ({len(AI_STOCKS)} stocks)...")

    import akshare as ak

    all_rows = []
    success = 0

    for code, name, label in AI_STOCKS:
        symbol = code.split(".")[0]
        try:
            df = ak.stock_profit_forecast_ths(symbol=symbol)
            if df is None or df.empty:
                print(f"  SKIP {name} ({symbol}): no forecast data")
                continue

            df["ts_code"] = code
            df["symbol"] = symbol
            df["name"] = name
            df["sector_label"] = label
            all_rows.append(df)
            success += 1
            print(f"  OK   {name} ({symbol}): {len(df)} forecast years, "
                  f"latest consensus EPS={df.iloc[-1, 3]}")
        except Exception as e:
            print(f"  FAIL {name} ({symbol}): {str(e)[:60]}")

        time.sleep(CALL_DELAY)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        store.write_df("analyst_forecast", combined, mode="replace")
        print(f"  -> {success}/{len(AI_STOCKS)} stocks, {len(combined)} rows")


def fetch_sw_pe(pro):
    """Fetch SW industry PE/PB from Tushare."""
    print("\n[2/2] SW industry PE/PB...")

    sw_codes = [
        ("801081.SI", "半导体"),
        ("801080.SI", "电子"),
        ("801101.SI", "计算机"),
        ("801102.SI", "通信设备"),
        ("801760.SI", "传媒"),
    ]

    all_frames = []
    for code, name in sw_codes:
        try:
            df = pro.sw_daily(ts_code=code)
            if df is not None and not df.empty:
                df["sector_code"] = code
                df["sector_name"] = name
                all_frames.append(df)
                print(f"  OK   {name} ({code}): {len(df)} rows")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
        time.sleep(0.5)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        store.write_df("sw_industry_pe_daily", combined, mode="replace")
        print(f"  -> {len(combined)} rows")


def print_summary():
    print("-" * 40)
    for t in ["analyst_forecast", "sw_industry_pe_daily"]:
        if store.table_exists(t):
            cnt = store.query(f"SELECT count(*) as n FROM {t}")["n"].iloc[0]
            print(f"  {t}: {cnt} rows")


if __name__ == "__main__":
    main()
