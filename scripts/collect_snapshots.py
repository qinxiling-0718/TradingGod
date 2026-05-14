"""Weekly forecast snapshot collector.

Pulls THS analyst consensus EPS for ~100 AI supply chain stocks.
Each run creates a new snapshot — after 4 weeks, the revision time
series becomes usable for expectation gap factors.

Run weekly:  uv run python scripts/collect_snapshots.py
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store.duckdb_store import DuckDBStore

DB_PATH = "data/trading_god.duckdb"
CALL_DELAY = 1.5

# ── ~100 AI Supply Chain Stocks ────────────────────────────────────
# Layer 1 (Materials): 有色金属, 化工
# Layer 2 (Core): 半导体设备, 芯片设计/制造/封测
# Layer 3 (Infrastructure): 光模块, PCB, 服务器, 数据中心, 通信
# Layer 4 (Applications): AI软件, 自动驾驶, 军工, 新能源

AI_STOCKS = [
    # ── Layer 1: Materials (~8) ──
    ("000807.SZ", "云铝股份"),      ("600362.SH", "江西铜业"),
    ("600111.SH", "北方稀土"),      ("600096.SH", "云天化"),
    ("600309.SH", "万华化学"),      ("002601.SZ", "龙佰集团"),
    ("002407.SZ", "多氟多"),        ("002709.SZ", "天赐材料"),

    # ── Layer 2a: Semiconductor Equipment (~10) ──
    ("002371.SZ", "北方华创"),      ("688012.SH", "中微公司"),
    ("688082.SH", "盛美上海"),      ("688200.SH", "华峰测控"),
    ("688037.SH", "芯源微"),        ("603690.SH", "至纯科技"),
    ("300604.SZ", "长川科技"),      ("688120.SH", "华海清科"),
    ("002151.SZ", "北斗星通"),      ("300293.SZ", "蓝英装备"),

    # ── Layer 2b: Chip Design (~15) ──
    ("688256.SH", "寒武纪"),        ("603986.SH", "兆易创新"),
    ("002049.SZ", "紫光国微"),      ("603501.SH", "韦尔股份"),
    ("688008.SH", "澜起科技"),      ("688099.SH", "晶晨股份"),
    ("300661.SZ", "圣邦股份"),      ("300782.SZ", "卓胜微"),
    ("603160.SH", "汇顶科技"),      ("300458.SZ", "全志科技"),
    ("300623.SZ", "捷捷微电"),      ("600460.SH", "士兰微"),
    ("688521.SH", "芯原股份"),      ("300672.SZ", "国科微"),
    ("688396.SH", "华润微"),

    # ── Layer 2c: Foundry/Packaging (~8) ──
    ("688981.SH", "中芯国际"),      ("600584.SH", "长电科技"),
    ("002156.SZ", "通富微电"),      ("603005.SH", "晶方科技"),
    ("600703.SH", "三安光电"),      ("002185.SZ", "华天科技"),
    ("688538.SH", "和辉光电"),      ("688126.SH", "沪硅产业"),

    # ── Layer 3a: Optical Modules (~8) ──
    ("300502.SZ", "新易盛"),        ("300308.SZ", "中际旭创"),
    ("300394.SZ", "天孚通信"),      ("002281.SZ", "光迅科技"),
    ("688313.SH", "仕佳光子"),      ("300548.SZ", "博创科技"),
    ("300570.SZ", "太辰光"),        ("688205.SH", "德科立"),

    # ── Layer 3b: PCB/Assembly (~8) ──
    ("002463.SZ", "沪电股份"),      ("002916.SZ", "深南电路"),
    ("300476.SZ", "胜宏科技"),      ("002384.SZ", "东山精密"),
    ("002938.SZ", "鹏鼎控股"),      ("603228.SH", "景旺电子"),
    ("300124.SZ", "汇川技术"),      ("002402.SZ", "和而泰"),

    # ── Layer 3c: AI Server/Computing (~8) ──
    ("000977.SZ", "浪潮信息"),      ("603019.SH", "中科曙光"),
    ("002415.SZ", "海康威视"),      ("002236.SZ", "大华股份"),
    ("300474.SZ", "景嘉微"),        ("688568.SH", "中科星图"),
    ("688561.SH", "奇安信"),        ("002439.SZ", "启明星辰"),

    # ── Layer 3d: Data Center/Power (~8) ──
    ("002335.SZ", "科华数据"),      ("300383.SZ", "光环新网"),
    ("600845.SH", "宝信软件"),      ("300383.SZ", "光环新网"),
    ("300738.SZ", "奥飞数据"),      ("600850.SH", "华东电脑"),
    ("002837.SZ", "英维克"),        ("300499.SZ", "高澜股份"),

    # ── Layer 3e: Telecom Equipment (~8) ──
    ("002396.SZ", "星网锐捷"),      ("600498.SH", "烽火通信"),
    ("300628.SZ", "亿联网络"),      ("002491.SZ", "通鼎互联"),
    ("300565.SZ", "科信技术"),      ("300353.SZ", "东土科技"),
    ("002583.SZ", "海能达"),        ("300050.SZ", "世纪鼎利"),

    # ── Layer 4a: AI Software/Platform (~10) ──
    ("002230.SZ", "科大讯飞"),      ("688111.SH", "金山办公"),
    ("300033.SZ", "同花顺"),        ("002410.SZ", "广联达"),
    ("600570.SH", "恒生电子"),      ("688088.SH", "虹软科技"),
    ("688369.SH", "致远互联"),      ("300454.SZ", "深信服"),
    ("688023.SH", "安恒信息"),      ("300624.SZ", "万兴科技"),

    # ── Layer 4b: Autonomous Driving (~8) ──
    ("002920.SZ", "德赛西威"),      ("601689.SH", "拓普集团"),
    ("600699.SH", "均胜电子"),      ("002906.SZ", "华阳集团"),
    ("002472.SZ", "双环传动"),      ("300552.SZ", "万集科技"),
    ("688326.SH", "经纬恒润"),      ("300496.SZ", "中科创达"),

    # ── Layer 4c: New Energy/Battery (~8) ──
    ("300750.SZ", "宁德时代"),      ("002594.SZ", "比亚迪"),
    ("300014.SZ", "亿纬锂能"),      ("002460.SZ", "赣锋锂业"),
    ("300124.SZ", "汇川技术"),      ("600438.SH", "通威股份"),
    ("601012.SH", "隆基绿能"),      ("002129.SZ", "TCL中环"),

    # ── Layer 4d: Defense/Tech (~8) ──
    ("002025.SZ", "航天电器"),      ("600118.SH", "中国卫星"),
    ("600760.SH", "中航沈飞"),      ("000768.SZ", "中航西飞"),
    ("600893.SH", "航发动力"),      ("002013.SZ", "中航机电"),
    ("600372.SH", "中航电子"),      ("600879.SH", "航天电子"),

    # ── Displays/Optoelectronic/Other (~5) ──
    ("000725.SZ", "京东方A"),       ("002456.SZ", "欧菲光"),
    ("300408.SZ", "三环集团"),      ("002241.SZ", "歌尔股份"),
    ("300207.SZ", "欣旺达"),
]

store = DuckDBStore(DB_PATH)


def main():
    snapshot_date = date.today()

    print("=" * 60)
    print(f"Forecast Snapshot Collector — {snapshot_date}")
    print(f"Target: {len(AI_STOCKS)} AI supply chain stocks")
    print("=" * 60)

    collect_snapshot(snapshot_date)

    print()
    print_summary(snapshot_date)


def collect_snapshot(snapshot_date: date):
    import akshare as ak

    all_rows = []
    success = 0
    skipped = 0

    for i, (code, name) in enumerate(AI_STOCKS):
        symbol = code.split(".")[0]

        if i > 0 and (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(AI_STOCKS)} done, waiting 5s ...")
            time.sleep(5)  # extra cooldown every 20 stocks

        try:
            df = ak.stock_profit_forecast_ths(symbol=symbol)
            if df is None or df.empty:
                skipped += 1
                continue

            df["snapshot_date"] = snapshot_date
            df["ts_code"] = code
            df["symbol"] = symbol
            df["name"] = name
            all_rows.append(df)
            success += 1

        except Exception as e:
            err = str(e)[:60]
            if i < 3:
                print(f"  FAIL {name} ({symbol}): {err}")

        time.sleep(CALL_DELAY)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        store.write_df("forecast_snapshots", combined, mode="append")
        print(f"\n  OK {success} stocks collected, {skipped} no data")
        print(f"  → Appended {len(combined)} rows to forecast_snapshots")

        # Track metadata
        meta = pd.DataFrame([{
            "snapshot_date": snapshot_date,
            "stocks_success": success,
            "stocks_skipped": skipped,
            "total_rows": len(combined),
            "collected_at": datetime.now().isoformat(),
        }])
        store.write_df("snapshot_metadata", meta, mode="append")


def print_summary(snapshot_date: date):
    if not store.table_exists("forecast_snapshots"):
        print("No snapshots yet.")
        return

    df = store.read_df("forecast_snapshots")
    snaps = df["snapshot_date"].nunique()
    stocks = df["ts_code"].nunique()
    print(f"Total: {snaps} snapshots, {stocks} unique stocks, {len(df)} rows")

    if snaps >= 4:
        print("OK Revision time series ready (4+ snapshots)")
    elif snaps > 0:
        print(f"[{snaps}/4] Accumulating... {4-snaps} more weeks until revision factors activate")


if __name__ == "__main__":
    main()
