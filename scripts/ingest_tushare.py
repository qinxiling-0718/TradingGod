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
    # ── Layer 2a: Semiconductor Equipment (~15) ──
    ("002371.SZ", "北方华创", "半导体设备"),    ("688012.SH", "中微公司", "半导体设备"),
    ("688082.SH", "盛美上海", "半导体设备"),    ("688200.SH", "华峰测控", "半导体设备"),
    ("688037.SH", "芯源微", "半导体设备"),      ("603690.SH", "至纯科技", "半导体设备"),
    ("300604.SZ", "长川科技", "半导体设备"),    ("688120.SH", "华海清科", "半导体设备"),
    ("688072.SH", "拓荆科技", "半导体设备"),    ("688147.SH", "微导纳米", "半导体设备"),
    ("688361.SH", "中科飞测", "半导体设备"),    ("688652.SH", "京仪装备", "半导体设备"),
    ("300567.SZ", "精测电子", "半导体设备"),    ("688409.SH", "富创精密", "半导体设备"),
    ("300293.SZ", "蓝英装备", "半导体设备"),
    # ── Layer 2b: Chip Design/Foundry (~25) ──
    ("688981.SH", "中芯国际", "晶圆代工"),      ("688256.SH", "寒武纪", "AI芯片"),
    ("603986.SH", "兆易创新", "存储芯片"),      ("002049.SZ", "紫光国微", "FPGA芯片"),
    ("688008.SH", "澜起科技", "内存接口"),      ("603501.SH", "韦尔股份", "CIS芯片"),
    ("688099.SH", "晶晨股份", "SoC芯片"),       ("300661.SZ", "圣邦股份", "模拟芯片"),
    ("300782.SZ", "卓胜微", "射频芯片"),        ("603160.SH", "汇顶科技", "指纹芯片"),
    ("300458.SZ", "全志科技", "SoC芯片"),       ("300623.SZ", "捷捷微电", "功率半导体"),
    ("600460.SH", "士兰微", "功率IDM"),         ("688521.SH", "芯原股份", "芯片IP"),
    ("300672.SZ", "国科微", "存储控制"),        ("688396.SH", "华润微", "功率代工"),
    ("688041.SH", "海光信息", "CPU/DCU"),       ("688536.SH", "思瑞浦", "模拟芯片"),
    ("688234.SH", "天岳先进", "碳化硅"),        ("688270.SH", "臻镭科技", "射频芯片"),
    ("688048.SH", "长光华芯", "激光芯片"),      ("300475.SZ", "香农芯创", "芯片分销"),
    ("688368.SH", "晶丰明源", "电源管理"),      ("688261.SH", "东微半导", "功率器件"),
    ("688332.SH", "中科蓝讯", "蓝牙芯片"),
    # ── Layer 2c: Foundry/Packaging (~8) ──
    ("600584.SH", "长电科技", "封测"),          ("002156.SZ", "通富微电", "封测"),
    ("603005.SH", "晶方科技", "封测"),          ("002185.SZ", "华天科技", "封测"),
    ("600703.SH", "三安光电", "化合物半导体"),  ("688538.SH", "和辉光电", "面板"),
    ("688126.SH", "沪硅产业", "硅片"),          ("688220.SH", "翱捷科技", "基带芯片"),
    # ── Layer 3a: Optical Modules/Devices (~10) ──
    ("300502.SZ", "新易盛", "光模块"),          ("300308.SZ", "中际旭创", "光模块"),
    ("300394.SZ", "天孚通信", "光器件"),        ("002281.SZ", "光迅科技", "光器件"),
    ("688313.SH", "仕佳光子", "光芯片"),        ("300548.SZ", "博创科技", "光器件"),
    ("300570.SZ", "太辰光", "光器件"),          ("688205.SH", "德科立", "光模块"),
    ("688662.SH", "富信科技", "光器件"),        ("300620.SZ", "光库科技", "光器件"),
    # ── Layer 3b: PCB/Assembly (~10) ──
    ("002463.SZ", "沪电股份", "PCB"),           ("002916.SZ", "深南电路", "PCB"),
    ("300476.SZ", "胜宏科技", "PCB"),           ("002384.SZ", "东山精密", "PCB"),
    ("002938.SZ", "鹏鼎控股", "PCB"),           ("603228.SH", "景旺电子", "PCB"),
    ("603186.SH", "华正新材", "CCL"),           ("002636.SZ", "金安国纪", "CCL"),
    ("300657.SZ", "弘信电子", "FPC"),           ("002579.SZ", "中京电子", "PCB"),
    # ── Layer 3c: AI Server/Computing (~10) ──
    ("000977.SZ", "浪潮信息", "AI服务器"),      ("603019.SH", "中科曙光", "AI服务器"),
    ("002415.SZ", "海康威视", "AI视觉"),        ("002236.SZ", "大华股份", "AI视觉"),
    ("300474.SZ", "景嘉微", "GPU"),             ("688568.SH", "中科星图", "数字地球"),
    ("688561.SH", "奇安信", "网安"),            ("002439.SZ", "启明星辰", "网安"),
    ("688031.SH", "星环科技", "大数据"),        ("688316.SH", "青云科技", "云计算"),
    # ── Layer 3d: Data Center/Cooling/Power (~10) ──
    ("002335.SZ", "科华数据", "数据中心"),      ("300383.SZ", "光环新网", "数据中心"),
    ("600845.SH", "宝信软件", "工业互联网"),    ("300738.SZ", "奥飞数据", "数据中心"),
    ("002837.SZ", "英维克", "液冷"),            ("300499.SZ", "高澜股份", "液冷"),
    ("300442.SZ", "润泽科技", "AIDC"),          ("301165.SZ", "锐捷网络", "网络设备"),
    ("300684.SZ", "中石科技", "散热材料"),      ("002518.SZ", "科士达", "UPS电源"),
    # ── Layer 3e: Telecom Equipment (~8) ──
    ("002396.SZ", "星网锐捷", "网络设备"),      ("600498.SH", "烽火通信", "光通信"),
    ("300628.SZ", "亿联网络", "通信终端"),      ("300565.SZ", "科信技术", "通信设备"),
    ("002583.SZ", "海能达", "专网通信"),        ("300353.SZ", "东土科技", "工业通信"),
    ("002491.SZ", "通鼎互联", "光纤光缆"),      ("300050.SZ", "世纪鼎利", "通信测试"),
    # ── Layer 4a: AI Software/Platform (~10) ──
    ("002230.SZ", "科大讯飞", "AI平台"),        ("688111.SH", "金山办公", "AI应用"),
    ("300033.SZ", "同花顺", "AI金融"),          ("002410.SZ", "广联达", "数字建筑"),
    ("600570.SH", "恒生电子", "金融IT"),        ("688088.SH", "虹软科技", "AI视觉"),
    ("300454.SZ", "深信服", "网安"),            ("300624.SZ", "万兴科技", "AI创意"),
    ("688023.SH", "安恒信息", "网安"),          ("688369.SH", "致远互联", "协同办公"),
    # ── Layer 4b: Auto/Autonomous Driving (~8) ──
    ("002920.SZ", "德赛西威", "自动驾驶"),      ("601689.SH", "拓普集团", "汽车电子"),
    ("600699.SH", "均胜电子", "汽车电子"),      ("002906.SZ", "华阳集团", "智能座舱"),
    ("002472.SZ", "双环传动", "减速器"),        ("300552.SZ", "万集科技", "激光雷达"),
    ("688326.SH", "经纬恒润", "汽车电子"),      ("300496.SZ", "中科创达", "智能OS"),
    # ── Layer 4c: New Energy/Battery (~8) ──
    ("300750.SZ", "宁德时代", "动力电池"),      ("002594.SZ", "比亚迪", "新能源"),
    ("300014.SZ", "亿纬锂能", "锂电池"),        ("002460.SZ", "赣锋锂业", "锂资源"),
    ("300124.SZ", "汇川技术", "电控"),          ("600438.SH", "通威股份", "光伏"),
    ("601012.SH", "隆基绿能", "光伏"),          ("002129.SZ", "TCL中环", "硅片"),
    # ── Layer 4d: Defense/Aerospace (~6) ──
    ("002025.SZ", "航天电器", "航天连接器"),    ("600118.SH", "中国卫星", "卫星应用"),
    ("600760.SH", "中航沈飞", "军工"),          ("000768.SZ", "中航西飞", "军工"),
    ("600893.SH", "航发动力", "发动机"),        ("600879.SH", "航天电子", "航天电子"),
    # ── Layer 4e: Display/Components (~5) ──
    ("000725.SZ", "京东方A", "面板"),           ("002456.SZ", "欧菲光", "光学模组"),
    ("300408.SZ", "三环集团", "电子陶瓷"),      ("002241.SZ", "歌尔股份", "声学"),
    ("300207.SZ", "欣旺达", "消费电池"),
    # ── AI Hardware/Industrial (~10) ──
    ("601138.SH", "工业富联", "AI服务器代工"),  ("603629.SH", "利通电子", "结构件"),
    ("002851.SZ", "麦格米特", "电源"),          ("688800.SH", "瑞可达", "连接器"),
    ("300602.SZ", "飞荣达", "散热"),            ("688772.SH", "珠海冠宇", "电池"),
    ("688779.SH", "长远锂科", "正极材料"),      ("002340.SZ", "格林美", "回收"),
    ("688005.SH", "容百科技", "正极材料"),      ("688819.SH", "天能股份", "储能"),
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
