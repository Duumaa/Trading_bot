"""
download_jpy_crosses.py
=======================
Download GBPJPY, AUDJPY, NZDJPY historical M15 + D1 data
from cTrader OpenAPI and save as CSV files.

Usage (PowerShell):
    python download_jpy_crosses.py

Output files (same folder as this script):
    GBPJPY_M15.csv
    AUDJPY_M15.csv
    NZDJPY_M15.csv
    GBPJPY_D1_19Y.csv
    AUDJPY_D1_19Y.csv
    NZDJPY_D1_19Y.csv
"""

import os
from twisted.internet import reactor, task, defer
from datetime import datetime, timezone, timedelta
import pandas as pd
from ctrader_open_api import Client, Protobuf, TcpProtocol
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# ═══════════════════════════════════════════════════════════════
# CREDENTIALS — ӨӨРИЙН МЭДЭЭЛЛИЙГ ЭНД ОРУУЛНА УУ
# ═══════════════════════════════════════════════════════════════
APP_CLIENT_ID     = ""        # ← солино
APP_CLIENT_SECRET = ""    # ← солино
ACCESS_TOKEN      = ""     # ← солино
ACCOUNT_ID        =                # ← солино (int)
# ═══════════════════════════════════════════════════════════════

HOST = EndPoints.PROTOBUF_LIVE_HOST
PORT = EndPoints.PROTOBUF_PORT

# JPY cross pairs — энэ 3 хос татна
MISSING_PAIRS = ["GBPJPY", "AUDJPY", "NZDJPY"]

# JPY pairs: cTrader /100000 биш /1000 хувааж тооцно
JPY_PAIRS = {"GBPJPY", "AUDJPY", "NZDJPY", "USDJPY", "EURJPY"}

# Хадгалах folder — энэ скриптийн байгаа хавтас
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_bar(bar, symbol_name: str) -> dict:
    """
    cTrader bar-ийг OHLCV dict болгон хувиргана.
    JPY pair-д /1000, бусад pair-д /100000 хувааж тооцно.
    """
    divisor = 1000.0 if symbol_name in JPY_PAIRS else 100000.0
    return {
        "timestamp": pd.to_datetime(bar.utcTimestampInMinutes * 60, unit="s", utc=True),
        "open":  (bar.low + bar.deltaOpen)  / divisor,
        "high":  (bar.low + bar.deltaHigh)  / divisor,
        "low":   bar.low                    / divisor,
        "close": (bar.low + bar.deltaClose) / divisor,
        "volume": bar.volume,
    }


def get_period_minutes(period) -> int:
    """Period enum-ийг минутаар буцаана."""
    mapping = {
        ProtoOATrendbarPeriod.M1:  1,
        ProtoOATrendbarPeriod.M2:  2,
        ProtoOATrendbarPeriod.M3:  3,
        ProtoOATrendbarPeriod.M4:  4,
        ProtoOATrendbarPeriod.M5:  5,
        ProtoOATrendbarPeriod.M10: 10,
        ProtoOATrendbarPeriod.M15: 15,
        ProtoOATrendbarPeriod.M30: 30,
        ProtoOATrendbarPeriod.H1:  60,
        ProtoOATrendbarPeriod.H4:  240,
        ProtoOATrendbarPeriod.H12: 720,
        ProtoOATrendbarPeriod.D1:  1440,
        ProtoOATrendbarPeriod.W1:  10080,
    }
    return mapping.get(period, 15)


def get_chunk_ms(period) -> int:
    """Нэг request-д хэдэн ms-ийн дата авах вэ."""
    if period >= ProtoOATrendbarPeriod.D1:
        return 5 * 365 * 24 * 60 * 60 * 1000       # 5 жил
    elif period >= ProtoOATrendbarPeriod.H1:
        return 300 * 24 * 60 * 60 * 1000            # 300 хоног
    elif period >= ProtoOATrendbarPeriod.M15:
        return 60 * 24 * 60 * 60 * 1000             # 60 хоног
    else:
        return 5 * 24 * 60 * 60 * 1000              # 5 хоног


async def download_pair(client, symbol_id: int, symbol_name: str,
                         period, years: int, label: str) -> pd.DataFrame:
    """
    Нэг хос, нэг timeframe-ийн бүх дата татаж DataFrame буцаана.

    Args:
        client:      cTrader Client instance
        symbol_id:   cTrader symbol ID
        symbol_name: e.g. "GBPJPY"
        period:      ProtoOATrendbarPeriod enum
        years:       хэдэн жилийн дата татах
        label:       log дээр харагдах нэр e.g. "GBPJPY M15"

    Returns:
        Sorted, deduplicated DataFrame
    """
    end_time   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp() * 1000)

    chunk_ms       = get_chunk_ms(period)
    period_minutes = get_period_minutes(period)
    current_from   = start_time
    bars_data      = []

    print(f"\n🚀 {label} татаж эхэллээ ({years} жил)...")

    while current_from < end_time:
        current_to = min(current_from + chunk_ms, end_time)

        req = ProtoOAGetTrendbarsReq(
            ctidTraderAccountId=ACCOUNT_ID,
            fromTimestamp=current_from,
            toTimestamp=current_to,
            period=period,
            symbolId=symbol_id,
        )

        from_dt = pd.to_datetime(current_from, unit="ms", utc=True).strftime("%Y-%m-%d")
        to_dt   = pd.to_datetime(current_to,   unit="ms", utc=True).strftime("%Y-%m-%d")
        print(f"  Requesting {from_dt} → {to_dt}...", end=" ")

        res_msg = await client.send(req)
        res     = Protobuf.extract(res_msg)

        if type(res).__name__ == "ProtoOAErrorRes":
            print(f"\n❌ API алдаа: {res.errorCode} - {res.description}")
            break

        if not res.trendbar:
            print("өгөгдөл байхгүй, алгасна.")
            current_from = current_to
            continue

        print(f"{len(res.trendbar)} bar авлаа.")

        for bar in res.trendbar:
            bars_data.append(parse_bar(bar, symbol_name))

        max_ts   = max(bar.utcTimestampInMinutes for bar in res.trendbar)
        new_from = (max_ts + period_minutes) * 60000

        if new_from <= current_from:
            new_from = current_from + (period_minutes * 60000)

        current_from = new_from

    if not bars_data:
        print(f"⚠️  {label}: өгөгдөл татагдсангүй.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(bars_data)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    print(f"✅ {label}: нийт {len(df):,} bar, "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    return df


async def main(reactor):
    """
    Үндсэн функц:
    1. cTrader-д нэвтэрнэ
    2. Symbol ID-уудыг олно
    3. M15 (15 жил) + D1 (19 жил) татна
    4. CSV хадгална
    """
    client = Client(HOST, PORT, TcpProtocol)
    client.startService()

    try:
        await client.whenConnected()
        print("✅ cTrader-д холбогдлоо.")

        # 1. Application auth
        app_auth = ProtoOAApplicationAuthReq(
            clientId=APP_CLIENT_ID,
            clientSecret=APP_CLIENT_SECRET,
        )
        await client.send(app_auth)
        print("✅ Application auth амжилттай.")

        # 2. Account auth
        acc_auth = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken=ACCESS_TOKEN,
        )
        await client.send(acc_auth)
        print("✅ Account auth амжилттай.")

        # 3. Symbol бүгдийг татаж ID олох
        symbols_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        symbols_msg = await client.send(symbols_req)
        symbols_res = Protobuf.extract(symbols_msg)

        if type(symbols_res).__name__ == "ProtoOAErrorRes":
            print(f"❌ Symbol жагсаалт авах алдаа: {symbols_res.errorCode}")
            return

        # Symbol нэр → ID map үүсгэнэ
        symbol_map = {s.symbolName: s.symbolId for s in symbols_res.symbol}

        # Байгаа эсэхийг шалгана
        for pair in MISSING_PAIRS:
            if pair not in symbol_map:
                print(f"⚠️  {pair} энэ account-д байхгүй байна!")
        print(f"✅ Symbol map: {', '.join(f'{p}={symbol_map.get(p)}' for p in MISSING_PAIRS)}")

        # 4. M15 + D1 татаж хадгална
        tasks = [
            # (pair, period, years, filename_suffix)
            ("GBPJPY", ProtoOATrendbarPeriod.M15, 15, "GBPJPY_M15.csv"),
            ("AUDJPY", ProtoOATrendbarPeriod.M15, 15, "AUDJPY_M15.csv"),
            ("NZDJPY", ProtoOATrendbarPeriod.M15, 15, "NZDJPY_M15.csv"),
            ("GBPJPY", ProtoOATrendbarPeriod.D1,  19, "GBPJPY_D1_19Y.csv"),
            ("AUDJPY", ProtoOATrendbarPeriod.D1,  19, "AUDJPY_D1_19Y.csv"),
            ("NZDJPY", ProtoOATrendbarPeriod.D1,  19, "NZDJPY_D1_19Y.csv"),
        ]

        for pair, period, years, filename in tasks:
            if pair not in symbol_map:
                print(f"⚠️  {pair} алгасав (symbol ID байхгүй).")
                continue

            period_name = ProtoOATrendbarPeriod.Name(period)
            label       = f"{pair} {period_name}"

            df = await download_pair(
                client=client,
                symbol_id=symbol_map[pair],
                symbol_name=pair,
                period=period,
                years=years,
                label=label,
            )

            if df.empty:
                continue

            # CSV хадгална
            out_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(out_path, index=False)
            print(f"💾 Хадгалагдлаа: {out_path}")

        print("\n🎉 Бүх таталт дууслаа!")
        print(f"📁 Файлууд: {OUTPUT_DIR}")
        print("\nДараагийн алхам:")
        print("  1. Эдгээр 6 CSV файлыг Codespace-д upload хийнэ")
        print("  2. /workspaces/Trading_bot/data/kaggle/m15/ → M15 файлууд")
        print("  3. /workspaces/Trading_bot/data/kaggle/d1/  → D1  файлууд")

    finally:
        client.stopService()


if __name__ == "__main__":
    task.react(lambda r: defer.ensureDeferred(main(r)))