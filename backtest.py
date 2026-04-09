"""
backtest.py
===========
Trade Performance Analytics — G10 Bot

Queries all closed trades from the DB and produces a structured
performance report covering:

  1. Overall summary
  2. Per-pair win rate and P&L
  3. Frequency cap impact (pre/post 2026-03-27 for AUDJPY/NZDJPY)
  4. Per-regime performance
  5. Session performance (ASIA / LONDON / NY)
  6. Signal path breakdown (GATE_1234 vs TECH_CONFLUENCE)
  7. R:R distribution (how far trades travel before close)
  8. Day-of-week performance

Usage
-----
  python3 /workspaces/Trading_bot/backtest.py
  python3 /workspaces/Trading_bot/backtest.py --since 2026-03-27
  python3 /workspaces/Trading_bot/backtest.py --pair AUDJPY
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from sqlalchemy import text
from loguru import logger

from data_pipeline.storage.timeseries_db import db

# ── constants ─────────────────────────────────────────────────────────────────

FREQ_CAP_DATE    = "2026-03-27"   # frequency caps live from this date
CLOSED_STATUSES  = ("CLOSED_SL", "CLOSED_TP", "CLOSED_MANUAL", "CLOSED_TECH_FLIP")
WIN_STATUSES     = ("CLOSED_TP", "CLOSED_MANUAL")   # CLOSED_MANUAL ≈ winner (positive PnL)
SESSION_BOUNDS   = {
    "ASIA":   (0,  8),    # 00:00–07:59 UTC
    "LONDON": (8,  16),   # 08:00–15:59 UTC
    "NY":     (16, 24),   # 16:00–23:59 UTC
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _session(hour: int) -> str:
    for name, (start, end) in SESSION_BOUNDS.items():
        if start <= hour < end:
            return name
    return "UNKNOWN"


def _win(row: pd.Series) -> bool:
    """True if trade was a winner (positive PnL)."""
    if row["status"] == "CLOSED_TP":
        return True
    if row["status"] in ("CLOSED_MANUAL", "CLOSED_TECH_FLIP"):
        return (row["pnl_pips"] or 0) > 0
    return False


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{n/total*100:.0f}%"


def _fmt(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}"

# ── data loading ──────────────────────────────────────────────────────────────

def load_trades(since: str | None = None, pair: str | None = None) -> pd.DataFrame:
    """Load all closed trades from DB into a DataFrame."""
    filters = ["status = ANY(:statuses)"]
    params: dict = {"statuses": list(CLOSED_STATUSES)}

    if since:
        filters.append("entry_time >= :since")
        params["since"] = since

    if pair:
        filters.append("symbol = :pair")
        params["pair"] = pair.upper()

    where = " AND ".join(filters)
    query = f"""
        SELECT
            id, symbol, direction, status, regime,
            entry_price, exit_price, stop_loss, take_profit,
            pnl_pips, pnl_usd, entry_time, exit_time,
            signal_1_regime, signal_1_confidence,
            signal_2_fund_score, signal_4_tsmixer,
            signal_5_sentiment,
            COALESCE(signal_path, 'GATE_1234') AS signal_path,
            trading_mode
        FROM trades
        WHERE {where}
        ORDER BY entry_time
    """
    with db.connection() as conn:
        rows = conn.execute(text(query), params).fetchall()

    cols = [
        "id", "symbol", "direction", "status", "regime",
        "entry_price", "exit_price", "stop_loss", "take_profit",
        "pnl_pips", "pnl_usd", "entry_time", "exit_time",
        "s1_regime", "s1_conf", "s2_score", "s4_tsmixer",
        "s5_sent", "signal_path", "trading_mode",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Derived columns
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["hour"]       = df["entry_time"].dt.hour
    df["session"]    = df["hour"].apply(_session)
    df["weekday"]    = df["entry_time"].dt.day_name()
    df["date"]       = df["entry_time"].dt.date
    df["is_win"]     = df.apply(_win, axis=1)
    df["pnl_pips"]   = pd.to_numeric(df["pnl_pips"], errors="coerce")
    df["pnl_usd"]    = pd.to_numeric(df["pnl_usd"],  errors="coerce")

    return df

# ── report sections ───────────────────────────────────────────────────────────

def _bar(val: float, max_val: float, width: int = 20) -> str:
    filled = int(round(val / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def section_overall(df: pd.DataFrame) -> None:
    n        = len(df)
    wins     = df["is_win"].sum()
    wr       = wins / n * 100 if n else 0
    total_pnl = df["pnl_usd"].sum()
    avg_win  = df.loc[df["is_win"],  "pnl_pips"].mean()
    avg_loss = df.loc[~df["is_win"], "pnl_pips"].mean()
    expectancy = (wr/100 * avg_win + (1-wr/100) * avg_loss) if n else 0

    print("\n" + "═"*60)
    print("  OVERALL PERFORMANCE SUMMARY")
    print("═"*60)
    print(f"  Total trades  : {n}")
    print(f"  Win rate      : {wr:.1f}%  ({wins}W / {n-wins}L)")
    print(f"  Total P&L     : ${total_pnl:+.2f}")
    print(f"  Avg win pips  : {avg_win:+.1f}")
    print(f"  Avg loss pips : {avg_loss:+.1f}")
    print(f"  Expectancy    : {expectancy:+.2f} pips/trade")
    print(f"  Date range    : {df['date'].min()} → {df['date'].max()}")
    print()

    # Status breakdown
    print("  Status breakdown:")
    for s in CLOSED_STATUSES:
        sub = df[df["status"] == s]
        pnl = sub["pnl_usd"].sum()
        print(f"    {s:<22} {len(sub):>4} trades   ${pnl:+8.2f}")


def section_per_pair(df: pd.DataFrame) -> None:
    print("\n" + "═"*60)
    print("  PER-PAIR PERFORMANCE")
    print("═"*60)
    print(f"  {'Pair':<10} {'N':>4}  {'WR':>6}  {'PnL $':>8}  {'Avg pip':>8}  {'Verdict'}")
    print("  " + "-"*56)

    by_pair = df.groupby("symbol")
    rows = []
    for sym, g in by_pair:
        n   = len(g)
        wr  = g["is_win"].mean() * 100
        pnl = g["pnl_usd"].sum()
        avg = g["pnl_pips"].mean()
        rows.append((sym, n, wr, pnl, avg))

    rows.sort(key=lambda x: x[3], reverse=True)  # sort by total PnL

    for sym, n, wr, pnl, avg in rows:
        if pnl > 5:
            verdict = "✅ Profitable"
        elif wr < 30 and n >= 5:
            verdict = "🔴 Losing"
        elif n < 3:
            verdict = "⚪ Low data"
        else:
            verdict = "⚠️  Marginal"
        print(f"  {sym:<10} {n:>4}  {wr:>5.0f}%  {pnl:>+8.2f}  {avg:>+8.1f}  {verdict}")


def section_freq_cap(df: pd.DataFrame) -> None:
    """AUDJPY/NZDJPY before vs after frequency cap (2026-03-27)."""
    print("\n" + "═"*60)
    print(f"  FREQUENCY CAP IMPACT (pre/post {FREQ_CAP_DATE})")
    print("═"*60)

    pairs = ["AUDJPY", "NZDJPY"]
    cap_date = pd.Timestamp(FREQ_CAP_DATE, tz="UTC")

    for sym in pairs:
        g = df[df["symbol"] == sym]
        pre  = g[g["entry_time"] <  cap_date]
        post = g[g["entry_time"] >= cap_date]

        if len(pre) == 0 and len(post) == 0:
            print(f"  {sym}: no data")
            continue

        # Trades per day
        def _per_day(sub: pd.DataFrame) -> float:
            if len(sub) == 0:
                return 0.0
            days = max((sub["date"].max() - sub["date"].min()).days + 1, 1)
            return len(sub) / days

        pre_n   = len(pre)
        post_n  = len(post)
        pre_wr  = pre["is_win"].mean() * 100  if pre_n  else 0
        post_wr = post["is_win"].mean() * 100 if post_n else 0
        pre_pnl  = pre["pnl_usd"].sum()
        post_pnl = post["pnl_usd"].sum()
        pre_day  = _per_day(pre)
        post_day = _per_day(post)

        print(f"\n  {sym}:")
        print(f"    {'':25}  {'PRE-CAP':>10}  {'POST-CAP':>10}  {'CHANGE':>10}")
        print(f"    {'Trades':25}  {pre_n:>10}  {post_n:>10}")
        print(f"    {'Trades/day':25}  {pre_day:>10.1f}  {post_day:>10.1f}  ", end="")
        delta = post_day - pre_day
        print(f"{delta:>+9.1f}")
        print(f"    {'Win rate':25}  {pre_wr:>9.0f}%  {post_wr:>9.0f}%  ", end="")
        print(f"{post_wr-pre_wr:>+9.0f}%")
        print(f"    {'Total P&L $':25}  {pre_pnl:>+10.2f}  {post_pnl:>+10.2f}  ", end="")
        print(f"{post_pnl-pre_pnl:>+10.2f}")


def section_regime(df: pd.DataFrame) -> None:
    print("\n" + "═"*60)
    print("  PER-REGIME PERFORMANCE")
    print("═"*60)
    print(f"  {'Regime':<8} {'N':>4}  {'WR':>6}  {'PnL $':>8}  {'Avg pip':>8}")
    print("  " + "-"*42)

    for regime in ["THV", "TLV", "MR", "PNS", "UNKNOWN"]:
        g = df[df["regime"] == regime] if regime != "UNKNOWN" else df[~df["regime"].isin(["THV","TLV","MR","PNS"])]
        if len(g) == 0:
            continue
        wr  = g["is_win"].mean() * 100
        pnl = g["pnl_usd"].sum()
        avg = g["pnl_pips"].mean()
        print(f"  {regime:<8} {len(g):>4}  {wr:>5.0f}%  {pnl:>+8.2f}  {avg:>+8.1f}")


def section_session(df: pd.DataFrame) -> None:
    print("\n" + "═"*60)
    print("  SESSION PERFORMANCE  (ASIA=00-08h, LONDON=08-16h, NY=16-24h UTC)")
    print("═"*60)
    print(f"  {'Session':<10} {'N':>4}  {'WR':>6}  {'PnL $':>8}  {'Avg pip':>8}")
    print("  " + "-"*46)

    for sess in ["ASIA", "LONDON", "NY"]:
        g = df[df["session"] == sess]
        if len(g) == 0:
            continue
        wr  = g["is_win"].mean() * 100
        pnl = g["pnl_usd"].sum()
        avg = g["pnl_pips"].mean()
        print(f"  {sess:<10} {len(g):>4}  {wr:>5.0f}%  {pnl:>+8.2f}  {avg:>+8.1f}")


def section_path(df: pd.DataFrame) -> None:
    print("\n" + "═"*60)
    print("  SIGNAL PATH BREAKDOWN")
    print("═"*60)
    print(f"  {'Path':<20} {'N':>4}  {'WR':>6}  {'PnL $':>8}  {'Avg pip':>8}")
    print("  " + "-"*52)

    for path in df["signal_path"].unique():
        g = df[df["signal_path"] == path]
        wr  = g["is_win"].mean() * 100
        pnl = g["pnl_usd"].sum()
        avg = g["pnl_pips"].mean()
        note = " ← baseline" if path == "GATE_1234" else " ← new path"
        print(f"  {path:<20} {len(g):>4}  {wr:>5.0f}%  {pnl:>+8.2f}  {avg:>+8.1f}{note}")


def section_rr(df: pd.DataFrame) -> None:
    """R:R distribution — how far trades go before closing."""
    print("\n" + "═"*60)
    print("  R:R DISTRIBUTION (pnl_pips vs stop distance)")
    print("═"*60)

    # Compute R achieved for each trade
    def _r_achieved(row: pd.Series) -> float | None:
        entry = row["entry_price"]
        sl    = row["stop_loss"]
        pips  = row["pnl_pips"]
        if not entry or not sl or pips is None or pd.isna(pips):
            return None
        sl_dist_pips = abs(entry - sl) / (0.01 if "JPY" in str(row["symbol"]) else 0.0001)
        if sl_dist_pips <= 0:
            return None
        return pips / sl_dist_pips

    df = df.copy()
    df["r_achieved"] = df.apply(_r_achieved, axis=1)

    buckets = [
        ("< -1R (blowout)",  df["r_achieved"] < -1.0),
        ("-1R to 0R (SL)",   (df["r_achieved"] >= -1.0) & (df["r_achieved"] < 0)),
        ("0R to +1R",        (df["r_achieved"] >= 0)    & (df["r_achieved"] < 1.0)),
        ("+1R to +2R",       (df["r_achieved"] >= 1.0)  & (df["r_achieved"] < 2.0)),
        ("+2R to +3R",       (df["r_achieved"] >= 2.0)  & (df["r_achieved"] < 3.0)),
        ("+3R+ (full TP)",   df["r_achieved"] >= 3.0),
    ]

    total = df["r_achieved"].notna().sum()
    max_n = 0
    counts = [(label, mask.sum()) for label, mask in buckets]
    max_n  = max(c for _, c in counts) or 1

    for label, count in counts:
        bar = _bar(count, max_n, width=24)
        pct = _pct(count, total)
        print(f"  {label:<22}  {bar}  {count:>4} ({pct})")

    avg_r_win  = df.loc[df["is_win"], "r_achieved"].mean()
    avg_r_loss = df.loc[~df["is_win"],"r_achieved"].mean()
    print(f"\n  Avg R on winners  : {avg_r_win:+.2f}R")
    print(f"  Avg R on losers   : {avg_r_loss:+.2f}R")


def section_weekday(df: pd.DataFrame) -> None:
    print("\n" + "═"*60)
    print("  DAY-OF-WEEK PERFORMANCE")
    print("═"*60)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print(f"  {'Day':<12} {'N':>4}  {'WR':>6}  {'PnL $':>8}  {'Avg pip':>8}")
    print("  " + "-"*46)
    for day in order:
        g = df[df["weekday"] == day]
        if len(g) == 0:
            continue
        wr  = g["is_win"].mean() * 100
        pnl = g["pnl_usd"].sum()
        avg = g["pnl_pips"].mean()
        print(f"  {day:<12} {len(g):>4}  {wr:>5.0f}%  {pnl:>+8.2f}  {avg:>+8.1f}")


def section_daily_trades(df: pd.DataFrame) -> None:
    """Trades per day — spot overtrading days."""
    print("\n" + "═"*60)
    print("  TRADES PER DAY")
    print("═"*60)
    by_day = df.groupby("date").agg(
        n=("id", "count"),
        pnl=("pnl_usd", "sum"),
        wr=("is_win", "mean"),
    ).reset_index()
    by_day["wr_pct"] = (by_day["wr"] * 100).round(0)

    max_n = by_day["n"].max() or 1
    for _, row in by_day.iterrows():
        bar = _bar(row["n"], max_n, width=20)
        flag = "  🔴 overtrade?" if row["n"] >= 10 else ""
        print(f"  {row['date']}  {bar}  {row['n']:>3} trades  "
              f"${row['pnl']:>+7.2f}  WR={row['wr_pct']:.0f}%{flag}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="G10 Bot Trade Analytics")
    parser.add_argument("--since", help="Filter trades since date (YYYY-MM-DD)")
    parser.add_argument("--pair",  help="Filter to one pair (e.g. AUDJPY)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  G10 BOT — TRADE PERFORMANCE ANALYTICS")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Generated: {ts}")
    if args.since:
        print(f"  Filter:    since {args.since}")
    if args.pair:
        print(f"  Filter:    pair={args.pair}")
    print("═"*60)

    df = load_trades(since=args.since, pair=args.pair)

    if df.empty:
        print("\n  No trades found.\n")
        return

    section_overall(df)
    section_per_pair(df)
    section_freq_cap(df)
    section_regime(df)
    section_session(df)
    section_path(df)
    section_rr(df)
    section_weekday(df)
    section_daily_trades(df)

    print("\n" + "═"*60)
    print("  END OF REPORT")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
