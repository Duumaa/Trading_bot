# CLAUDE.md — G10 Autonomous Dominance Architecture
# For: Claude Code CLI | Updated: 2026-04-06
# Location: /workspaces/Trading_bot/CLAUDE.md

---

## WHO YOU ARE & WHAT THIS IS

You are an expert forex trader AND data scientist helping maintain and improve
an **institutional-grade G10 algorithmic trading bot** that is **LIVE on paper
trading** on a Hetzner CX32 VPS.

The bot is NOT being built from scratch — Phases 1–5 are complete and running.
You are in **Phase 6 (Intelligence Layer)** and beyond.

**Do not re-architect anything without asking first.**
**Do not change risk rules without explicit approval.**
**Always read the file before editing it.**

---

## CURRENT STATUS (2026-04-06)

```
Bot:       LIVE — Hetzner CX32, Ubuntu 24.04
Mode:      PAPER trading (demo $1,448, started ~$1,801 Mar 19)
Service:   systemd g10bot.service — active, auto-restart
Rating:    9.2/10 architecture completeness
```

### Phase Status
| Phase | Name | Status |
|-------|------|--------|
| 1 | Infrastructure | ✅ Complete |
| 2 | Data Pipeline | ✅ Complete |
| 3 | Feature Engineering | ✅ Complete |
| 4 | ML Models (A1+A2+A4+A5+A6) | ✅ Complete |
| 5 | Execution Engine | ✅ Complete |
| 6 | Intelligence Layer | ✅ postmortem_analyzer.py BUILT (2026-03-27) |
| 7 | Monitoring & Deployment | ⏳ Partial |

### Current Trade Stats (Mar 27 session 2 — 210 total trades)
| Category | Count |
|----------|-------|
| CLOSED_MANUAL | 80 |
| CLOSED_SL | 98 |
| CLOSED_TECH_FLIP | 31 |
| CLOSED_TP | 1 |
| **Total** | **210** |

Note: CLOSED_MANUAL jumped 22→80 — high manual intervention rate, worth watching.

### Postmortem Results (first full run — 2026-03-27)
```
Analyzed: 98 losses | Total PnL: $-367.60

Failure Tag Breakdown:
  OVERTRADED_PAIR   67× (68%)  $-171.03   ← #1 problem — frequency not edge
  MODEL_FAILURE     25× (26%)  $-173.49   ← direction was simply wrong
  BAD_TIMING         4× (4%)   $-23.25
  SPREAD_COST        2× (2%)   +$0.17

Most Lost Pairs:
  AUDJPY  24× losses
  NZDJPY  24× losses
  AUDUSD  20× losses
  USDCAD   7× losses
  GBPJPY   6× losses
```
**Key insight:** 68% of losses are OVERTRADED_PAIR — models fire too often on same pairs without enough edge between trades. Fix is frequency caps, not model tuning.

### Per-Pair Performance
| Pair | Verdict |
|------|---------|
| AUDJPY | 🔴 PROBLEM — 26% WR, 37% of all trades, overtraded |
| AUDUSD | ✅ Only profitable pair (+$35.37, +7.2 avg pips) |
| NZDJPY | ⚠️ No edge — 17 trades, near-zero PnL |

---

## WHAT TO BUILD NEXT (in order, do not skip ahead)

### 1. ✅ postmortem_analyzer.py — BUILT (2026-03-27)
**File:** `/workspaces/Trading_bot/intelligence/postmortem_analyzer.py`
**Singleton:** `from intelligence.postmortem_analyzer import postmortem_analyzer`
**Key methods:**
- `postmortem_analyzer.analyze_all_unanalyzed(status_filter="CLOSED_SL")` — backfill/batch
- `postmortem_analyzer.weekly_summary(days_back=30)` — print loss pattern report
- Auto-triggered from `main_loop.py` on every CLOSED_SL trade

**Failure tags used:** `BAD_TIMING | REGIME_WRONG | NEWS_SURPRISE | MODEL_FAILURE | OVERTRADED_PAIR | SPREAD_COST | ENTRY_LEVEL`

**ENTRY_LEVEL tag:** Direction was correct — price moved ≥1R in signal direction within 4h of SL hit. Cause: bad entry price (near S/R, chased, or SL too tight for ATR). Determined by `_fetch_post_sl_price_action()` which checks M15 bars post-exit.

**DB table:** `trade_postmortems`
```
id, trade_id (FK→trades), analyzed_at, failure_reason (text),
market_context (text), analyst_conflict (jsonb), lessons (text[]),
penalty_applied (jsonb), groq_raw_output (text), model_used (varchar50)
```
**Groq rate limit:** Use `time.sleep(5)` between calls — 2s causes 429 errors.
**Telegram command:** `/postmortem` (add to telegram_alerts.py — not yet wired)

---

### 2. ✅ Frequency caps — COMPLETE (2026-03-27 session 2)
Implemented in `execution/risk_manager.py`:

**SL cooldown** (`risk:sl_cooldown:{PAIR}`, TTL=4h, JSON value):
- `CLOSED_SL` → blocks BOTH directions: 0–2h hard block, 2–4h soft block (×0.5 conf)
- `CLOSED_TECH_FLIP` → blocks SAME direction only for 2h; opposite direction allowed immediately
- Redis value format: `{"set_at": "ISO", "type": "CLOSED_SL|CLOSED_TECH_FLIP", "direction": "BUY|SELL"}`
- Legacy plain ISO string values handled via fallback (treated as CLOSED_SL, both directions)

**Session cap** (`risk:session_cap:{PAIR}:{SESSION}`, TTL=seconds until session end):
- AUDJPY and NZDJPY only: max 1 trade per session (ASIA=00–08h, LONDON=08–16h, NY=16–24h UTC)
- Set in `record_trade_open()` on every trade open for those pairs
- Checked as gate 6c in `approve_trade()`

**Gate order in `approve_trade()`:**
```
6.  Duplicate pair check
6b. SL / TECH_FLIP cooldown (directional-aware)
6c. Session cap (AUDJPY/NZDJPY only)
7.  Correlation guard
```

### 3. ⭐ features/sr_features.py — NEXT SESSION PRIORITY
Support/Resistance level engine — Redis key `features:sr:{PAIR}`
- Swing highs/lows (H1+D1 timeframes), round numbers, weekly/daily high-low
- Output: `nearest_resistance`, `nearest_support`, `distance_to_resistance_pips`
- Plug into Gate 4: block entry if < 15 pips from resistance (for LONG)

### 4. Price target engine (after S/R features)
- SL = daily ATR × 1.0–1.5 | Target = daily ATR × 2.0–3.0 (regime-dependent)
- Snap target to nearest S/R level
- Store as `price_target` and `r_ratio` in signal output

### 5. /postmortem Telegram command (wire into telegram_alerts.py)
Call `postmortem_analyzer.weekly_summary()` and send formatted result.

### 6. apex_framework.py (after 60+ tagged CLOSED_SL trades)
Dynamic threshold adjustment based on postmortem patterns.

### 7. Session blackout rules (after postmortem confirms pattern)
Candidate rules:
- Sunday full day blackout ✅ (already live)
- Monday 00:00–07:29 UTC ✅ (already live)
- Daily 21:00–21:59 UTC rollover ✅ (already live)
- AUDJPY session filter → add only after postmortem confirms specific session

### 8. backtest.py (after April 10 — 30 days live signal data)

### 9. Grafana dashboard (Phase 7)

### CANCELLED — A3 TFT (permanently removed 2026-03-31)
### NOT YET — Market Maker Move model (Oct–Nov 2026)

---

## ARCHITECTURE — THE 4-GATE SYSTEM

```
Gate 1 — FUNDAMENTAL (must pass first)
  A1 LightGBM → regime (THV/TLV/MR/PNS) + probabilities
  A2 Endo/Exo → macro score -1 to +1
  Regime stability check → gap < 0.25 triggers 0.25x size
  Both must agree on direction. PNS = always flat.

Gate 2 — SENTIMENT (must not oppose Gate 1)
  A5 Groq LLaMA → daily sentiment score
  A6 Opus weekly → Hawk/Dove/Strategist macro bias

Gate 3 — TECHNICAL (must agree on direction)
  A4 TCN (AUC 0.737) + PatchTSMixer (AUC 0.719) — ONNX inference
  Both models must agree. MR regime: A4 provides direction when A2 neutral.

Gate 4 — ENTRY ENGINE
  VWAP position + RSI state + OBI + momentum
  Score ≥ 0.65 → enter | 0.40–0.65 → PASS_WAIT | < 0.40 → block

RULE: NEVER ensemble across gate groups. Each gate is independent.
```

### Regime Sizing
| Regime | Size | Min Confidence | Notes |
|--------|------|----------------|-------|
| THV | 1.0x | 0.55 | Trending High Vol |
| TLV | 0.5x | 0.65 | Weakest regime — reduced size |
| MR | 0.5x | 0.60 | Fade extremes |
| PNS | 0.0x | — | Always flat |
| Unstable | ×0.25 | — | Applied on top of above |

### Trailing Stop Stages
- Stage 0: SL at ATR × 1.5
- Stage 1: Breakeven — profit ≥ +1R → move SL to entry
- Stage 2: Trail — profit ≥ +2R → trail behind price
- Stage 3: Lock — profit ≥ +2.5R → lock 2.5R profit

---

## HARD RISK RULES (never change without explicit approval)

| Rule | Value |
|------|-------|
| Max risk per trade | 2% of account |
| Max concurrent trades | 3 directional + 3 hedge = 6 total (separate caps, see position cap split) |
| Daily loss kill switch | -5% → halt all trading |
| Pre-event blackout | 2h before FOMC/NFP/ECB/BoE/BoJ |
| Correlation guard | Max 2 pairs per currency, 3% net exposure |
| Portfolio stop | -20% margin → halt + reassess |

---

## FILE STRUCTURE

```
/workspaces/Trading_bot/
├── config/
│   ├── settings.py
│   └── ctrader_config.yaml
├── data_pipeline/
│   ├── storage/timeseries_db.py          ← DB singleton: use db.connection()
│   └── connectors/
│       ├── ctrader_rest_api.py           ← M15 bars, REST
│       ├── ctrader_live_stream.py        ← Live ticks WebSocket
│       ├── ctrader_fix.py                ← OpenAPI WebSocket (tick:<pair> Redis)
│       ├── forexfactory_calendar.py      ← Economic calendar
│       ├── cot_downloader.py             ← COT TFF reports
│       ├── rss_news_scraper.py           ← ForexLive + FXStreet
│       └── yield_scraper.py              ← FRED + ECB
├── features/
│   ├── yield_features.py                 ← 30 features, TTL=4h
│   ├── statistical_features.py           ← 17 features, TTL=1h
│   ├── vwap_features.py                  ← 12 features, TTL=15min
│   ├── orderflow_features.py             ← 12 features, TTL=5min
│   ├── liquidity_features.py             ← 10 features, TTL=5min
│   └── event_features.py                 ← 8 features, TTL=1h
├── models/
│   ├── analyst1_regime/regime_classifier.py
│   ├── analyst2_endo_exo/score_calculator.py
│   ├── analyst4_inference/analyst4_inference.py  ← ONNX, CPU-only
│   ├── analyst5_groq/groq_sentiment.py
│   └── analyst6_opus/opus_deep_analysis.py
├── execution/
│   ├── signal_aggregator.py              ← 4-gate logic + stability check
│   ├── risk_manager.py                   ← sizing + kill switches + corr guard
│   ├── order_executor.py                 ← cTrader demo orders
│   ├── trailing_stop.py                  ← 3-stage trailing stop
│   └── main_loop.py                      ← M15 loop, all threads
├── intelligence/
│   ├── postmortem_analyzer.py            ← ✅ BUILT — Groq loss classifier, auto-trigger on CLOSED_SL
│   └── monitoring/
│       └── telegram_alerts.py            ← /signals /price /groq /status /trades
└── .env
```

**ONNX models location:** check `analyst4_inference.py` for exact paths.
**A3 TFT:** PERMANENTLY CANCELLED — GPU required, Hetzner CX32 is CPU-only, ONNX export unreliable.
  If a 3rd technical model is ever added: use LightGBM/XGBoost with lag features (CPU-native, ONNX-friendly).

---

## DATABASE — REAL SCHEMA (confirmed 2026-03-09)

### Table Name Corrections (brief is wrong, these are real)
| Brief Says | Actual Table | Notes |
|------------|--------------|-------|
| ohlcv_data | **ohlcv** | column: `time` not `timestamp` |
| tick_data | **ticks** | column: `time` not `timestamp` |
| order_book | **order_book_snapshots** | |
| yield_data | **yield_data** ✅ | DATE PK |
| — | **bond_yields** | TIMESTAMPTZ, separate table |

### trades Table (real columns — Mar 2026)
Run `\d trades` to get full schema. Key columns confirmed:
```
id, fix_order_id, ctrader_position_id, symbol, direction,
entry_price, stop_loss, take_profit, lot_size, status,
opened_at → USE: check actual column name with \d trades
```
**ALWAYS run `\d tablename` before writing any INSERT — schema has surprises.**

### Key Schema Rules
```sql
-- economic_events dedup (fixed 2026-03-23)
ON CONFLICT (event_time, country, event_name) DO NOTHING

-- ticks: filter out epoch-zero rows
WHERE time > '2020-01-01'

-- JPY pairs stored as price×100 in ohlcv
-- Divide by 100 if value > 1000: USDJPY 149.00 stored as 14900.00
-- Also applies to ticks table (check bid > 1000)

-- Always use UTC timestamps
-- datetime.now(timezone.utc) not datetime.now()
```

### DB Access Pattern (always use this)
```python
from data_pipeline.storage.timeseries_db import db
from sqlalchemy import text

with db.connection() as conn:
    result = conn.execute(text("SELECT ..."), {"param": value})
    rows = result.fetchall()

# NEVER use db.query() — does not exist
# ALWAYS wrap in text() and use named params :name
```

### psql Direct Access
```bash
docker exec trading_bot-timescaledb-1 psql -U g10bot -d g10bot -c "SELECT ..."
# NO -it flag — Hetzner has no TTY for docker exec
```

---

## REDIS KEY MAP (confirmed)

| Key | Contents | TTL |
|-----|----------|-----|
| `features:regime:{PAIR}` | A1 regime + probabilities | 4h |
| `features:analyst2:{PAIR}` | A2 endo/exo score + bias + drivers | 4h |
| `features:sentiment:{PAIR}` | A5 Groq sentiment | 24h |
| `features:analyst4:{PAIR}` | A4 TCN+Patch signal | 15min |
| `features:opus:weekly` | A6 Opus weekly bias | 7d |
| `features:stat:{PAIR}` | Statistical features | 1h |
| `features:vwap:{PAIR}` | VWAP features | 15min |
| `features:yield:{PAIR}` | Yield/carry features | 4h |
| `features:event:{PAIR}` | Event proximity | 1h |
| `features:order:{PAIR}` | Order flow | 5min |
| `features:liq:{PAIR}` | Liquidity | 5min |
| `signal:{PAIR}` | Full signal decision | 15min |
| `regime:stability:{PAIR}` | Last 6 regime prob readings | 2h |
| `tick:{PAIR}` | Latest bid/ask from WebSocket | live |
| `risk:account_state` | Balance/equity/daily PnL | 5min |
| `risk:halted` | "true"/"false" | permanent |
| `risk:halt_reason` | Reason string | permanent |
| `risk:sl_cooldown:{PAIR}` | JSON `{set_at,type,direction}` — SL/flip cooldown | 4h |
| `risk:session_cap:{PAIR}:{SESSION}` | "1" — AUDJPY/NZDJPY session trade taken | until session end |

### Redis Connection (ALWAYS include password)
```python
import redis, os
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
)
```

### A2 Data Access Pattern
```python
d = json.loads(r.get(f"features:analyst2:{pair}"))
score       = d['score']   # -1.0 to +1.0
bias        = d['bias']    # "LONG_BASE" | "SHORT_BASE" | "NEUTRAL"
carry_score = d['drivers']['carry_score']
```

---

## ENVIRONMENT

### Server
- Hetzner CX32: 4 vCPU, 8GB RAM, 80GB SSD, Ubuntu 24.04
- Workspace: `/workspaces/Trading_bot`
- Service: `sudo systemctl [start|stop|restart|status] g10bot`
- Logs: `sudo journalctl -u g10bot -f` or `-n 50 --no-pager`

### Docker Stack
| Container | Port | |
|-----------|------|-|
| trading_bot-timescaledb-1 | 5432 | ✅ |
| trading_bot-redis-1 | 6379 | ✅ password auth |
| trading_bot-influxdb-1 | 8086 | ✅ |
| trading_bot-mlflow-1 | 5000 | ✅ |

### .env Key Variables
```
FRED_API_KEY, CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET,
CTRADER_ACCOUNT_ID, CTRADER_ACCESS_TOKEN,
REDIS_PASSWORD, DB_PASSWORD,
GROQ_API_KEY, ANTHROPIC_API_KEY,
TRADING_MODE=PAPER
```

### Load .env (always do this first in scripts)
```python
from dotenv import load_dotenv
load_dotenv('/workspaces/Trading_bot/.env')
# MUST be before any module import that reads env vars
```

### cTrader Symbol IDs
```python
SYMBOL_MAP = {
    "EURUSD": 1,  "GBPUSD": 2,  "EURJPY": 3,
    "USDJPY": 4,  "AUDUSD": 5,  "USDCHF": 6,
    "GBPJPY": 7,  "USDCAD": 8,  "EURGBP": 9,
    "AUDJPY": 11, "NZDUSD": 12, "NZDJPY": 21,
}
```

---

## CRITICAL GOTCHAS (never repeat these)

```
❌ db.query()         → does not exist. Use db.connection()
❌ docker exec -it    → no TTY on Hetzner. Remove -it flag
❌ datetime.now()     → always use datetime.now(timezone.utc)
❌ print()            → use logger.info/success/error (loguru)
❌ ohlcv_data table   → real name is ohlcv, column is time not timestamp
❌ tick_data table    → real name is ticks
❌ trades.opened_at   → column may not exist, check with \d trades first
❌ Raw Python in bash → always wrap: python3 << 'PYEOF' ... PYEOF
❌ Kill switch using hardcoded balance → must fetch live daily_start_balance
   from Redis key risk:account_state at midnight UTC reset

✅ JPY pairs: divide price by 100 if value > 1000 (stored as price×100)
✅ Ticks: filter WHERE time > '2020-01-01' (epoch-zero bug rows exist)
✅ TFF COT file: URL is fut_fin_txt_{year}.zip, file inside is FinFutYY.txt
✅ COT dedup: 3 EUR contracts in TFF → keep highest open_interest per date
✅ economic_events: column is event_name (not event)
✅ impact column: uppercase 'HIGH'/'MEDIUM'/'LOW' (not boolean)
✅ currencies array: CAST(:currencies AS VARCHAR(3)[]) with "{USD,EUR}" format
✅ ForexFactory methods: scrape_current_week() and scrape_next_week()
   → NO fetch_and_store() method
✅ news_scraper: real method is scrape_feeds() (not scrape_all())
✅ yield_scraper: check actual file for method name (not scrape_all())
✅ live_stream: check actual file for correct start method name
✅ ONNX only on VPS — A4 models exported to ONNX (no PyTorch on server)
✅ Sunday open blackout: confirmed dangerous (gap hit EURGBP SL Mar 23)
✅ Monday 00:00–07:29 UTC: thin liquidity blackout (already in risk_manager)
✅ Daily 21:00–21:59 UTC: rollover blackout (already in risk_manager)
✅ OpenAPI WebSocket (ctrader_fix.py) updates tick:{pair} in Redis every tick
✅ TLV regime: weakest precision (62.3%) → 0.5x size, 0.65 min confidence
✅ Regime stability: if A1 top-2 regime gap < 0.25 → apply 0.25x size mult

✅ risk:sl_cooldown value is JSON not plain string (as of 2026-03-27 session 2)
   Format: {"set_at": "ISO", "type": "CLOSED_SL|CLOSED_TECH_FLIP", "direction": "BUY|SELL"}
   Legacy plain ISO strings still exist in Redis — fallback handles them (treated as CLOSED_SL)
✅ _is_pair_in_sl_cooldown(pair, direction) — takes direction param, must pass it from approve_trade
✅ record_trade_close already fetches direction from DB at line ~1140 — no extra query needed
✅ CLOSED_TECH_FLIP was NOT triggering cooldown in original code — it was fixed this session
✅ approve_trade already receives direction param — no signature change needed to wire gate 6b
✅ Session cap keys: risk:session_cap:{PAIR}:{SESSION} — TTL = seconds to session boundary, not flat 8h
   Sessions: ASIA=00:00–08:00 UTC, LONDON=08:00–16:00 UTC, NY=16:00–24:00 UTC
❌ CLAUDE.md spec said cooldown key = cooldown:{PAIR} — actual key is risk:sl_cooldown:{PAIR}
   Do not create a new key with the spec name — use the existing risk: namespace

❌ A5 Redis field name `score` → does not exist. Use `sentiment`. Full structure: {"sentiment": float, "bias": str, "confidence": float, "pair": str, "currencies": {}, "updated_at": str, "article_count": int}

✅ MACRO_OVERRIDE fires when A1=MR AND |A5 sentiment|≥0.20 AND A5 conf≥0.30 AND A6 conv≥0.45 AND A5/A6 agree direction. Overrides regime MR→TLV at 0.5x size. A4 Gate 3 still required. Constants: MACRO_OVERRIDE_SENT_MIN=0.20, MACRO_OVERRIDE_SENT_CONF=0.30, MACRO_OVERRIDE_OPUS_CONV=0.45

✅ Position cap is 3+3 SPLIT (2026-04-02): directional cap=3 (risk_manager), hedge cap=3 (hedge_manager)
   _count_open_positions() subtracts ALL hedge legs (×2 per pair) — hedge trades NEVER consume directional slots
   Total max = 6 open positions simultaneously. This is working correctly — do not change without approval.

✅ Startup reconcile was ONE-DIRECTIONAL (fixed 2026-04-02):
   Old: only checked DB OPEN → cTrader (found DB trades missing from cTrader)
   New: also checks cTrader positions → DB (finds cTrader positions with no DB record = zombies)
   Zombie recovery: INSERT into trades with real ctrader_position_id, fix_order_id='RECOVERED_{pos_id}',
   signal_path='RECOVERED', volume_units=3000. Trailing stop + liveness check then manage them.
```

---

## FEATURE ENGINEERING PATTERN (all modules follow this)

```python
class XFeatureEngine:
    def get_features(self, pair: str) -> Dict[str, float]:
        cached = self._redis_get(f"features:x:{pair}")
        if cached:
            return cached
        return self.compute_all().get(pair, {})

    def compute_all(self) -> Dict[str, Dict[str, float]]:
        # compute → store in Redis with TTL → return
        pass

    def start_scheduler(self) -> None:
        # Background daemon thread, runs compute_all() on schedule
        pass

engine = XFeatureEngine()   # module singleton at bottom
```

Redis key: `features:{module}:{PAIR}` | TTL: 14400s (4h) standard

---

## KNOWN METHOD NAME BUGS (fix when encountered)

These are in `main_loop.py` background threads — correct names TBD by reading actual files:
- `news_scraper.scrape_all()` → real method: check `rss_news_scraper.py`
- `yield_scraper.scrape_all()` → real method: check `yield_scraper.py`
- `live_stream.start()` → real method: check `ctrader_live_stream.py`

---

## TELEGRAM COMMANDS (confirmed working)

| Command | Output |
|---------|--------|
| /signals | All 12 pairs — regime, gate, direction, size |
| /price | Live bid/ask + spread for all pairs |
| /groq | A5 Groq sentiment + A6 Opus weekly bias |
| /status | Account balance, daily P&L, open positions |
| /trades | Recent closed trades |

---

## SESSION HEALTH CHECK COMMANDS

```bash
# 1. Bot service status
sudo systemctl status g10bot
sudo journalctl -u g10bot -n 20 --no-pager

# 2. Open positions
docker exec trading_bot-timescaledb-1 psql -U g10bot -d g10bot \
  -c "SELECT id, symbol, direction, entry_price, stop_loss, status FROM trades WHERE status='OPEN';"

# 3. Postmortem readiness
docker exec trading_bot-timescaledb-1 psql -U g10bot -d g10bot \
  -c "SELECT status, COUNT(*) FROM trades GROUP BY status ORDER BY status;"

# 4. Redis feature health
python3 << 'PYEOF'
from dotenv import load_dotenv
load_dotenv('/workspaces/Trading_bot/.env')
import redis, os
r = redis.Redis(host="localhost", port=6379,
    password=os.getenv("REDIS_PASSWORD") or None, decode_responses=True)
checks = {
    "regime": "features:regime:*",
    "analyst2": "features:analyst2:*",
    "analyst4": "features:analyst4:*",
    "signal": "signal:*",
    "stability": "regime:stability:*",
}
for name, pattern in checks.items():
    count = len(list(r.scan_iter(pattern)))
    print(f"  {name:<12} {count:>3} keys {'✅' if count >= 10 else '❌ EMPTY'}")
PYEOF

# 5. Force signal recompute if signal keys are empty
python3 << 'PYEOF'
from dotenv import load_dotenv
load_dotenv('/workspaces/Trading_bot/.env')
import sys; sys.path.insert(0, '/workspaces/Trading_bot')
from execution.signal_aggregator import signal_aggregator
result = signal_aggregator.evaluate_all()
errors = [p for p, d in result.items() if d.get("gate1") == "FAIL:ERROR"]
print(f"Computed: {len(result)} pairs | Errors: {errors}")
PYEOF
```

---

## CODE QUALITY STANDARDS (every new file must follow)

```python
"""
module/path.py
==============
One-line summary.

Longer description. Trading rationale for key decisions.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from loguru import logger
import numpy as np
import pandas as pd

class ClassName:
    """Docstring."""

    def method(self, param: str) -> Dict[str, float]:
        """What this does. Args/Returns documented."""
        logger.info(f"ClassName | Starting {param}...")
        try:
            result = self._do_work()
            logger.success(f"ClassName | Done: {len(result)} items")
            return result
        except Exception as exc:
            logger.error(f"ClassName | Failed: {exc}")
            return {}

engine = ClassName()   # module singleton — always at bottom
```

---

## TRADING PRINCIPLES (always apply in code)

- **Direction > absolute level** — always compute 4w AND 13w trend for yield differentials and COT
- **Extreme differentials = lower conviction** — non-linear scoring: |diff| > 4% → score drops from 0.9 → 0.6
- **All features as pairs** — never single-currency features alone
- **Positive carry = credit, negative = debit** — rollover_daily_usd is a viability filter
- **Endo vs Exo** — endogenous (GDP/CPI/PMI) and exogenous (yield diff/COT/flows) feed A2 separately
- **When conviction extreme and extreme move happens → step down** (APEX principle, Phase 6)

---

## IMPROVEMENT ROADMAP (March 2026 research)

### Tier 1 — Build after postmortem (high impact, low effort)
1. **S/R features** (`features/sr_features.py`) — finishing line for trades
2. **Price target engine** — SL=ATR×1.5, target=ATR×3.0, snap to S/R
3. **Cross-pair exposure netting** — correlation guard already blocks, but net delta calc missing
4. **Session filter** — Asia/London/NY session tagging in entry engine

### Tier 2 — Build after 60+ trades (medium effort)
5. **Real interest rate differential** — nominal yield - CPI = real yield (stronger than nominal alone)
6. **Risk reversal features** — 25-delta RR from options market (institutional positioning signal)
7. **AUDJPY frequency cap** — max 1 AUDJPY trade per session until postmortem data confirms fix

### Tier 3 — Phase 6+ (high effort)
8. **apex_framework.py** — dynamic threshold injection based on postmortem patterns
9. **retraining_loop.py** — monthly retrain A1/A4 + MLflow A/B test
10. **drift_detector.py** — feature/concept drift alerts
11. ~~**A3 TFT**~~ — CANCELLED. GPU required, VPS is CPU-only. Replace with LightGBM lag-feature model if 3rd signal ever needed.

---

## GO-LIVE CHECKLIST (do not skip)

```
□ 1 month paper trading (started Mar 19 → ready ~Apr 19)
□ 100+ total closed trades
□ Win rate > 55% on CLOSED_SL trades
□ Sharpe ratio > 1.0
□ Max drawdown < 15%
□ postmortem_analyzer.py built and running
□ Session blackouts confirmed by postmortem data
□ All kill switches tested and verified
□ Daily balance reset confirmed working (fix kill switch bug)
□ Sunday open blackout added
□ Grafana dashboard running
□ Unit tests written

SCALE PLAN:
  Now:     Demo $1,961 virtual
  Month 2: Micro live $50–100
  Month 4: Scale if profitable: $100 → $200 → $500
```

---

## DECISIONS LOG

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-09 | TFT (A3) deferred | Need more paper trading data first |
| 2026-03-31 | TFT (A3) CANCELLED permanently | GPU required, VPS is CPU-only, ONNX export unreliable. Future 3rd model = LightGBM lag features |
| 2026-03-09 | ONNX for A4 | No PyTorch on VPS — CPU-only inference |
| 2026-03-23 | TLV size → 0.5x | 62.3% precision = weakest regime |
| 2026-03-23 | Regime stability check | Unstable regimes → 0.25x size, not blocked |
| 2026-03-23 | Correlation guard | 3 JPY longs = 6% correlated risk — capped at 2 |
| 2026-03-23 | Sunday blackout | Gap hit EURGBP SL — confirmed dangerous |
| 2026-03-25 | OpenAPI WebSocket | Replaces FIX, tick:{pair} in Redis <10ms |
| 2026-03-25 | Kill switch bug found | Was using hardcoded balance not live daily reset |
| 2026-03-27 | postmortem_analyzer.py built | 40+ CLOSED_SL threshold reached |
| 2026-03-27 | Groq sleep = 5s in postmortem | 2s causes 429 rate limit, 5s reliable |
| 2026-03-27 | order_executor _get_closed_deal_price fixed | Now uses tick Redis + ticks DB as fallback for real close price |
| 2026-03-27 | #1 fix = frequency caps, not model tuning | Postmortem: 68% losses = OVERTRADED_PAIR |
| 2026-03-27 | Do not adjust AUDJPY thresholds yet | Wait for frequency cap + more postmortem data first |
| 2026-03-27 | Moved to Claude Code | Web chat → CLI for Phase 6 builds |
| 2026-03-27 | CLOSED_TECH_FLIP added to SL cooldown trigger | Was missing — 31 trades were re-entering immediately after regime flip |
| 2026-03-27 | CLOSED_TECH_FLIP gets directional block (Option B) | Flip IS the new signal — blocking opposite direction for 4h wastes the trend move. Same direction blocked 2h, opposite free immediately |
| 2026-03-27 | Cooldown Redis value changed to JSON | Needed to carry close type + direction for directional logic. Legacy plain ISO string handled via fallback |
| 2026-03-27 | Session cap set at trade OPEN not trade CLOSE | Prevents 2nd entry in same session even before first trade resolves |
| 2026-04-01 | S/R engine rebuilt with 7 sources + 2-tier output | EMAs needed for hedge cycle trigger; structural only for SL sizing |
| 2026-04-01 | Hedge profit preservation: SL walk + profit target + time stop | Captures directional breakouts from range without unlimited risk |
| 2026-04-01 | Position cap split to 3+3 (directional + hedge separate) | Hedges were consuming all 3 slots and blocking directional trades |
| 2026-04-01 | Three-stage asymmetric hedge exit (CYCLING→WINNER_RUNNING→LOSER_INVALIDATED) | Asymmetric: let winner run, aggressively cut loser when trend confirmed |
| 2026-04-01 | Gate 1 A2 veto removed — combined A1×A2 confidence formula | A2 NEUTRAL was blocking 40%+ of valid THV/TLV signals unnecessarily |
| 2026-04-02 | Startup reconcile fixed — reverse check added (cTrader→DB) | Found 3 zombie positions bot didn't know about (DB insert failure race condition) |
| 2026-04-06 | Hedge manager DISABLED | Backtest: avg 1.1 cycles vs 3+ needed. WINNER_RUNNING fires at 3 pips while loser is -50 pips → guaranteed net loss |
| 2026-04-06 | A7 SMC LightGBM built and wired | AUC 0.604 entry quality model. Applied as ×0.90–×1.15 multiplier on entry_score in entry_engine.py |
| 2026-04-06 | MACRO_OVERRIDE gate added to signal_aggregator.py | A1=MR but A5+A6 both agree with conviction → override regime MR→TLV, direction from macro consensus. A4 confirmation still required |
| 2026-04-06 | A5 Redis field is `sentiment` not `score` | A5 JSON stores `{"sentiment": -0.25, "bias": "BEARISH", "confidence": 0.38}` — never use d.get('score') |
| 2026-04-06 | A5/A6 weaker than A1 normally, stronger in macro shocks | Criteria: |A5 sentiment|≥0.20, A5 conf≥0.30, A6 conv≥0.45, both agree on direction, A1=MR |
