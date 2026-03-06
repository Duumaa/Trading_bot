-- =============================================================================
-- G10 Dominance Architecture — TimescaleDB Schema
-- =============================================================================
-- Run order: extensions → core tables → hypertables → indexes → views
-- Compatible with: TimescaleDB 2.x on PostgreSQL 15
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- =============================================================================
-- SECTION 1: MARKET DATA
-- =============================================================================

CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ      NOT NULL,
    symbol      VARCHAR(10)      NOT NULL,
    timeframe   VARCHAR(5)       NOT NULL DEFAULT 'M15',
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL DEFAULT 0,
    source      VARCHAR(20)      NOT NULL DEFAULT 'ctrader',
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days');
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
    ON ohlcv (time, symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON ohlcv (symbol, time DESC);

CREATE TABLE IF NOT EXISTS ticks (
    time    TIMESTAMPTZ      NOT NULL,
    symbol  VARCHAR(10)      NOT NULL,
    bid     DOUBLE PRECISION NOT NULL,
    ask     DOUBLE PRECISION NOT NULL,
    spread  DOUBLE PRECISION GENERATED ALWAYS AS (ask - bid) STORED,
    source  VARCHAR(20)      NOT NULL DEFAULT 'fix'
);
SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time
    ON ticks (symbol, time DESC);
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ticks', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    time                TIMESTAMPTZ      NOT NULL,
    symbol              VARCHAR(10)      NOT NULL,
    bids                JSONB            NOT NULL,
    asks                JSONB            NOT NULL,
    bid_ask_imbalance   DOUBLE PRECISION,
    total_bid_volume    DOUBLE PRECISION,
    total_ask_volume    DOUBLE PRECISION,
    depth_levels        SMALLINT         NOT NULL DEFAULT 20
);
SELECT create_hypertable('order_book_snapshots', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS idx_ob_symbol_time
    ON order_book_snapshots (symbol, time DESC);
ALTER TABLE order_book_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('order_book_snapshots', INTERVAL '3 days', if_not_exists => TRUE);

-- =============================================================================
-- SECTION 2: FUNDAMENTAL DATA
-- =============================================================================

CREATE TABLE IF NOT EXISTS bond_yields (
    time        TIMESTAMPTZ      NOT NULL,
    currency    VARCHAR(3)       NOT NULL,
    tenor       VARCHAR(5)       NOT NULL,
    yield_pct   DOUBLE PRECISION NOT NULL,
    source      VARCHAR(30)      NOT NULL DEFAULT 'manual',
    created_at  TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('bond_yields', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days');
CREATE UNIQUE INDEX IF NOT EXISTS idx_yields_unique
    ON bond_yields (time, currency, tenor);
CREATE INDEX IF NOT EXISTS idx_yields_currency_time
    ON bond_yields (currency, time DESC);

CREATE TABLE IF NOT EXISTS yield_spreads (
    time            TIMESTAMPTZ      NOT NULL,
    pair            VARCHAR(10)      NOT NULL,
    base_currency   VARCHAR(3)       NOT NULL,
    quote_currency  VARCHAR(3)       NOT NULL,
    tenor           VARCHAR(5)       NOT NULL,
    spread_bps      DOUBLE PRECISION NOT NULL,
    zscore_5d       DOUBLE PRECISION,
    zscore_20d      DOUBLE PRECISION,
    cip_deviation   DOUBLE PRECISION
);
SELECT create_hypertable('yield_spreads', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days');
CREATE UNIQUE INDEX IF NOT EXISTS idx_spreads_unique
    ON yield_spreads (time, pair, tenor);

CREATE TABLE IF NOT EXISTS macro_indicators (
    time            TIMESTAMPTZ      NOT NULL,
    country         VARCHAR(3)       NOT NULL,
    indicator       VARCHAR(50)      NOT NULL,
    actual          DOUBLE PRECISION,
    forecast        DOUBLE PRECISION,
    previous        DOUBLE PRECISION,
    surprise_score  DOUBLE PRECISION GENERATED ALWAYS AS (
                        CASE WHEN forecast IS NOT NULL AND forecast != 0
                             THEN (actual - forecast) / ABS(forecast) * 100
                             ELSE NULL END
                    ) STORED,
    source          VARCHAR(30)      NOT NULL DEFAULT 'manual',
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('macro_indicators', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '90 days');
CREATE INDEX IF NOT EXISTS idx_macro_country_indicator
    ON macro_indicators (country, indicator, time DESC);

-- =============================================================================
-- SECTION 3: POSITIONING DATA
-- =============================================================================

CREATE TABLE IF NOT EXISTS cot_reports (
    report_date     DATE    NOT NULL,
    currency        VARCHAR(3) NOT NULL,
    nc_long         BIGINT  NOT NULL DEFAULT 0,
    nc_short        BIGINT  NOT NULL DEFAULT 0,
    nc_net          BIGINT  GENERATED ALWAYS AS (nc_long - nc_short) STORED,
    nc_net_prev     BIGINT,
    nc_net_change   BIGINT,
    comm_long       BIGINT  NOT NULL DEFAULT 0,
    comm_short      BIGINT  NOT NULL DEFAULT 0,
    open_interest   BIGINT  NOT NULL DEFAULT 0,
    nc_net_pct_oi   DOUBLE PRECISION,
    source          VARCHAR(30) NOT NULL DEFAULT 'cftc',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (report_date, currency)
);
CREATE INDEX IF NOT EXISTS idx_cot_currency_date
    ON cot_reports (currency, report_date DESC);

-- =============================================================================
-- SECTION 4: NEWS & SENTIMENT
-- =============================================================================

CREATE TABLE IF NOT EXISTS economic_events (
    id          SERIAL      PRIMARY KEY,
    event_time  TIMESTAMPTZ NOT NULL,
    country     VARCHAR(3)  NOT NULL,
    currency    VARCHAR(3)  NOT NULL,
    event_name  VARCHAR(200) NOT NULL,
    impact      VARCHAR(10) NOT NULL CHECK (impact IN ('HIGH', 'MEDIUM', 'LOW')),
    actual      VARCHAR(20),
    forecast    VARCHAR(20),
    previous    VARCHAR(20),
    source      VARCHAR(30) NOT NULL DEFAULT 'forexfactory',
    scraped_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_time_impact
    ON economic_events (event_time, impact);
CREATE INDEX IF NOT EXISTS idx_events_currency_time
    ON economic_events (currency, event_time);

CREATE TABLE IF NOT EXISTS news_articles (
    id              SERIAL      PRIMARY KEY,
    published_at    TIMESTAMPTZ NOT NULL,
    source          VARCHAR(50) NOT NULL,
    title           TEXT        NOT NULL,
    summary         TEXT,
    url             TEXT,
    currencies      VARCHAR(3)[] NOT NULL DEFAULT '{}',
    sentiment_score DOUBLE PRECISION,
    impact_level    VARCHAR(10),
    processed       BOOLEAN     NOT NULL DEFAULT FALSE,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_news_published
    ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_processed
    ON news_articles (processed, published_at DESC) WHERE NOT processed;
CREATE INDEX IF NOT EXISTS idx_news_title_trgm
    ON news_articles USING GIN (title gin_trgm_ops);

-- =============================================================================
-- SECTION 5: ANALYST SIGNALS
-- =============================================================================

CREATE TABLE IF NOT EXISTS analyst_signals (
    time          TIMESTAMPTZ      NOT NULL,
    analyst_id    SMALLINT         NOT NULL CHECK (analyst_id BETWEEN 1 AND 6),
    analyst_name  VARCHAR(50)      NOT NULL,
    pair          VARCHAR(10),
    currency      VARCHAR(3),
    signal_type   VARCHAR(30)      NOT NULL,
    signal_value  DOUBLE PRECISION NOT NULL,
    confidence    DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    metadata      JSONB            NOT NULL DEFAULT '{}',
    model_version VARCHAR(50),
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('analyst_signals', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days');
CREATE INDEX IF NOT EXISTS idx_signals_analyst_time
    ON analyst_signals (analyst_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_pair_time
    ON analyst_signals (pair, time DESC) WHERE pair IS NOT NULL;

CREATE TABLE IF NOT EXISTS regime_history (
    time          TIMESTAMPTZ      NOT NULL,
    pair          VARCHAR(10)      NOT NULL,
    regime        VARCHAR(30)      NOT NULL CHECK (
                      regime IN ('TRENDING_HV','TRENDING_LV','MEAN_REVERTING','PRE_NEWS_STAGNATION')
                  ),
    confidence    DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    prob_thv      DOUBLE PRECISION,
    prob_tlv      DOUBLE PRECISION,
    prob_mr       DOUBLE PRECISION,
    prob_pns      DOUBLE PRECISION,
    model_version VARCHAR(50)
);
SELECT create_hypertable('regime_history', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days');
CREATE INDEX IF NOT EXISTS idx_regime_pair_time
    ON regime_history (pair, time DESC);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    time          TIMESTAMPTZ      NOT NULL,
    analyst_id    SMALLINT         NOT NULL CHECK (analyst_id IN (5, 6)),
    currency      VARCHAR(3)       NOT NULL,
    score         DOUBLE PRECISION NOT NULL CHECK (score BETWEEN -1 AND 1),
    confidence    DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    stance        VARCHAR(30),
    nps           DOUBLE PRECISION,
    articles_count INTEGER,
    model_used    VARCHAR(50),
    raw_output    JSONB            NOT NULL DEFAULT '{}',
    valid_until   TIMESTAMPTZ      NOT NULL
);
SELECT create_hypertable('sentiment_scores', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days');
CREATE INDEX IF NOT EXISTS idx_sentiment_currency_time
    ON sentiment_scores (currency, time DESC);

-- =============================================================================
-- SECTION 6: EXECUTION & TRADES
-- =============================================================================

CREATE TABLE IF NOT EXISTS trades (
    id                  SERIAL          PRIMARY KEY,
    fix_order_id        VARCHAR(50)     UNIQUE,
    ctrader_position_id VARCHAR(50),
    symbol              VARCHAR(10)     NOT NULL,
    direction           VARCHAR(5)      NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    volume_units        BIGINT          NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL,
    exit_price          DOUBLE PRECISION,
    stop_loss           DOUBLE PRECISION NOT NULL,
    take_profit         DOUBLE PRECISION NOT NULL,
    entry_time          TIMESTAMPTZ     NOT NULL,
    exit_time           TIMESTAMPTZ,
    status              VARCHAR(20)     NOT NULL DEFAULT 'OPEN'
                            CHECK (status IN ('OPEN','CLOSED_TP','CLOSED_SL','CLOSED_MANUAL','CANCELLED')),
    pnl_pips            DOUBLE PRECISION,
    pnl_usd             DOUBLE PRECISION,
    regime              VARCHAR(30),
    signal_1_regime     VARCHAR(30),
    signal_1_confidence DOUBLE PRECISION,
    signal_2_fund_score DOUBLE PRECISION,
    signal_3_tft_prob   DOUBLE PRECISION,
    signal_4_tsmixer    DOUBLE PRECISION,
    signal_5_sentiment  DOUBLE PRECISION,
    signal_6_bias       DOUBLE PRECISION,
    requested_price     DOUBLE PRECISION,
    slippage_pips       DOUBLE PRECISION,
    risk_pct            DOUBLE PRECISION,
    position_size_pct   DOUBLE PRECISION,
    trading_mode        VARCHAR(10)     NOT NULL DEFAULT 'PAPER',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry
    ON trades (symbol, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status
    ON trades (status) WHERE status = 'OPEN';

CREATE TABLE IF NOT EXISTS trade_postmortems (
    id               SERIAL      PRIMARY KEY,
    trade_id         INTEGER     NOT NULL REFERENCES trades(id),
    analyzed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    failure_reason   TEXT,
    market_context   TEXT,
    analyst_conflict JSONB       NOT NULL DEFAULT '{}',
    lessons          TEXT[],
    penalty_applied  JSONB       NOT NULL DEFAULT '{}',
    groq_raw_output  TEXT,
    model_used       VARCHAR(50)
);
CREATE INDEX IF NOT EXISTS idx_postmortem_trade
    ON trade_postmortems (trade_id);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_date   DATE             PRIMARY KEY,
    equity          DOUBLE PRECISION NOT NULL,
    balance         DOUBLE PRECISION NOT NULL,
    daily_pnl       DOUBLE PRECISION NOT NULL DEFAULT 0,
    daily_pnl_pct   DOUBLE PRECISION NOT NULL DEFAULT 0,
    trades_today    INTEGER          NOT NULL DEFAULT 0,
    wins_today      INTEGER          NOT NULL DEFAULT 0,
    losses_today    INTEGER          NOT NULL DEFAULT 0,
    total_trades    INTEGER          NOT NULL DEFAULT 0,
    win_rate        DOUBLE PRECISION,
    avg_win_pips    DOUBLE PRECISION,
    avg_loss_pips   DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    daily_loss_pct  DOUBLE PRECISION NOT NULL DEFAULT 0,
    trading_halted  BOOLEAN          NOT NULL DEFAULT FALSE,
    halt_reason     TEXT,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- SECTION 7: FEATURE CACHE
-- =============================================================================

CREATE TABLE IF NOT EXISTS feature_vectors (
    time                         TIMESTAMPTZ      NOT NULL,
    pair                         VARCHAR(10)      NOT NULL,
    yield_spread_2y_bps          DOUBLE PRECISION,
    yield_spread_10y_bps         DOUBLE PRECISION,
    yield_spread_zscore_5d       DOUBLE PRECISION,
    yield_spread_zscore_20d      DOUBLE PRECISION,
    cip_deviation                DOUBLE PRECISION,
    bid_ask_imbalance_1m         DOUBLE PRECISION,
    large_order_count_5m         INTEGER,
    liquidity_disappearance      BOOLEAN,
    vpin                         DOUBLE PRECISION,
    vwap_tokyo                   DOUBLE PRECISION,
    vwap_london                  DOUBLE PRECISION,
    vwap_ny                      DOUBLE PRECISION,
    price_vs_vwap_london         DOUBLE PRECISION,
    atr_14                       DOUBLE PRECISION,
    atr_percentile_20d           DOUBLE PRECISION,
    zscore_close_20d             DOUBLE PRECISION,
    hours_until_high_impact      DOUBLE PRECISION,
    hours_since_last_high_impact DOUBLE PRECISION,
    event_surprise_score         DOUBLE PRECISION,
    nc_net_pct_oi                DOUBLE PRECISION,
    nc_net_change_weekly         DOUBLE PRECISION,
    feature_version              VARCHAR(10)      NOT NULL DEFAULT 'v1'
);
SELECT create_hypertable('feature_vectors', 'time', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days');
CREATE UNIQUE INDEX IF NOT EXISTS idx_features_unique
    ON feature_vectors (time, pair);
CREATE INDEX IF NOT EXISTS idx_features_pair_time
    ON feature_vectors (pair, time DESC);

-- =============================================================================
-- SECTION 8: SYSTEM METADATA
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_freshness (
    source        VARCHAR(50)  NOT NULL PRIMARY KEY,
    last_update   TIMESTAMPTZ  NOT NULL,
    next_expected TIMESTAMPTZ,
    status        VARCHAR(20)  NOT NULL DEFAULT 'OK'
                      CHECK (status IN ('OK','STALE','MISSING','ERROR')),
    record_count  INTEGER,
    error_message TEXT,
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

INSERT INTO data_freshness (source, last_update, status) VALUES
    ('ohlcv_ctrader',       NOW() - INTERVAL '1 year', 'MISSING'),
    ('bond_yields_manual',  NOW() - INTERVAL '1 year', 'MISSING'),
    ('bond_yields_scraped', NOW() - INTERVAL '1 year', 'MISSING'),
    ('macro_indicators',    NOW() - INTERVAL '1 year', 'MISSING'),
    ('cot_reports',         NOW() - INTERVAL '1 year', 'MISSING'),
    ('economic_events',     NOW() - INTERVAL '1 year', 'MISSING'),
    ('news_articles',       NOW() - INTERVAL '1 year', 'MISSING'),
    ('sentiment_daily',     NOW() - INTERVAL '1 year', 'MISSING'),
    ('sentiment_weekly',    NOW() - INTERVAL '1 year', 'MISSING'),
    ('feature_vectors',     NOW() - INTERVAL '1 year', 'MISSING')
ON CONFLICT (source) DO NOTHING;

CREATE TABLE IF NOT EXISTS apex_weights (
    feature_name     VARCHAR(100)     NOT NULL PRIMARY KEY,
    weight           DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    penalty_applied  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_adjusted    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    adjustment_count INTEGER          NOT NULL DEFAULT 0,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id            SERIAL      PRIMARY KEY,
    analyst_id    SMALLINT    NOT NULL,
    model_name    VARCHAR(100) NOT NULL,
    version       VARCHAR(50) NOT NULL,
    mlflow_run_id VARCHAR(100),
    artifact_path TEXT,
    metrics       JSONB       NOT NULL DEFAULT '{}',
    is_active     BOOLEAN     NOT NULL DEFAULT FALSE,
    trained_at    TIMESTAMPTZ NOT NULL,
    deployed_at   TIMESTAMPTZ,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_model_versions_analyst
    ON model_versions (analyst_id, is_active);

-- =============================================================================
-- SECTION 9: CONTINUOUS AGGREGATES
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    FIRST(open, time)          AS open,
    MAX(high)                  AS high,
    MIN(low)                   AS low,
    LAST(close, time)          AS close,
    SUM(volume)                AS volume,
    COUNT(*)                   AS candle_count
FROM ohlcv
WHERE timeframe = 'M15'
GROUP BY day, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ohlcv_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE
);

-- =============================================================================
-- SECTION 10: UTILITY FUNCTIONS
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE FUNCTION get_latest_yield(p_currency VARCHAR, p_tenor VARCHAR)
RETURNS DOUBLE PRECISION AS $$
    SELECT yield_pct FROM bond_yields
    WHERE currency = p_currency AND tenor = p_tenor
    ORDER BY time DESC LIMIT 1;
$$ LANGUAGE SQL STABLE;

CREATE OR REPLACE FUNCTION compute_imbalance(bids JSONB, asks JSONB, levels INTEGER DEFAULT 5)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    bid_vol DOUBLE PRECISION;
    ask_vol DOUBLE PRECISION;
BEGIN
    SELECT COALESCE(SUM((elem->>'volume')::DOUBLE PRECISION), 0)
    INTO bid_vol
    FROM jsonb_array_elements(bids) WITH ORDINALITY AS t(elem, idx)
    WHERE idx <= levels;

    SELECT COALESCE(SUM((elem->>'volume')::DOUBLE PRECISION), 0)
    INTO ask_vol
    FROM jsonb_array_elements(asks) WITH ORDINALITY AS t(elem, idx)
    WHERE idx <= levels;

    IF (bid_vol + ask_vol) = 0 THEN RETURN 0; END IF;
    RETURN (bid_vol - ask_vol) / (bid_vol + ask_vol);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- =============================================================================
-- GRANTS
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'g10bot') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO g10bot;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO g10bot;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO g10bot;
    END IF;
END $$;

-- =============================================================================
-- DONE — 19 tables created
-- =============================================================================
