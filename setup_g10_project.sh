#!/bin/bash
# =============================================================================
# G10 Dominance Architecture — Project Structure Setup Script
# =============================================================================
# Run this ONCE in your GitHub workspace root directory.
# It creates every folder and placeholder file the bot needs.
#
# Usage:
#   chmod +x setup_g10_project.sh
#   ./setup_g10_project.sh
# =============================================================================

set -e  # Exit on any error

BASE="g10_dominance"
echo "=============================================="
echo "  G10 Dominance — Building project structure"
echo "=============================================="

# ── Create all directories ────────────────────────────────────────────────────
echo "Creating directories..."

mkdir -p $BASE/config
mkdir -p $BASE/data_pipeline/connectors
mkdir -p $BASE/data_pipeline/ingestion
mkdir -p $BASE/data_pipeline/storage
mkdir -p $BASE/features
mkdir -p $BASE/models/analyst1_lightgbm
mkdir -p $BASE/models/analyst2_endo_exo
mkdir -p $BASE/models/analyst3_tft
mkdir -p $BASE/models/analyst4_tsmixer
mkdir -p $BASE/models/analyst5_daily_groq
mkdir -p $BASE/models/analyst6_weekly_opus
mkdir -p $BASE/execution/rl_agent
mkdir -p $BASE/intelligence/recursive_evolution
mkdir -p $BASE/intelligence/monitoring
mkdir -p $BASE/web_portal/templates
mkdir -p $BASE/web_portal/static/css
mkdir -p $BASE/web_portal/static/js
mkdir -p $BASE/scripts/sql
mkdir -p $BASE/tests
mkdir -p $BASE/notebooks
mkdir -p $BASE/monitoring/grafana/datasources
mkdir -p $BASE/monitoring/grafana/dashboards
mkdir -p $BASE/logs

# ── Create __init__.py for all Python packages ────────────────────────────────
echo "Creating __init__.py files..."

touch $BASE/__init__.py
touch $BASE/config/__init__.py
touch $BASE/data_pipeline/__init__.py
touch $BASE/data_pipeline/connectors/__init__.py
touch $BASE/data_pipeline/ingestion/__init__.py
touch $BASE/data_pipeline/storage/__init__.py
touch $BASE/features/__init__.py
touch $BASE/models/__init__.py
touch $BASE/models/analyst1_lightgbm/__init__.py
touch $BASE/models/analyst2_endo_exo/__init__.py
touch $BASE/models/analyst3_tft/__init__.py
touch $BASE/models/analyst4_tsmixer/__init__.py
touch $BASE/models/analyst5_daily_groq/__init__.py
touch $BASE/models/analyst6_weekly_opus/__init__.py
touch $BASE/execution/__init__.py
touch $BASE/execution/rl_agent/__init__.py
touch $BASE/intelligence/__init__.py
touch $BASE/intelligence/recursive_evolution/__init__.py
touch $BASE/intelligence/monitoring/__init__.py
touch $BASE/web_portal/__init__.py
touch $BASE/tests/__init__.py

# ── Create placeholder Python files (all future modules) ─────────────────────
echo "Creating placeholder source files..."

# config/
touch $BASE/config/settings.py
touch $BASE/config/logging_setup.py
touch $BASE/config/ctrader_config.yaml
touch $BASE/config/groq_config.yaml
touch $BASE/config/anthropic_config.yaml
touch $BASE/config/model_hyperparameters.yaml

# data_pipeline/connectors/
touch $BASE/data_pipeline/connectors/ctrader_fix.py
touch $BASE/data_pipeline/connectors/ctrader_rest_api.py
touch $BASE/data_pipeline/connectors/manual_data_api.py

# data_pipeline/ingestion/
touch $BASE/data_pipeline/ingestion/cot_ingest.py
touch $BASE/data_pipeline/ingestion/yield_scraper.py
touch $BASE/data_pipeline/ingestion/calendar_scraper.py
touch $BASE/data_pipeline/ingestion/rss_news_scraper.py
touch $BASE/data_pipeline/ingestion/data_entry_service.py

# data_pipeline/storage/
touch $BASE/data_pipeline/storage/timeseries_db.py

# features/
touch $BASE/features/yield_features.py
touch $BASE/features/liquidity_features.py
touch $BASE/features/statistical_features.py
touch $BASE/features/vwap_features.py
touch $BASE/features/orderflow_features.py
touch $BASE/features/event_features.py

# models/
touch $BASE/models/analyst1_lightgbm/regime_classifier.py
touch $BASE/models/analyst1_lightgbm/trainer.py
touch $BASE/models/analyst2_endo_exo/score_calculator.py
touch $BASE/models/analyst3_tft/tft_model.py
touch $BASE/models/analyst3_tft/inference.py
touch $BASE/models/analyst4_tsmixer/patch_tsmixer_tcn.py
touch $BASE/models/analyst4_tsmixer/directional_filter.py
touch $BASE/models/analyst5_daily_groq/groq_sentiment.py
touch $BASE/models/analyst6_weekly_opus/multi_agent_parser.py
touch $BASE/models/analyst6_weekly_opus/opus_deep_analysis.py

# execution/
touch $BASE/execution/signal_aggregator.py
touch $BASE/execution/risk_manager.py
touch $BASE/execution/order_executor.py
touch $BASE/execution/rl_agent/ppo_trainer.py
touch $BASE/execution/rl_agent/environment.py
touch $BASE/execution/rl_agent/reward_functions.py

# intelligence/
touch $BASE/intelligence/recursive_evolution/postmortem_analyzer.py
touch $BASE/intelligence/recursive_evolution/apex_framework.py
touch $BASE/intelligence/recursive_evolution/retraining_loop.py
touch $BASE/intelligence/monitoring/performance_tracker.py
touch $BASE/intelligence/monitoring/drift_detector.py
touch $BASE/intelligence/monitoring/alert_manager.py

# web_portal/
touch $BASE/web_portal/app.py
touch $BASE/web_portal/templates/base.html
touch $BASE/web_portal/templates/dashboard.html
touch $BASE/web_portal/templates/data_entry.html
touch $BASE/web_portal/templates/bulk_import.html
touch $BASE/web_portal/templates/charts.html
touch $BASE/web_portal/static/css/styles.css
touch $BASE/web_portal/static/js/main.js

# scripts/
touch $BASE/scripts/sql/init_schema.sql
touch $BASE/scripts/import_economic_data.py
touch $BASE/scripts/validate_data.py
touch $BASE/scripts/check_missing_data.py
touch $BASE/scripts/daily_reminder.py

# tests/
touch $BASE/tests/test_data_pipeline.py
touch $BASE/tests/test_features.py
touch $BASE/tests/test_models.py
touch $BASE/tests/test_execution.py
touch $BASE/tests/test_fix_connector.py
touch $BASE/tests/test_risk_manager.py

# notebooks/
touch $BASE/notebooks/01_data_exploration.ipynb
touch $BASE/notebooks/02_feature_analysis.ipynb
touch $BASE/notebooks/03_model_training.ipynb
touch $BASE/notebooks/04_backtest_results.ipynb

# monitoring/
touch $BASE/monitoring/prometheus.yml
touch $BASE/monitoring/alert_rules.yml
touch $BASE/monitoring/alertmanager.yml

# root files
touch $BASE/main.py
touch $BASE/.env.example
touch $BASE/.gitignore
touch $BASE/README.md
touch $BASE/Dockerfile
touch $BASE/Dockerfile.portal

# ── Create .gitignore ─────────────────────────────────────────────────────────
cat > $BASE/.gitignore << 'EOF'
# Secrets — NEVER commit
.env
*.key
*.pem
secrets/

# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# ML artifacts (use MLflow)
models/*/saved_models/
models/*/checkpoints/
*.pkl
*.joblib
*.pt
*.pth
*.onnx
mlruns/
mlartifacts/

# Jupyter
.ipynb_checkpoints/

# Logs
logs/
*.log
*.log.gz

# Temp / cache
.cache/
.pytest_cache/
.mypy_cache/
tmp/
temp/
/tmp/g10_fix.cfg

# IDE
.vscode/
.idea/
*.swp
.DS_Store
EOF

# ── Create .env.example ───────────────────────────────────────────────────────
cat > $BASE/.env.example << 'EOF'
# =============================================================================
# G10 Dominance — Environment Variables Template
# Copy to .env and fill in your real values. Never commit .env to git.
# =============================================================================

# cTrader FIX API
CTRADER_FIX_HOST=h51.p.ctrader.com
CTRADER_FIX_PORT=5201
CTRADER_SENDER_COMP_ID=your_sender_id
CTRADER_TARGET_COMP_ID=CSERVER
CTRADER_FIX_USERNAME=your_username
CTRADER_FIX_PASSWORD=your_password

# cTrader REST API
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCOUNT_ID=your_account_id
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_REFRESH_TOKEN=your_refresh_token

# LLM APIs
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key

# TimescaleDB
DB_HOST=localhost
DB_PORT=5432
DB_NAME=g10bot
DB_USER=g10bot
DB_PASSWORD=your_db_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# InfluxDB
INFLUX_URL=http://localhost:8086
INFLUX_PASSWORD=your_influx_password
INFLUX_ADMIN_TOKEN=your_influx_token
INFLUX_ORG=g10_trading
INFLUX_BUCKET=system_metrics

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Flask Portal
FLASK_SECRET_KEY=generate_a_long_random_string_here
FLASK_DEBUG=false
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_portal_password

# Grafana
GRAFANA_ADMIN_PASSWORD=admin

# Telegram (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# System
TRADING_MODE=PAPER
LOG_LEVEL=INFO
EOF

# ── Create README.md ──────────────────────────────────────────────────────────
cat > $BASE/README.md << 'EOF'
# G10 Dominance Architecture

Institutional-grade autonomous G10 forex trading system.

## Stack
- **Execution**: cTrader FIX API (orders + L2 depth)
- **Data**: cTrader REST API + scrapers + manual entry
- **Storage**: TimescaleDB + Redis + InfluxDB
- **ML**: LightGBM + TFT + PatchTSMixer + TCN
- **Sentiment**: Groq (daily) + Claude Opus (weekly)
- **RL**: PPO position sizer
- **Monitoring**: Grafana + Prometheus + Telegram

## Quick Start
```bash
cp .env.example .env        # fill in your credentials
docker-compose up -d        # start infrastructure
python main.py              # start the bot (PAPER mode)
```

## Build Phases
- Phase 1: Infrastructure + DB schema + config  ✅
- Phase 2: Data pipeline (connectors + scrapers)
- Phase 3: Feature engineering
- Phase 4: 6 Analyst ML models
- Phase 5: Execution engine
- Phase 6: Intelligence layer
- Phase 7: Testing + deployment
EOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Structure created successfully!"
echo "=============================================="
echo ""
echo "Directory count: $(find $BASE -type d | wc -l)"
echo "File count:      $(find $BASE -type f | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Copy the Phase 1 files (main.py, settings.py, etc.) into their folders"
echo "  2. git add . && git commit -m 'Phase 1: project scaffold'"
echo "  3. Apply DB schema: docker exec -i <timescaledb_container> psql -U g10bot -d g10bot < g10_dominance/scripts/sql/init_schema.sql"
echo ""