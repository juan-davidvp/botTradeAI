# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_risk_stress.py -v

# Run a single test by name
python -m pytest tests/test_risk_stress.py::TestCircuitBreaker::test_daily_halt_triggered_at_3pct -v

# Dry run (full pipeline, no real orders)
python main.py --dry-run

# Walk-forward backtest
python main.py --backtest

# Train HMM only
python main.py --train-only

# Stress tests (crash injection, gap risk, regime scramble)
python main.py --stress-test

# Compare vs benchmarks
python main.py --compare
```

**Required environment variables** (never hardcoded):
```
ETORO_API_KEY
ETORO_USER_KEY
```

Tests inject fake values via `tests/conftest.py` — no real credentials needed to run the test suite.

## Architecture

The system is a regime-driven portfolio allocator, not a signal-first bot. The HMM classifies volatility regime → regime determines how much capital to expose → technical filters (MACD + Donchian) determine entry quality within that regime. The `RiskManager` has absolute veto power over every signal, independent of all other components.

### Data flow (per 30-second poll cycle)

```
EToroClient.get_instrument_rates()      # live prices
EToroClient.get_historical_candles()   # daily OHLCV
    │
data/feature_engineering.build_features()   # 14 features, rolling z-score 252d
    │
core/hmm_engine.HMMEngine.predict_regime_filtered()   # Forward algorithm only
    │                                                  # (no Viterbi = no look-ahead bias)
core/signal_generator.SignalGenerator.generate_all()
    │   └── core/regime_strategies.StrategyOrchestrator
    │           ├── MACDFilter
    │           └── DonchianBreakoutFilter
    │
core/risk_manager.RiskManager.validate_signal()   # 16 sequential checks, any fails → reject
    │
broker/order_executor.OrderExecutor.submit_order()   # dry_run=True skips API call
    │
broker/position_tracker.PositionTracker   # 30s polling, reconciles with state_snapshot.json
```

### Key architectural constraints

**HMM — no look-ahead bias**: `predict_regime_filtered()` uses the Forward algorithm in log-space with a cached `_log_alpha`. The regime at time T must be identical whether computed with `data[0:T]` or `data[0:T+100]`. `test_look_ahead.py` enforces this mathematically.

**RiskManager veto chain**: `validate_signal()` in `core/risk_manager.py` runs 16 checks in order. Each check can either reject (returns `RiskDecision(approved=False)`) or modify the signal (adjusts `position_size_usd`). The checks are numbered 0–15 in the source — order matters (lock file → CB → flicker → leverage → stop → duplicate → spread → daily trades → concurrent → cash → exposure → 20% cap → $100 min → gap risk → correlation → quality).

**eToro constraints that affect design**:
- No stop-loss modification via PATCH — `adjust_stop_loss()` in `order_executor.py` closes + reopens the position with the new rate.
- No WebSocket — all state comes from 30s polling via `PositionTracker`.
- `leverage` is always forced to `1` in every order body regardless of signal value.
- `stopLossRate` is mandatory in every `open_market_order` call; a missing stop is a hard reject at risk check step 5.

**Circuit Breaker thresholds** (calibrated to $546.14 base equity, defined in `core/risk_manager.py`):
- Daily DD > 2% ($10.92) → `REDUCE_50` (halve position sizes)
- Daily DD > 3% ($16.38) → `HALT` (no new trades)
- Weekly DD > 5% ($27.31) → `REDUCE_50`
- Weekly DD > 7% ($38.23) → `HALT`
- Peak DD > 10% ($54.61) → `LOCKED` (creates `trading_halted.lock` file, requires manual intervention)

**BIC model selection**: `HMMEngine.train()` trains one `GaussianHMM` per candidate in `n_candidates` (default [3,4,5,6,7]) and picks the model with the lowest BIC. Regime labels are assigned deterministically by ascending mean volatility of the hidden states.

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `core/hmm_engine.py` | Trains GaussianHMM, runs Forward algorithm, manages stability filter and flicker detection, pickles model |
| `core/regime_strategies.py` | `Signal` dataclass, `MACDFilter`, `DonchianBreakoutFilter`, per-regime strategy classes, `StrategyOrchestrator` |
| `core/signal_generator.py` | Combines HMM state + strategy to produce `Signal`; applies quality filter before handing off to RiskManager |
| `core/risk_manager.py` | `CircuitBreaker`, `RiskManager.validate_signal()` (16 checks), `PortfolioState` dataclass |
| `broker/etoro_client.py` | All HTTP calls to eToro REST API v1; exponential backoff [1,2,4]s; raises `EToroAPIError` on 401/403 |
| `broker/order_executor.py` | Wraps `EToroClient` with `dry_run` flag, stop-adjustment logic (close+reopen), UUID-based trade tracing |
| `broker/position_tracker.py` | 30s scheduler, builds `PortfolioState`, saves/loads `state_snapshot.json` for crash recovery |
| `data/feature_engineering.py` | Pure functions: 14 OHLCV features, all normalized with 252-bar rolling z-score |
| `data/market_data.py` | Caches candles, handles 3 eToro response formats, returns spread=1.0 on API error to force rejection |
| `monitoring/logger.py` | 4 rotating JSON log files (main/trades/alerts/regime), typed helpers `log_trade()`, `log_alert()`, etc. |
| `monitoring/alerts.py` | `AlertManager`: 10 trigger types, rate-limited 15 min, optional email (smtplib) + webhook (Discord/Slack) |
| `monitoring/dashboard.py` | Rich terminal dashboard, `render()` for one-shot print, `run_live()` for 5s refresh loop |

### State persistence

`state_snapshot.json` (project root) is written by `PositionTracker.save_snapshot()` on shutdown (SIGINT/SIGTERM). On restart, `load_snapshot()` returns `True/False` (not the dict). Internal tracker state (`_peak_equity`, `_entry_regimes`, `_last_order_ts`) is restored from the file; positions are reconciled against the live API to detect ghosts (closed externally) and new positions (opened externally).

### Logging routing

Log records are routed by `log_type` extra field:
- No `log_type` → `main.log` (via `MainFilter`)
- `log_type="trade"` → `trades.log`
- `log_type="alert"` → `alerts.log`
- `log_type="regime"` → `regime.log`

### Capital calibration

All monetary constants reference `$546.14` (real account as of 2026-04-14). If capital changes, update `risk.initial_equity` in `config/settings.yaml` — the code reads it at startup. Do not hardcode dollar amounts in new code; derive them as `equity * pct`.

### Real-money safety

`EToroClient.__init__` with `environment="real"` calls `_confirm_live_trading()` which blocks on console input requiring the string `CONFIRMO`. This is intentional — never bypass it in production code. The `trading_halted.lock` file must be manually deleted to resume after a peak-DD lockout.
