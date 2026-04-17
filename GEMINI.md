# GEMINI.md - Regime-Trader Project Context

## Project Overview
**regime-trader** is an algorithmic trading system for eToro that utilizes Hidden Markov Models (HMM) to detect market volatility regimes. It prioritizes capital preservation through a rigorous risk management layer that has absolute veto power over all trading signals.

### Core Philosophy
- **Risk Management > Signal Generation:** Survival is prioritized over profit maximization.
- **Regime-Driven Allocation:** The system adapts its exposure based on the current volatility state (e.g., CRASH, BEAR, BULL, EUPHORIA) rather than predicting price direction.
- **Causal Inference:** Uses the HMM Forward algorithm (not Viterbi) to ensure zero look-ahead bias in real-time execution.

### Tech Stack
- **Language:** Python 3.10+
- **ML/Stats:** `hmmlearn`, `scikit-learn`, `numpy`, `pandas`, `ta` (Technical Analysis Library).
- **Broker Integration:** eToro REST API (Custom Client).
- **UI/Monitoring:** `Rich` (Terminal Dashboard), `PySide6` (Desktop GUI), `YAML` for configuration.

---

## Architecture & Data Flow
1. **Data Acquisition:** `EToroClient` fetches daily OHLCV candles and live prices.
2. **Feature Engineering:** `data/feature_engineering.py` builds 14 normalized indicators (z-score 252d).
3. **HMM Engine:** `core/hmm_engine.py` classifies the regime using Forward probabilities. Includes stability filters (3-bar confirmation) and flicker detection.
4. **Signal Generation:** `core/signal_generator.py` combines HMM state with technical filters (MACD + Donchian) via `StrategyOrchestrator`.
5. **Risk Management:** `core/risk_manager.py` executes 16 sequential checks (Circuit Breakers, Exposure, Correlation, Stop Loss, etc.).
6. **Execution:** `broker/order_executor.py` submits orders. Note: eToro stop-loss adjustments require a Close + Reopen cycle.
7. **Persistence:** `state_snapshot.json` stores portfolio state and metadata for crash recovery.

---

## Development & Operations

### Key Commands
| Task | Command |
| :--- | :--- |
| **Install** | `pip install -r requirements.txt` |
| **Run (Live/Demo)** | `python main.py` |
| **Dry Run** | `python main.py --dry-run` (Full pipeline, no real orders) |
| **Backtest** | `python main.py --backtest` (Walk-forward IS/OOS) |
| **Train HMM** | `python main.py --train-only` |
| **Stress Test** | `python main.py --stress-test` (Crash/Gap/Regime Scramble) |
| **Tests** | `python -m pytest tests/ -v` |
| **Benchmarks** | `python main.py --compare` (vs Buy&Hold, SMA200) |

### Environment Variables
Required for eToro API access (store in `.env`):
- `ETORO_API_KEY`
- `ETORO_USER_KEY`

### Configuration
Central settings are in `config/settings.yaml`. 
- **Broker:** Set `environment: "demo"` for testing.
- **Risk:** Calibrated to a base equity of **$546.14**. Update `initial_equity` if capital changes.

---

## Engineering Standards & Constraints

### 1. Risk Veto Chain
The `RiskManager.validate_signal()` method runs exactly 16 checks in order. **Never bypass these checks.** Any new risk rule must be added to this chain.

### 2. No Look-Ahead Bias
All HMM predictions must use `predict_regime_filtered()`. This implementation uses the Forward algorithm in log-space. Do not use Viterbi for live trading as it re-evaluates the entire history, introducing non-causal bias.

### 3. eToro Constraints
- **Stop Loss:** Mandatory for every order. Missing `stopLossRate` is a hard rejection.
- **Leverage:** Fixed at `1.0` for all automated trades.
- **Polling:** State is reconciled every 30 seconds (no WebSockets).
- **Live Confirmation:** Real-money trading requires a manual "CONFIRMO" input in the console. Do not automate this confirmation.

### 4. Persistence & Recovery
- The system must be able to recover from `state_snapshot.json`.
- Positions are reconciled against the API on startup to detect external changes.

### 5. Logging & Alerts
- Logs are rotated and categorized (main, trades, alerts, regime).
- AlertManager rate-limits notifications (default 15 mins) to prevent spam.

---

## Project Structure
- `backtest/`: Walk-forward testing and performance metrics.
- `broker/`: eToro API client, order execution, and position tracking.
- `core/`: HMM engine, strategy orchestration, and risk management.
- `data/`: Market data fetching and feature engineering.
- `monitoring/`: Dashboards (Rich/Qt), logging, and alerting systems.
- `tests/`: Comprehensive test suite including mathematical look-ahead verification.
