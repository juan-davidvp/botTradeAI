"""
backtest/backtester.py
Fase 4 — Motor de Backtesting Walk-Forward basado en Asignación.

Capital inicial : $546.14
Comisión        : $1.00 por operación (eToro - totalExternalFees)
Slippage        : 0.05% por rebalanceo
Fill delay      : 1 barra (señal en N → ejecución en apertura de N+1)
IS / OOS        : 252 / 126 días de trading | step = 126 días
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import build_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes calibradas al portafolio real
# ---------------------------------------------------------------------------
INITIAL_CAPITAL     = 546.14
COMMISSION_USD      = 1.00
SLIPPAGE_PCT        = 0.0005
REBALANCE_THRESHOLD = 0.10
IS_WINDOW           = 252
OOS_WINDOW          = 126
STEP_SIZE           = 126
MAX_POSITION_USD    = 109.23
MAX_POSITION_PCT    = 0.20
FILL_DELAY          = 1


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    action: str                 # "BUY" | "SELL" | "REBALANCE"
    old_allocation: float
    new_allocation: float
    price: float
    equity_before: float
    equity_after: float
    commission: float
    slippage: float
    regime: str
    regime_prob: float
    window_id: int


@dataclass
class BarRecord:
    timestamp: datetime
    equity: float
    regime: str
    regime_prob: float
    allocations: Dict[str, float]
    returns: Dict[str, float]
    is_oos: bool
    window_id: int


@dataclass
class WalkForwardResult:
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    n_regimes: int
    bic_score: float
    equity_curve: List[BarRecord]
    trades: List[TradeRecord]
    oos_return: float
    oos_monthly_return: float


@dataclass
class BacktestResult:
    windows: List[WalkForwardResult]
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    regime_history: pd.DataFrame
    final_equity: float
    total_return_pct: float
    passes_validation: bool       # >= 4.66% mensual compuesto en OOS


# ---------------------------------------------------------------------------
# Motor Walk-Forward
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Backtester de asignación walk-forward.

    No rastrea entradas/salidas individuales.
    En cada barra OOS:
      1. Calcula features con datos hasta esa barra (sin look-ahead).
      2. Corre el algoritmo Forward del HMM.
      3. Determina asignación objetivo por régimen de volatilidad.
      4. Si |delta asignación| > 10% → rebalanceo (con fill delay de 1 barra).
      5. Mark-to-market: equity[t] = equity[t-1] × Π(1 + alloc[i] × ret[i])
    """

    def __init__(
        self,
        is_window: int   = IS_WINDOW,
        oos_window: int  = OOS_WINDOW,
        step_size: int   = STEP_SIZE,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float      = COMMISSION_USD,
        slippage_pct: float    = SLIPPAGE_PCT,
        rebalance_threshold: float = REBALANCE_THRESHOLD,
        n_candidates: List[int] = None,
    ):
        self.is_window   = is_window
        self.oos_window  = oos_window
        self.step_size   = step_size
        self.capital     = initial_capital
        self.commission  = commission
        self.slippage    = slippage_pct
        self.rebal_thr   = rebalance_threshold
        self.n_candidates = n_candidates or [3, 4, 5, 6, 7]

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        symbols: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        Ejecuta el backtesting walk-forward completo.

        Parámetros
        ----------
        bars_by_symbol : dict {symbol: DataFrame OHLCV con índice DatetimeIndex}
        symbols        : lista de símbolos a usar (default: todas las claves)
        """
        if symbols is None:
            symbols = list(bars_by_symbol.keys())

        # Alinear todos los DataFrames al índice común
        common_index = self._common_index(bars_by_symbol, symbols)
        n_total = len(common_index)

        if n_total < self.is_window + self.oos_window:
            raise ValueError(
                f"Datos insuficientes: {n_total} barras. "
                f"Mínimo requerido: {self.is_window + self.oos_window}."
            )

        all_windows : List[WalkForwardResult] = []
        all_equity  : List[BarRecord]         = []
        all_trades  : List[TradeRecord]        = []

        equity = self.capital
        window_id = 0
        pos = 0

        while pos + self.is_window + self.oos_window <= n_total:
            is_idx  = common_index[pos : pos + self.is_window]
            oos_idx = common_index[pos + self.is_window : pos + self.is_window + self.oos_window]

            logger.info(
                "Ventana %d — IS: %s→%s | OOS: %s→%s",
                window_id, is_idx[0].date(), is_idx[-1].date(),
                oos_idx[0].date(), oos_idx[-1].date(),
            )

            result, equity = self._run_window(
                window_id, bars_by_symbol, symbols,
                is_idx, oos_idx, equity,
            )

            all_windows.append(result)
            all_equity.extend(result.equity_curve)
            all_trades.extend(result.trades)

            pos += self.step_size
            window_id += 1

        eq_df      = self._equity_df(all_equity)
        trade_df   = self._trade_df(all_trades)
        regime_df  = self._regime_df(all_equity)

        total_ret  = (equity - self.capital) / self.capital * 100
        passes     = self._validate_oos(all_windows)

        logger.info(
            "Backtest completo: %d ventanas | equity final $%.2f | retorno %.2f%% | validación=%s",
            len(all_windows), equity, total_ret, passes,
        )

        return BacktestResult(
            windows=all_windows,
            equity_curve=eq_df,
            trade_log=trade_df,
            regime_history=regime_df,
            final_equity=equity,
            total_return_pct=total_ret,
            passes_validation=passes,
        )

    # ------------------------------------------------------------------
    # Una ventana walk-forward
    # ------------------------------------------------------------------

    def _run_window(
        self,
        window_id: int,
        bars_by_symbol: Dict[str, pd.DataFrame],
        symbols: List[str],
        is_idx: pd.DatetimeIndex,
        oos_idx: pd.DatetimeIndex,
        equity_start: float,
    ) -> Tuple[WalkForwardResult, float]:

        # 1. Entrenamiento IS -------------------------------------------------
        is_bars = {s: bars_by_symbol[s].loc[is_idx[0]:is_idx[-1]] for s in symbols}
        primary_sym = symbols[0]

        features_is = build_features(is_bars[primary_sym])
        hmm = HMMEngine(n_candidates=self.n_candidates, n_init=10)
        hmm.train(features_is.values)

        orchestrator = StrategyOrchestrator(equity=equity_start)

        # 2. Simulación OOS ---------------------------------------------------
        equity = equity_start
        allocations: Dict[str, float] = {s: 0.0 for s in symbols}
        pending_rebalance: Optional[Dict[str, float]] = None
        bar_records : List[BarRecord]   = []
        trade_records: List[TradeRecord] = []

        for i, ts in enumerate(oos_idx):
            # Datos disponibles hasta esta barra (IS completo + OOS hasta i)
            full_end = ts
            oos_bars = {
                s: bars_by_symbol[s].loc[:full_end].tail(self.is_window + i + 1)
                for s in symbols
            }
            features_now = build_features(oos_bars[primary_sym])
            if features_now.empty:
                continue

            # Forward step (sin look-ahead)
            regime_state = hmm.predict_regime_filtered(features_now.values[-1])

            # Retornos de la barra actual
            bar_returns: Dict[str, float] = {}
            for s in symbols:
                df = oos_bars[s]
                if len(df) >= 2:
                    bar_returns[s] = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
                else:
                    bar_returns[s] = 0.0

            # Ejecutar rebalanceo pendiente (fill delay = 1 barra)
            if pending_rebalance is not None:
                equity, trade_records = self._execute_rebalance(
                    pending_rebalance, allocations, equity, ts,
                    bar_returns, regime_state, window_id, trade_records,
                )
                allocations = dict(pending_rebalance)
                pending_rebalance = None

            # Mark-to-market
            equity = self._mark_to_market(equity, allocations, bar_returns)

            # Calcular asignación objetivo
            strategy   = orchestrator.get_strategy(regime_state, hmm.regime_infos)
            target_alloc = self._target_allocations(
                symbols, oos_bars, regime_state, strategy, equity
            )

            # Decidir si rebalancear
            if self._needs_rebalance(allocations, target_alloc):
                pending_rebalance = target_alloc

            bar_records.append(BarRecord(
                timestamp=ts,
                equity=equity,
                regime=regime_state.label,
                regime_prob=regime_state.probability,
                allocations=dict(allocations),
                returns=bar_returns,
                is_oos=True,
                window_id=window_id,
            ))

        oos_ret = (equity - equity_start) / equity_start
        n_months = self.oos_window / 21
        monthly   = (1 + oos_ret) ** (1 / n_months) - 1 if n_months > 0 else 0.0

        result = WalkForwardResult(
            window_id=window_id,
            is_start=is_idx[0],
            is_end=is_idx[-1],
            oos_start=oos_idx[0],
            oos_end=oos_idx[-1],
            n_regimes=hmm.n_regimes,
            bic_score=hmm.bic_score,
            equity_curve=bar_records,
            trades=trade_records,
            oos_return=oos_ret,
            oos_monthly_return=monthly,
        )
        return result, equity

    # ------------------------------------------------------------------
    # Asignación objetivo
    # ------------------------------------------------------------------

    def _target_allocations(
        self,
        symbols: List[str],
        oos_bars: Dict[str, pd.DataFrame],
        regime_state,
        strategy,
        equity: float,
    ) -> Dict[str, float]:
        """Calcula la fracción objetivo para cada símbolo."""
        target: Dict[str, float] = {}
        for s in symbols:
            signal = strategy.generate_signal(s, oos_bars[s], regime_state)
            if signal is None:
                target[s] = 0.0
            else:
                # Limitar al máximo por posición
                max_pct = min(signal.position_size_pct / len(symbols), MAX_POSITION_PCT)
                target[s] = max_pct
        return target

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _mark_to_market(
        self,
        equity: float,
        allocations: Dict[str, float],
        bar_returns: Dict[str, float],
    ) -> float:
        """
        equity[t] = equity[t-1] × Π(1 + alloc[i] × ret[i])
        El cash libre (1 - sum(alloc)) no genera retorno.
        """
        portfolio_return = sum(
            allocations.get(s, 0.0) * bar_returns.get(s, 0.0)
            for s in bar_returns
        )
        return equity * (1 + portfolio_return)

    # ------------------------------------------------------------------
    # Rebalanceo
    # ------------------------------------------------------------------

    def _needs_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
    ) -> bool:
        for s in set(list(current.keys()) + list(target.keys())):
            if abs(current.get(s, 0.0) - target.get(s, 0.0)) > self.rebal_thr:
                return True
        return False

    def _execute_rebalance(
        self,
        target: Dict[str, float],
        current: Dict[str, float],
        equity: float,
        ts: datetime,
        bar_returns: Dict[str, float],
        regime_state,
        window_id: int,
        records: List[TradeRecord],
    ) -> Tuple[float, List[TradeRecord]]:
        """Aplica comisión + slippage por cada posición que cambia."""
        for s in target:
            delta = abs(target.get(s, 0.0) - current.get(s, 0.0))
            if delta < 0.001:
                continue

            trade_value = equity * delta
            slip_cost   = trade_value * self.slippage
            total_cost  = self.commission + slip_cost
            equity     -= total_cost

            records.append(TradeRecord(
                timestamp=ts,
                symbol=s,
                action="REBALANCE",
                old_allocation=current.get(s, 0.0),
                new_allocation=target.get(s, 0.0),
                price=float(bar_returns.get(s, 0.0)),
                equity_before=equity + total_cost,
                equity_after=equity,
                commission=self.commission,
                slippage=slip_cost,
                regime=regime_state.label,
                regime_prob=regime_state.probability,
                window_id=window_id,
            ))

        return equity, records

    # ------------------------------------------------------------------
    # Validación OOS
    # ------------------------------------------------------------------

    def _validate_oos(self, windows: List[WalkForwardResult]) -> bool:
        """
        Retorna True si el promedio del retorno mensual compuesto en OOS
        es >= 4.66% (camino al +20% en 120 días desde $546.14).
        """
        monthly_returns = [w.oos_monthly_return for w in windows if w.oos_monthly_return != 0]
        if not monthly_returns:
            return False
        avg_monthly = float(np.mean(monthly_returns))
        logger.info("Retorno mensual OOS promedio: %.2f%% (objetivo >= 4.66%%)", avg_monthly * 100)
        return avg_monthly >= 0.0466

    # ------------------------------------------------------------------
    # Helpers de índice y serialización
    # ------------------------------------------------------------------

    def _common_index(
        self, bars_by_symbol: Dict[str, pd.DataFrame], symbols: List[str]
    ) -> pd.DatetimeIndex:
        idx = bars_by_symbol[symbols[0]].index
        for s in symbols[1:]:
            idx = idx.intersection(bars_by_symbol[s].index)
        return idx.sort_values()

    def _equity_df(self, records: List[BarRecord]) -> pd.DataFrame:
        rows = [{
            "timestamp":   r.timestamp,
            "equity":      r.equity,
            "regime":      r.regime,
            "regime_prob": r.regime_prob,
            "window_id":   r.window_id,
            **{f"alloc_{s}": v for s, v in r.allocations.items()},
        } for r in records]
        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()

    def _trade_df(self, records: List[TradeRecord]) -> pd.DataFrame:
        rows = [{
            "timestamp":       r.timestamp,
            "symbol":          r.symbol,
            "action":          r.action,
            "old_allocation":  r.old_allocation,
            "new_allocation":  r.new_allocation,
            "equity_before":   r.equity_before,
            "equity_after":    r.equity_after,
            "commission":      r.commission,
            "slippage":        r.slippage,
            "regime":          r.regime,
            "regime_prob":     r.regime_prob,
            "window_id":       r.window_id,
        } for r in records]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _regime_df(self, records: List[BarRecord]) -> pd.DataFrame:
        rows = [{
            "timestamp":   r.timestamp,
            "regime":      r.regime,
            "regime_prob": r.regime_prob,
            "window_id":   r.window_id,
        } for r in records]
        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
