"""
backtest/performance.py
Fase 4 — Métricas de Rendimiento y Comparación con Benchmarks.

Salida: tablas Rich en terminal + CSV (equity_curve, trade_log,
        regime_history, benchmark_comparison).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

logger   = logging.getLogger(__name__)
console  = Console()

TRADING_DAYS_YEAR = 252
RISK_FREE_RATE    = 0.045
OUTPUT_DIR        = "backtest_results"


# ---------------------------------------------------------------------------
# Dataclass de métricas
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    max_drawdown_days: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    avg_holding_days: float
    worst_day: float
    worst_week: float
    worst_month: float
    max_consec_losses: int
    max_underwater_days: int
    label: str = "Strategy"


@dataclass
class RegimeStats:
    regime: str
    pct_time: float
    return_contribution: float
    avg_pnl_per_trade: float
    win_rate: float
    sharpe: float


@dataclass
class ConfidenceBucket:
    bucket: str                 # "<50%", "50-60%", "60-70%", "70%+"
    n_trades: int
    sharpe: float
    win_rate: float
    avg_pnl: float


# ---------------------------------------------------------------------------
# Calculador principal
# ---------------------------------------------------------------------------

class PerformanceCalculator:

    def __init__(self, risk_free: float = RISK_FREE_RATE):
        self.rf = risk_free

    # ------------------------------------------------------------------
    # Métricas core
    # ------------------------------------------------------------------

    def compute(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        label: str = "Strategy",
    ) -> PerformanceMetrics:
        """Calcula todas las métricas core a partir de equity_curve y trade_log."""
        if equity_curve.empty:
            raise ValueError("equity_curve vacía.")

        eq     = equity_curve["equity"].astype(float)
        rets   = eq.pct_change().dropna()
        n_days = len(eq)

        total_ret   = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        cagr        = self._cagr(eq)
        sharpe      = self._sharpe(rets)
        sortino     = self._sortino(rets)
        max_dd, dd_days = self._max_drawdown(eq)
        calmar      = cagr / abs(max_dd) if max_dd != 0 else 0.0

        win_rate, avg_win, avg_loss, pf, n_trades, avg_hold = self._trade_stats(trade_log)

        wc_day, wc_week, wc_month = self._worst_cases(rets, eq)
        max_consec    = self._max_consecutive_losses(trade_log)
        max_underwater = self._max_underwater(eq)

        return PerformanceMetrics(
            total_return_pct=round(total_ret, 2),
            cagr=round(cagr * 100, 2),
            sharpe=round(sharpe, 3),
            sortino=round(sortino, 3),
            calmar=round(calmar, 3),
            max_drawdown_pct=round(max_dd * 100, 2),
            max_drawdown_days=dd_days,
            win_rate=round(win_rate * 100, 1),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(pf, 2),
            total_trades=n_trades,
            avg_holding_days=round(avg_hold, 1),
            worst_day=round(wc_day * 100, 2),
            worst_week=round(wc_week * 100, 2),
            worst_month=round(wc_month * 100, 2),
            max_consec_losses=max_consec,
            max_underwater_days=max_underwater,
            label=label,
        )

    # ------------------------------------------------------------------
    # Métricas por régimen
    # ------------------------------------------------------------------

    def regime_breakdown(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
    ) -> List[RegimeStats]:
        """Estadísticas desagregadas por régimen HMM."""
        if equity_curve.empty or "regime" not in equity_curve.columns:
            return []

        results: List[RegimeStats] = []
        total_bars = len(equity_curve)

        for regime, grp in equity_curve.groupby("regime"):
            pct_time   = len(grp) / total_bars
            rets_grp   = grp["equity"].pct_change().dropna()
            sharpe     = self._sharpe(rets_grp)

            regime_trades = trade_log[trade_log["regime"] == regime] if not trade_log.empty else pd.DataFrame()
            wrate, avg_w, avg_l, pf, n, _ = self._trade_stats(regime_trades)

            # Contribución al retorno total
            ret_contrib = float(np.prod(1 + rets_grp) - 1) if len(rets_grp) > 0 else 0.0

            results.append(RegimeStats(
                regime=str(regime),
                pct_time=round(pct_time * 100, 1),
                return_contribution=round(ret_contrib * 100, 2),
                avg_pnl_per_trade=round(avg_w if avg_w > 0 else avg_l, 4),
                win_rate=round(wrate * 100, 1),
                sharpe=round(sharpe, 3),
            ))

        return sorted(results, key=lambda x: x.return_contribution, reverse=True)

    # ------------------------------------------------------------------
    # Cubetas de confianza
    # ------------------------------------------------------------------

    def confidence_buckets(self, trade_log: pd.DataFrame) -> List[ConfidenceBucket]:
        """Analiza métricas por nivel de confianza del régimen."""
        if trade_log.empty or "regime_prob" not in trade_log.columns:
            return []

        buckets = [
            ("<50%",   0.00, 0.50),
            ("50-60%", 0.50, 0.60),
            ("60-70%", 0.60, 0.70),
            ("70%+",   0.70, 1.01),
        ]
        results: List[ConfidenceBucket] = []

        for label, lo, hi in buckets:
            mask = (trade_log["regime_prob"] >= lo) & (trade_log["regime_prob"] < hi)
            grp  = trade_log[mask]
            if grp.empty:
                continue

            pnls = (grp["equity_after"] - grp["equity_before"]).values
            wins = pnls[pnls > 0]
            loss = pnls[pnls <= 0]
            wrate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
            avg_pnl = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

            daily_rets = pd.Series(pnls / grp["equity_before"].values)
            sharpe = self._sharpe(daily_rets)

            results.append(ConfidenceBucket(
                bucket=label,
                n_trades=len(grp),
                sharpe=round(sharpe, 3),
                win_rate=round(wrate * 100, 1),
                avg_pnl=round(avg_pnl, 4),
            ))

        return results

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def benchmark_buyhold(
        self,
        price_series: pd.Series,
        initial_capital: float = 546.14,
    ) -> PerformanceMetrics:
        """Buy-and-hold: comprado todo el periodo."""
        eq = initial_capital * price_series / price_series.iloc[0]
        eq_df = pd.DataFrame({"equity": eq})
        return self.compute(eq_df, pd.DataFrame(), label="Buy-and-Hold")

    def benchmark_sma200(
        self,
        price_series: pd.Series,
        initial_capital: float = 546.14,
    ) -> PerformanceMetrics:
        """
        SMA200 trend: LONG si precio > SMA200, cash si precio < SMA200.
        """
        sma200   = price_series.rolling(200).mean()
        invested = (price_series > sma200).astype(float)
        daily    = price_series.pct_change().fillna(0)
        strat    = initial_capital * (1 + (daily * invested)).cumprod()
        eq_df    = pd.DataFrame({"equity": strat})
        return self.compute(eq_df, pd.DataFrame(), label="SMA200-Trend")

    def benchmark_random(
        self,
        price_series: pd.Series,
        rebalance_freq: int,
        initial_capital: float = 546.14,
        n_seeds: int = 100,
    ) -> Tuple[PerformanceMetrics, float]:
        """
        Entradas aleatorias con la misma frecuencia de rebalanceo.
        Retorna (media_métricas, std_retorno_total).
        """
        daily   = price_series.pct_change().fillna(0).values
        n       = len(daily)
        returns = []

        for seed in range(n_seeds):
            rng     = np.random.default_rng(seed)
            eq      = initial_capital
            alloc   = 0.0
            for i in range(n):
                eq *= (1 + alloc * daily[i])
                if i % rebalance_freq == 0:
                    alloc = rng.uniform(0.0, 0.95)
            returns.append((eq - initial_capital) / initial_capital * 100)

        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns))

        # Construir equity curve media para métricas
        eq_vals = [initial_capital * (1 + mean_ret / 100)]
        eq_df   = pd.DataFrame({"equity": pd.Series(eq_vals * n)})
        metrics = self.compute(eq_df, pd.DataFrame(), label=f"Random(n={n_seeds})")
        metrics.total_return_pct = round(mean_ret, 2)
        return metrics, round(std_ret, 2)

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------

    def print_summary(
        self,
        strategy: PerformanceMetrics,
        benchmarks: List[PerformanceMetrics],
        regime_stats: List[RegimeStats],
        conf_buckets: List[ConfidenceBucket],
    ) -> None:

        # ── Tabla resumen ──────────────────────────────────────────────
        t = Table(title="[bold cyan]RESUMEN DE RENDIMIENTO[/bold cyan]", show_lines=True)
        t.add_column("Métrica", style="bold")
        for m in [strategy] + benchmarks:
            t.add_column(m.label, justify="right")

        rows = [
            ("Retorno Total (%)",    lambda m: f"{m.total_return_pct:.2f}%"),
            ("CAGR (%)",             lambda m: f"{m.cagr:.2f}%"),
            ("Sharpe",               lambda m: f"{m.sharpe:.3f}"),
            ("Sortino",              lambda m: f"{m.sortino:.3f}"),
            ("Calmar",               lambda m: f"{m.calmar:.3f}"),
            ("Max Drawdown (%)",     lambda m: f"{m.max_drawdown_pct:.2f}%"),
            ("DD Duración (días)",   lambda m: str(m.max_drawdown_days)),
            ("Win Rate (%)",         lambda m: f"{m.win_rate:.1f}%"),
            ("Profit Factor",        lambda m: f"{m.profit_factor:.2f}"),
            ("Total Trades",         lambda m: str(m.total_trades)),
            ("Peor Día (%)",         lambda m: f"{m.worst_day:.2f}%"),
            ("Peor Semana (%)",      lambda m: f"{m.worst_week:.2f}%"),
            ("Max Consec. Pérdidas", lambda m: str(m.max_consec_losses)),
            ("Max Bajo el Agua (d)", lambda m: str(m.max_underwater_days)),
        ]
        for name, fn in rows:
            t.add_row(name, *[fn(m) for m in [strategy] + benchmarks])
        console.print(t)

        # ── Tabla por régimen ──────────────────────────────────────────
        if regime_stats:
            rt = Table(title="[bold cyan]MÉTRICAS POR RÉGIMEN[/bold cyan]", show_lines=True)
            for col in ["Régimen", "% Tiempo", "Contrib. Retorno", "P&L Prom/Op", "Win Rate", "Sharpe"]:
                rt.add_column(col, justify="right")
            for rs in regime_stats:
                rt.add_row(
                    rs.regime,
                    f"{rs.pct_time:.1f}%",
                    f"{rs.return_contribution:.2f}%",
                    f"{rs.avg_pnl_per_trade:.4f}",
                    f"{rs.win_rate:.1f}%",
                    f"{rs.sharpe:.3f}",
                )
            console.print(rt)

        # ── Tabla cubetas confianza ────────────────────────────────────
        if conf_buckets:
            ct = Table(title="[bold cyan]ANÁLISIS POR CONFIANZA[/bold cyan]", show_lines=True)
            for col in ["Confianza", "# Trades", "Sharpe", "Win Rate", "P&L Prom"]:
                ct.add_column(col, justify="right")
            for cb in conf_buckets:
                ct.add_row(cb.bucket, str(cb.n_trades), f"{cb.sharpe:.3f}",
                           f"{cb.win_rate:.1f}%", f"{cb.avg_pnl:.4f}")
            console.print(ct)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_csv(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        regime_history: pd.DataFrame,
        benchmark_metrics: List[PerformanceMetrics],
    ) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not equity_curve.empty:
            equity_curve.to_csv(f"{OUTPUT_DIR}/equity_curve.csv")
        if not trade_log.empty:
            trade_log.to_csv(f"{OUTPUT_DIR}/trade_log.csv", index=False)
        if not regime_history.empty:
            regime_history.to_csv(f"{OUTPUT_DIR}/regime_history.csv")

        if benchmark_metrics:
            rows = []
            for m in benchmark_metrics:
                rows.append({
                    "label":         m.label,
                    "total_return":  m.total_return_pct,
                    "cagr":          m.cagr,
                    "sharpe":        m.sharpe,
                    "sortino":       m.sortino,
                    "max_drawdown":  m.max_drawdown_pct,
                    "win_rate":      m.win_rate,
                    "profit_factor": m.profit_factor,
                })
            pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/benchmark_comparison.csv", index=False)

        logger.info("CSVs exportados a %s/", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Utilidades estadísticas privadas
    # ------------------------------------------------------------------

    def _cagr(self, eq: pd.Series) -> float:
        n_years = len(eq) / TRADING_DAYS_YEAR
        return (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    def _sharpe(self, rets: pd.Series) -> float:
        if rets.empty or rets.std() == 0:
            return 0.0
        excess = rets.mean() - self.rf / TRADING_DAYS_YEAR
        return float(excess / rets.std() * np.sqrt(TRADING_DAYS_YEAR))

    def _sortino(self, rets: pd.Series) -> float:
        if rets.empty:
            return 0.0
        downside = rets[rets < 0]
        if downside.empty or downside.std() == 0:
            return 0.0
        excess = rets.mean() - self.rf / TRADING_DAYS_YEAR
        return float(excess / downside.std() * np.sqrt(TRADING_DAYS_YEAR))

    def _max_drawdown(self, eq: pd.Series) -> Tuple[float, int]:
        roll_max = eq.cummax()
        dd       = (eq - roll_max) / roll_max
        max_dd   = float(dd.min())
        # Duración: buscar el periodo más largo bajo el agua
        underwater = (dd < 0)
        max_days, cur = 0, 0
        for val in underwater:
            cur = cur + 1 if val else 0
            max_days = max(max_days, cur)
        return max_dd, max_days

    def _trade_stats(
        self, trade_log: pd.DataFrame
    ) -> Tuple[float, float, float, float, int, float]:
        if trade_log.empty or "equity_after" not in trade_log.columns:
            return 0.0, 0.0, 0.0, 0.0, 0, 0.0

        pnls  = (trade_log["equity_after"] - trade_log["equity_before"]).values
        wins  = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        n     = len(pnls)

        wrate   = len(wins) / n if n > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        pf      = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

        avg_hold = 0.0
        if "timestamp" in trade_log.columns and n > 1:
            ts = pd.to_datetime(trade_log["timestamp"])
            avg_hold = float((ts.max() - ts.min()).days / n)

        return wrate, avg_win, avg_loss, pf, n, avg_hold

    def _worst_cases(
        self, rets: pd.Series, eq: pd.Series
    ) -> Tuple[float, float, float]:
        worst_day   = float(rets.min()) if not rets.empty else 0.0
        weekly      = eq.resample("W").last().pct_change().dropna()
        worst_week  = float(weekly.min()) if not weekly.empty else 0.0
        monthly     = eq.resample("ME").last().pct_change().dropna()
        worst_month = float(monthly.min()) if not monthly.empty else 0.0
        return worst_day, worst_week, worst_month

    def _max_consecutive_losses(self, trade_log: pd.DataFrame) -> int:
        if trade_log.empty or "equity_after" not in trade_log.columns:
            return 0
        pnls = (trade_log["equity_after"] - trade_log["equity_before"]).values
        max_consec, cur = 0, 0
        for p in pnls:
            cur = cur + 1 if p < 0 else 0
            max_consec = max(max_consec, cur)
        return max_consec

    def _max_underwater(self, eq: pd.Series) -> int:
        roll_max   = eq.cummax()
        underwater = (eq < roll_max)
        max_d, cur = 0, 0
        for val in underwater:
            cur = cur + 1 if val else 0
            max_d = max(max_d, cur)
        return max_d
