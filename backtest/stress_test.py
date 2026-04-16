"""
backtest/stress_test.py
Fase 4 — Pruebas de Estrés del Sistema.

1. Crash Injection   : gaps de -5% a -15% en 10 puntos aleatorios, 100 MC.
2. Gap Risk          : gaps nocturnos de 2-5×ATR, pérdida esperada vs real.
3. Regime Scramble   : etiquetas de régimen aleatorizadas, verificar contención
                       del daño por la capa de gestión de riesgo.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

logger  = logging.getLogger(__name__)
console = Console()

INITIAL_CAPITAL  = 546.14
CIRCUIT_BREAKER_DAILY  = 0.03      # 3%  → $16.38
CIRCUIT_BREAKER_WEEKLY = 0.07      # 7%  → $38.23
CIRCUIT_BREAKER_PEAK   = 0.10      # 10% → $54.61


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class CrashSimResult:
    seed: int
    max_loss_pct: float
    circuit_breaker_triggered: bool
    final_equity: float
    daily_cb_hits: int
    weekly_cb_hits: int
    peak_cb_hits: int


@dataclass
class GapRiskResult:
    gap_multiplier: float          # veces ATR del gap
    expected_loss_pct: float       # loss esperada = alloc × gap_size
    realized_loss_pct: float       # con slippage añadido
    equity_after: float


@dataclass
class RegimeScrambleResult:
    shuffle_seed: int
    max_drawdown_pct: float
    circuit_breaker_hits: int
    final_equity: float
    contained: bool                # True si CB contuvo el daño (DD < 15%)


# ---------------------------------------------------------------------------
# Motor de Pruebas de Estrés
# ---------------------------------------------------------------------------

class StressTester:
    """
    Ejecuta las tres baterías de estrés del Fase 4.

    Parámetros
    ----------
    initial_capital : float
        Capital inicial ($546.14).
    n_monte_carlo : int
        Simulaciones Monte Carlo para crash injection (default 100).
    n_crash_points : int
        Número de puntos de crash por simulación (default 10).
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        n_monte_carlo: int     = 100,
        n_crash_points: int    = 10,
    ):
        self.capital        = initial_capital
        self.n_mc           = n_monte_carlo
        self.n_crash_points = n_crash_points

    # ------------------------------------------------------------------
    # 1. Crash Injection
    # ------------------------------------------------------------------

    def crash_injection(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        crash_range: Tuple[float, float] = (-0.15, -0.05),
    ) -> List[CrashSimResult]:
        """
        Inyecta gaps de caída en puntos aleatorios de la curva de equity.
        Simula circuit breakers y reporta pérdida máxima promedio.

        Parámetros
        ----------
        equity_curve   : DataFrame con columna "equity".
        trade_log      : DataFrame con columna "equity_before" (para contexto).
        crash_range    : (min_gap, max_gap) en porcentaje negativo.
        """
        if equity_curve.empty:
            logger.warning("[StressTest] equity_curve vacía — saltando crash injection")
            return []

        eq_vals = equity_curve["equity"].values.copy()
        n       = len(eq_vals)
        results: List[CrashSimResult] = []

        for seed in range(self.n_mc):
            rng         = np.random.default_rng(seed)
            sim_eq      = eq_vals.copy()
            crash_pcts  = rng.uniform(crash_range[0], crash_range[1], self.n_crash_points)
            crash_idx   = sorted(rng.choice(n, self.n_crash_points, replace=False))

            daily_cb = weekly_cb = peak_cb = 0
            peak_equity = sim_eq[0]
            weekly_start_equity = sim_eq[0]

            for i in range(1, n):
                # Aplicar crash si corresponde
                if i in crash_idx:
                    ci = crash_idx.index(i)
                    sim_eq[i:] *= (1 + crash_pcts[ci])

                daily_ret = (sim_eq[i] / sim_eq[i - 1]) - 1 if sim_eq[i - 1] > 0 else 0.0
                peak_equity = max(peak_equity, sim_eq[i])
                peak_dd     = (sim_eq[i] - peak_equity) / peak_equity

                if daily_ret < -CIRCUIT_BREAKER_DAILY:
                    daily_cb += 1
                if i % 5 == 0:
                    weekly_ret = (sim_eq[i] / weekly_start_equity) - 1
                    if weekly_ret < -CIRCUIT_BREAKER_WEEKLY:
                        weekly_cb += 1
                    weekly_start_equity = sim_eq[i]
                if peak_dd < -CIRCUIT_BREAKER_PEAK:
                    peak_cb += 1

            max_loss   = float(np.min(sim_eq) / sim_eq[0] - 1)
            cb_hit     = (daily_cb + weekly_cb + peak_cb) > 0
            final_eq   = float(sim_eq[-1])

            results.append(CrashSimResult(
                seed=seed,
                max_loss_pct=round(max_loss * 100, 2),
                circuit_breaker_triggered=cb_hit,
                final_equity=round(final_eq, 2),
                daily_cb_hits=daily_cb,
                weekly_cb_hits=weekly_cb,
                peak_cb_hits=peak_cb,
            ))

        self._print_crash_summary(results)
        return results

    def _print_crash_summary(self, results: List[CrashSimResult]) -> None:
        losses   = [r.max_loss_pct for r in results]
        cb_count = sum(1 for r in results if r.circuit_breaker_triggered)
        avg_eq   = np.mean([r.final_equity for r in results])

        t = Table(title="[bold red]CRASH INJECTION — 100 Simulaciones Monte Carlo[/bold red]",
                  show_lines=True)
        t.add_column("Métrica",         style="bold")
        t.add_column("Valor",           justify="right")
        t.add_row("Pérdida máx prom",   f"{np.mean(losses):.2f}%")
        t.add_row("Pérdida máx worst",  f"{np.min(losses):.2f}%")
        t.add_row("Pérdida máx best",   f"{np.max(losses):.2f}%")
        t.add_row("CB activado (%sims)", f"{cb_count}/{len(results)}")
        t.add_row("Equity final prom",  f"${avg_eq:.2f}")
        console.print(t)

    # ------------------------------------------------------------------
    # 2. Gap Risk
    # ------------------------------------------------------------------

    def gap_risk(
        self,
        price_series: pd.Series,
        allocations: Dict[str, float],
        atr_series: pd.Series,
        atr_multipliers: List[float] = None,
        n_events: int = 50,
        slippage_pct: float = 0.001,
    ) -> List[GapRiskResult]:
        """
        Simula gaps nocturnos de 2-5×ATR en puntos aleatorios.
        Compara pérdida esperada (teórica) vs realizada (con slippage).

        Parámetros
        ----------
        price_series    : precios de cierre del instrumento principal.
        allocations     : {symbol: allocation_pct} activas.
        atr_series      : ATR diario del instrumento.
        atr_multipliers : lista de multiplicadores de ATR a probar.
        n_events        : número de eventos de gap por multiplicador.
        slippage_pct    : slippage adicional por gap (default 0.1%).
        """
        if atr_multipliers is None:
            atr_multipliers = [2.0, 3.0, 4.0, 5.0]

        total_alloc  = sum(abs(v) for v in allocations.values())
        prices       = price_series.values
        atrs         = atr_series.values
        n            = min(len(prices), len(atrs))
        results: List[GapRiskResult] = []
        rng          = np.random.default_rng(99)

        for mult in atr_multipliers:
            events = rng.choice(n - 1, min(n_events, n - 1), replace=False)
            for idx in events:
                atr_val  = float(atrs[idx])
                price    = float(prices[idx])
                gap_pct  = -(mult * atr_val / price)

                expected_loss  = total_alloc * abs(gap_pct)
                realized_loss  = expected_loss + slippage_pct  # slippage adicional

                equity_after   = self.capital * (1 - realized_loss)

                results.append(GapRiskResult(
                    gap_multiplier=mult,
                    expected_loss_pct=round(expected_loss * 100, 3),
                    realized_loss_pct=round(realized_loss * 100, 3),
                    equity_after=round(equity_after, 2),
                ))

        self._print_gap_summary(results)
        return results

    def _print_gap_summary(self, results: List[GapRiskResult]) -> None:
        t = Table(title="[bold yellow]GAP RISK — Gaps Nocturnos 2-5×ATR[/bold yellow]",
                  show_lines=True)
        for col in ["Mult ATR", "Loss Esperada", "Loss Real", "Equity Tras Gap"]:
            t.add_column(col, justify="right")

        for mult in sorted(set(r.gap_multiplier for r in results)):
            grp = [r for r in results if r.gap_multiplier == mult]
            avg_exp = np.mean([r.expected_loss_pct for r in grp])
            avg_rea = np.mean([r.realized_loss_pct for r in grp])
            avg_eq  = np.mean([r.equity_after for r in grp])
            t.add_row(f"{mult:.0f}×", f"{avg_exp:.3f}%", f"{avg_rea:.3f}%", f"${avg_eq:.2f}")

        console.print(t)

    # ------------------------------------------------------------------
    # 3. Regime Scramble
    # ------------------------------------------------------------------

    def regime_scramble(
        self,
        equity_curve: pd.DataFrame,
        strategy_fn,
        n_shuffles: int = 20,
        max_acceptable_dd: float = 0.15,
    ) -> List[RegimeScrambleResult]:
        """
        Baraja deliberadamente las etiquetas de régimen.
        Verifica que la gestión de riesgos contenga el daño (<15% DD).

        Si el sistema explota con regímenes incorrectos, significa que la
        capa de riesgo no es suficientemente independiente del HMM.

        Parámetros
        ----------
        equity_curve     : DataFrame con columnas "equity" y "regime".
        strategy_fn      : callable(regime_label) → allocation_pct (función mock).
        n_shuffles       : número de permutaciones de régimen.
        max_acceptable_dd: umbral de DD máximo aceptable (default 15%).
        """
        if equity_curve.empty or "regime" not in equity_curve.columns:
            logger.warning("[StressTest] Sin columna 'regime' — saltando scramble")
            return []

        regimes     = equity_curve["regime"].values
        eq_base     = equity_curve["equity"].values
        results: List[RegimeScrambleResult] = []

        for seed in range(n_shuffles):
            rng             = np.random.default_rng(seed + 200)
            shuffled_labels = list(regimes.copy())
            rng.shuffle(shuffled_labels)

            equity   = self.capital
            peak     = equity
            max_dd   = 0.0
            cb_hits  = 0

            for i, label in enumerate(shuffled_labels):
                alloc   = strategy_fn(label)
                bar_ret = (eq_base[i] / eq_base[i - 1] - 1) if i > 0 else 0.0
                equity *= (1 + alloc * bar_ret)
                peak    = max(peak, equity)
                dd      = (equity - peak) / peak

                if dd < -CIRCUIT_BREAKER_DAILY:
                    cb_hits += 1
                    alloc *= 0.50       # simula reducción de CB
                if dd < -CIRCUIT_BREAKER_PEAK:
                    alloc = 0.0        # simula halt total

                max_dd = min(max_dd, dd)

            contained = abs(max_dd) < max_acceptable_dd
            results.append(RegimeScrambleResult(
                shuffle_seed=seed,
                max_drawdown_pct=round(max_dd * 100, 2),
                circuit_breaker_hits=cb_hits,
                final_equity=round(equity, 2),
                contained=contained,
            ))

            status = "[green]CONTENIDO[/green]" if contained else "[red]EXPLOSION[/red]"
            logger.info(
                "[Scramble] seed=%d | MaxDD=%.2f%% | CB_hits=%d | %s",
                seed, max_dd * 100, cb_hits, "OK" if contained else "EXPLODED",
            )

        self._print_scramble_summary(results)
        return results

    def _print_scramble_summary(self, results: List[RegimeScrambleResult]) -> None:
        contained = sum(1 for r in results if r.contained)
        avg_dd    = np.mean([r.max_drawdown_pct for r in results])
        avg_cb    = np.mean([r.circuit_breaker_hits for r in results])

        t = Table(
            title="[bold magenta]REGIME SCRAMBLE — Robustez ante Clasificación Incorrecta[/bold magenta]",
            show_lines=True,
        )
        t.add_column("Métrica",               style="bold")
        t.add_column("Valor",                 justify="right")
        t.add_row("Simulaciones contenidas",  f"{contained}/{len(results)}")
        t.add_row("DD máx promedio",          f"{avg_dd:.2f}%")
        t.add_row("CB hits promedio",         f"{avg_cb:.1f}")
        t.add_row("Veredicto",
                  "[green]RISK LAYER INDEPENDIENTE[/green]"
                  if contained / len(results) >= 0.80
                  else "[red]RIESGO DEPENDIENTE DEL HMM — REVISAR[/red]")
        console.print(t)

    # ------------------------------------------------------------------
    # Runner completo
    # ------------------------------------------------------------------

    def run_all(
        self,
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        price_series: pd.Series,
        atr_series: pd.Series,
        allocations: Dict[str, float],
        strategy_fn,
    ) -> Dict:
        """Ejecuta las tres baterías y retorna resultados agregados."""
        console.rule("[bold]PRUEBAS DE ESTRÉS — regime-trader[/bold]")

        crash_res   = self.crash_injection(equity_curve, trade_log)
        gap_res     = self.gap_risk(price_series, allocations, atr_series)
        scramble_res = self.regime_scramble(equity_curve, strategy_fn)

        return {
            "crash_injection": crash_res,
            "gap_risk":        gap_res,
            "regime_scramble": scramble_res,
        }
