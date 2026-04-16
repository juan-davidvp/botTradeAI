"""
core/hmm_engine.py
Fase 2 — Motor HMM de Detección de Regímenes de Volatilidad.

- Selección automática de modelo via BIC (n_components ∈ [3,4,5,6,7]).
- Inferencia sin look-ahead bias: Algoritmo Forward puro.
- Filtro de estabilidad (min 3 barras) y detección de Flicker.
- Persistencia con pickle + metadatos.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeo de etiquetas por número de regímenes
# ---------------------------------------------------------------------------
REGIME_LABELS: Dict[int, List[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeInfo:
    regime_id: int
    name: str
    expected_return: float
    expected_volatility: float
    max_leverage_allowed: float
    max_position_size_pct: float


@dataclass
class RegimeState:
    label: str
    state_id: int
    probability: float
    state_probabilities: List[float]
    timestamp: datetime
    is_confirmed: bool
    consecutive_bars: int


# ---------------------------------------------------------------------------
# Motor HMM
# ---------------------------------------------------------------------------

class HMMEngine:
    """
    Motor de detección de regímenes de volatilidad con Gaussian HMM.

    Parámetros
    ----------
    n_candidates : list[int]
        Candidatos de n_components para selección por BIC.
    n_init : int
        Número de inicializaciones aleatorias por candidato.
    covariance_type : str
        Tipo de covarianza (default "full").
    stability_bars : int
        Barras mínimas para confirmar cambio de régimen.
    flicker_window : int
        Ventana de barras para calcular tasa de parpadeo.
    flicker_threshold : int
        Cambios máximos en la ventana antes de activar modo incertidumbre.
    min_confidence : float
        Probabilidad mínima para aceptar el régimen predicho.
    model_path : str
        Ruta para persistencia del modelo.
    """

    def __init__(
        self,
        n_candidates: List[int] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
        model_path: str = "models/hmm_model.pkl",
        min_train_bars: int = 126,
    ):
        self.n_candidates    = n_candidates or [3, 4, 5, 6, 7]
        self.n_init          = n_init
        self.covariance_type = covariance_type
        self.stability_bars  = stability_bars
        self.flicker_window  = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence  = min_confidence
        self.model_path      = model_path
        self.min_train_bars  = min_train_bars

        # Estado interno
        self.model: Optional[GaussianHMM] = None
        self.n_regimes: int = 0
        self.regime_labels: List[str] = []
        self.regime_order: List[int] = []          # índices HMM ordenados por retorno medio
        self.regime_infos: Dict[int, RegimeInfo] = {}
        self.trained_at: Optional[datetime] = None
        self.bic_score: float = float("inf")

        # Estado de inferencia filtrada
        self._log_alpha: Optional[np.ndarray] = None   # log-prob forward del paso anterior
        self._prev_label: Optional[str] = None
        self._consecutive_bars: int = 0
        self._pending_label: Optional[str] = None
        self._pending_count: int = 0
        self._regime_history: List[str] = []            # últimas flicker_window predicciones

    # ------------------------------------------------------------------
    # BIC
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bic(model: GaussianHMM, X: np.ndarray) -> float:
        """BIC = -2 * logL + k * log(N)."""
        log_likelihood = model.score(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n = model.n_components
        # parámetros: startprob (n-1), transmat (n*(n-1)), means (n*f), covars (n*f*f para "full")
        k = (n - 1) + n * (n - 1) + n * n_features + n * n_features ** 2
        return -2 * log_likelihood * n_samples + k * np.log(n_samples)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray) -> None:
        """
        Entrena el HMM con selección de n_components via BIC.

        Parámetros
        ----------
        X : np.ndarray shape (T, n_features)
            Matriz de características normalizadas.
        """
        if X.shape[0] < self.min_train_bars:
            raise ValueError(
                f"Datos insuficientes: {X.shape[0]} barras. Mínimo requerido: {self.min_train_bars}."
            )

        best_model  = None
        best_bic    = float("inf")
        bic_scores: Dict[int, float] = {}

        for n in self.n_candidates:
            for attempt in range(self.n_init):
                try:
                    model = GaussianHMM(
                        n_components=n,
                        covariance_type=self.covariance_type,
                        n_iter=200,
                        tol=1e-4,
                        random_state=attempt,
                        verbose=False,
                    )
                    model.fit(X)
                    bic = self._compute_bic(model, X)
                    if bic < bic_scores.get(n, float("inf")):
                        bic_scores[n] = bic
                        if bic < best_bic:
                            best_bic   = bic
                            best_model = model
                except Exception as exc:
                    logger.debug("n=%d attempt=%d fallo: %s", n, attempt, exc)

        if best_model is None:
            raise RuntimeError("Ningún modelo HMM convergió durante el entrenamiento.")

        # Log de scores BIC
        for n, bic in sorted(bic_scores.items()):
            marker = " ← SELECCIONADO" if n == best_model.n_components else ""
            logger.info("BIC n=%d: %.2f%s", n, bic, marker)

        self.model      = best_model
        self.n_regimes  = best_model.n_components
        self.bic_score  = best_bic
        self.trained_at = datetime.utcnow()

        # Ordenar estados por retorno medio (usa primera feature ~ ret_1)
        mean_returns = best_model.means_[:, 0]
        self.regime_order  = list(np.argsort(mean_returns))
        self.regime_labels = REGIME_LABELS[self.n_regimes]

        # Construir RegimeInfo por estado HMM
        self._build_regime_infos()

        # Resetear estado de inferencia
        self._log_alpha = None
        self._prev_label = None
        self._consecutive_bars = 0
        self._pending_label = None
        self._pending_count = 0
        self._regime_history = []

        logger.info(
            "HMM entrenado: n_regimes=%d, BIC=%.2f, convergido=%s, fecha=%s",
            self.n_regimes, self.bic_score,
            best_model.monitor_.converged,
            self.trained_at.isoformat(),
        )

    def _build_regime_infos(self) -> None:
        """Construye RegimeInfo para cada estado HMM basado en estadísticas aprendidas."""
        self.regime_infos = {}
        vol_col_idx = 3  # índice de vol_20 en el vector de features

        for rank, state_id in enumerate(self.regime_order):
            label = self.regime_labels[rank]
            exp_ret = float(self.model.means_[state_id, 0])
            exp_vol = float(np.sqrt(self.model.covars_[state_id][vol_col_idx, vol_col_idx]))

            # Política conservadora: menor asignación en regímenes extremos
            if label in ("CRASH", "STRONG_BEAR", "BEAR"):
                max_pos = 0.10
                max_lev = 1.0
            elif label in ("EUPHORIA",):
                max_pos = 0.15
                max_lev = 1.0
            else:
                max_pos = 0.20
                max_lev = 1.0

            self.regime_infos[state_id] = RegimeInfo(
                regime_id=state_id,
                name=label,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                max_leverage_allowed=max_lev,
                max_position_size_pct=max_pos,
            )

    # ------------------------------------------------------------------
    # Algoritmo Forward (sin look-ahead bias)
    # ------------------------------------------------------------------

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """
        Log-probabilidad de emisión para cada estado dado el vector de observación.
        Asume distribución Gaussiana multivariada.
        """
        n = self.n_regimes
        log_probs = np.zeros(n)
        for state in range(n):
            mean  = self.model.means_[state]
            covar = self.model.covars_[state]
            diff  = obs - mean
            try:
                sign, logdet = np.linalg.slogdet(covar)
                inv_covar    = np.linalg.inv(covar)
                k            = len(obs)
                log_probs[state] = -0.5 * (
                    k * np.log(2 * np.pi) + logdet + diff @ inv_covar @ diff
                )
            except np.linalg.LinAlgError:
                log_probs[state] = -np.inf
        return log_probs

    def _forward_step(self, obs: np.ndarray) -> np.ndarray:
        """
        Un paso del algoritmo Forward en espacio logarítmico.
        Retorna log_alpha_t (distribución filtrada normalizada).
        """
        log_emit = self._log_emission(obs)

        if self._log_alpha is None:
            # Inicialización: alpha_0 = startprob * emission(obs_0)
            log_alpha = np.log(self.model.startprob_ + 1e-300) + log_emit
        else:
            # alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)
            log_transmat = np.log(self.model.transmat_ + 1e-300)
            # log-sum-exp para estabilidad numérica
            log_pred = np.array([
                np.logaddexp.reduce(self._log_alpha + log_transmat[:, s])
                for s in range(self.n_regimes)
            ])
            log_alpha = log_pred + log_emit

        # Normalización (log-sum-exp)
        log_norm = np.logaddexp.reduce(log_alpha)
        log_alpha -= log_norm
        self._log_alpha = log_alpha
        return log_alpha

    def predict_regime_filtered(self, features_up_to_now: np.ndarray) -> RegimeState:
        """
        Predice el régimen en t usando ÚNICAMENTE datos hasta t (sin look-ahead).

        Parámetros
        ----------
        features_up_to_now : np.ndarray shape (T, n_features) o (n_features,)
            Si es 2D, procesa toda la secuencia (útil para inicialización).
            Si es 1D, aplica un solo paso forward.

        Retorna
        -------
        RegimeState con el régimen actual, probabilidades y metadatos de estabilidad.
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")

        if features_up_to_now.ndim == 1:
            log_alpha = self._forward_step(features_up_to_now)
        else:
            for obs in features_up_to_now:
                log_alpha = self._forward_step(obs)

        state_probs = np.exp(log_alpha)
        raw_state   = int(np.argmax(state_probs))
        probability = float(state_probs[raw_state])

        # Convertir state HMM a etiqueta legible usando regime_order
        rank  = self.regime_order.index(raw_state)
        label = self.regime_labels[rank]

        # -- Filtro de estabilidad --
        is_confirmed = self._update_stability(label)

        # -- Historial para Flicker --
        self._regime_history.append(label)
        if len(self._regime_history) > self.flicker_window:
            self._regime_history.pop(0)

        return RegimeState(
            label=label,
            state_id=raw_state,
            probability=probability,
            state_probabilities=state_probs.tolist(),
            timestamp=datetime.utcnow(),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    # ------------------------------------------------------------------
    # Estabilidad y Flicker
    # ------------------------------------------------------------------

    def _update_stability(self, new_label: str) -> bool:
        """
        Aplica el filtro de estabilidad: el régimen debe persistir
        stability_bars barras antes de confirmarse.

        Retorna True si el régimen está confirmado.
        """
        if self._prev_label is None:
            # Primera predicción
            self._prev_label       = new_label
            self._consecutive_bars = 1
            self._pending_label    = None
            self._pending_count    = 0
            return False

        if new_label == self._prev_label:
            self._consecutive_bars += 1
            self._pending_label = None
            self._pending_count = 0
        else:
            # Posible cambio de régimen — esperar stability_bars
            if new_label == self._pending_label:
                self._pending_count += 1
            else:
                self._pending_label = new_label
                self._pending_count = 1

            if self._pending_count >= self.stability_bars:
                old_label = self._prev_label
                self._prev_label       = new_label
                self._consecutive_bars = self._pending_count
                self._pending_label    = None
                self._pending_count    = 0
                logger.warning(
                    "CAMBIO DE RÉGIMEN CONFIRMADO: %s → %s", old_label, new_label
                )

        is_confirmed = (
            self._prev_label == new_label
            and self._consecutive_bars >= self.stability_bars
        )
        return is_confirmed

    # ------------------------------------------------------------------
    # Métodos auxiliares
    # ------------------------------------------------------------------

    def predict_regime_proba(self) -> List[float]:
        """Distribución de probabilidad sobre todos los regímenes."""
        if self._log_alpha is None:
            raise RuntimeError("No hay inferencia previa. Llama predict_regime_filtered() primero.")
        return np.exp(self._log_alpha).tolist()

    def get_regime_stability(self) -> int:
        """Número de barras consecutivas en el régimen actual."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """Matriz de transición aprendida (n_regimes × n_regimes)."""
        if self.model is None:
            raise RuntimeError("Modelo no entrenado.")
        return self.model.transmat_.copy()

    def detect_regime_change(self) -> bool:
        """True si hay un cambio de régimen confirmado en el paso actual."""
        return (
            self._pending_count == 0
            and self._consecutive_bars == self.stability_bars
        )

    def is_flickering(self) -> bool:
        """True si el ratio de parpadeo supera el umbral en la ventana reciente."""
        if len(self._regime_history) < 2:
            return False
        changes = sum(
            1 for i in range(1, len(self._regime_history))
            if self._regime_history[i] != self._regime_history[i - 1]
        )
        return changes >= self.flicker_threshold

    def position_size_multiplier(self) -> float:
        """
        Retorna el multiplicador de tamaño de posición según estabilidad.
        - 1.0  si el régimen está confirmado.
        - 0.75 durante transición (pendiente de confirmación).
        - 0.50 en modo incertidumbre (flicker activo).
        """
        if self.is_flickering():
            return 0.50
        if self._pending_count > 0:
            return 0.75
        return 1.0

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Guarda el modelo y metadatos en disco via pickle."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        payload = {
            "model":         self.model,
            "n_regimes":     self.n_regimes,
            "regime_labels": self.regime_labels,
            "regime_order":  self.regime_order,
            "regime_infos":  self.regime_infos,
            "bic_score":     self.bic_score,
            "trained_at":    self.trained_at,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Modelo HMM guardado en %s", self.model_path)

    def load(self) -> bool:
        """
        Carga el modelo desde disco.
        Retorna True si se cargó exitosamente, False si no existe.
        """
        if not os.path.exists(self.model_path):
            logger.info("No se encontró modelo guardado en %s", self.model_path)
            return False
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self.model         = payload["model"]
        self.n_regimes     = payload["n_regimes"]
        self.regime_labels = payload["regime_labels"]
        self.regime_order  = payload["regime_order"]
        self.regime_infos  = payload["regime_infos"]
        self.bic_score     = payload["bic_score"]
        self.trained_at    = payload["trained_at"]
        logger.info(
            "Modelo HMM cargado: n_regimes=%d, BIC=%.2f, entrenado=%s",
            self.n_regimes, self.bic_score, self.trained_at.isoformat(),
        )
        return True

    def is_stale(self, max_age_days: int = 7) -> bool:
        """True si el modelo tiene más de max_age_days días desde su entrenamiento."""
        if self.trained_at is None:
            return True
        delta = datetime.utcnow() - self.trained_at
        return delta.days >= max_age_days
