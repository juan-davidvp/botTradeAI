# regime-trader

Bot de trading algorítmico para eToro basado en detección de regímenes de volatilidad con Hidden Markov Models (HMM).

---

## Filosofía

> **Gestión de riesgos > Generación de señales**

El sistema parte de una premisa clara: en trading algorítmico, sobrevivir a las drawdowns es más importante que maximizar el retorno. Por eso el `RiskManager` tiene **poder de veto absoluto** sobre cualquier señal. Ninguna orden llega al broker sin pasar por las 16 validaciones de riesgo en serie.

El HMM no predice precios. Clasifica el régimen de volatilidad del mercado para saber cuánto capital exponer. La estrategia correcta depende del régimen actual, no del régimen futuro.

---

## Diagrama de Arquitectura

```
Datos eToro (candles diarias)
        │
        ▼
Feature Engineering (14 indicadores)
        │
        ▼
HMM Engine ──► Régimen de Volatilidad
   (BIC)             │
                     ▼
              Rango Vol (low/mid/high)
                     │
                     ▼
              Strategy Orchestrator
              (MACD + Donchian + HMM)
                     │
                     ▼
              Signal Generator
                     │
                     ▼
              Risk Manager (16 checks) ──► VETO
                     │
                     ▼
              Order Executor (eToro API)
                     │
                     ▼
              Position Tracker (30s polling)
                     │
                     ▼
              Dashboard + Alertas + Logs
```

---

## Inicio Rápido (Quick Start)

### Paso 1 — Clonar y crear entorno virtual

```bash
git clone <repo-url> regime-trader
cd regime-trader
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Paso 2 — Configurar credenciales

```bash
# Windows CMD
set ETORO_API_KEY=tu_api_key_aqui
set ETORO_USER_KEY=tu_user_key_aqui

# Windows PowerShell
$env:ETORO_API_KEY = "tu_api_key_aqui"
$env:ETORO_USER_KEY = "tu_user_key_aqui"

# Linux / macOS
export ETORO_API_KEY=tu_api_key_aqui
export ETORO_USER_KEY=tu_user_key_aqui
```

> **Recomendación**: crea un archivo `.env` y usa `python-dotenv` para cargar las variables automáticamente.

### Paso 3 — Revisar settings.yaml

Verifica que los valores en `config/settings.yaml` coincidan con tu cuenta:

```yaml
broker:
  environment: "demo"       # Empieza SIEMPRE en demo
  expected_cid: 34044505    # Tu CID real de eToro

risk:
  initial_equity: 546.14   # Tu capital inicial
```

### Paso 4 — Ejecutar el backtest

```bash
python main.py --backtest
```

Revisa los resultados en `backtest_results/`. El bot solo pasa a producción si el OOS mensual promedio ≥ 4.66%.

### Paso 5 — Dry run (sin órdenes reales)

```bash
python main.py --dry-run
```

Verifica en la consola que:
- El HMM se entrena correctamente
- Las señales se generan y pasan el risk check
- Las 4 posiciones sin stop aparecen en las alertas

### Paso 6 — Arranque en entorno Demo/Real

```bash
# Demo (recomendado primero)
python main.py

# Real (pide confirmación explícita por consola)
# Cambiar en settings.yaml: environment: "real"
python main.py
```

---

## Referencia de CLI

| Comando | Descripción |
|---------|-------------|
| `python main.py` | Arranca el bot en modo normal (según `environment` en settings) |
| `python main.py --dry-run` | Ejecuta el loop sin enviar órdenes reales |
| `python main.py --backtest` | Ejecuta walk-forward backtest y guarda resultados |
| `python main.py --train-only` | Solo entrena el modelo HMM sin arrancar el loop |
| `python main.py --stress-test` | Ejecuta crash injection, gap risk y regime scramble |
| `python main.py --compare` | Compara performance vs benchmarks (buy&hold, SMA200, random) |
| `python main.py --dashboard` | Muestra dashboard Rich sin arrancar el loop de trading |

---

## Guía de Configuración (`config/settings.yaml`)

### Sección `broker`

| Parámetro | Descripción |
|-----------|-------------|
| `environment` | `"demo"` o `"real"`. En real pide confirmación por consola |
| `base_url` | URL base de la API de eToro (no modificar) |
| `expected_cid` | CID numérico de tu cuenta eToro (health check) |
| `polling_interval_seconds` | Frecuencia de sincronización de posiciones (default: 30) |
| `active_instruments` | Lista de instrument IDs de eToro a monitorear |
| `timeframe` | `"1Day"` — velas diarias para el HMM |

### Sección `hmm`

| Parámetro | Descripción |
|-----------|-------------|
| `n_candidates` | Números de estados a evaluar con BIC (ej. [3,4,5,6,7]) |
| `n_init` | Inicializaciones aleatorias por candidato (mayor = más estable, más lento) |
| `stability_bars` | Barras consecutivas para confirmar un cambio de régimen |
| `flicker_window` | Ventana (en barras) para calcular la tasa de parpadeo |
| `flicker_threshold` | Cambios máximos permitidos en la ventana antes de alerta |
| `min_confidence` | Probabilidad mínima del HMM para considerar el régimen válido |
| `retrain_every_days` | Días entre re-entrenamientos del HMM |
| `model_path` | Ruta del modelo pickle serializado |

### Sección `strategy`

| Parámetro | Descripción |
|-----------|-------------|
| `low_vol_allocation` | Fracción del equity en régimen de baja volatilidad (0.95 = 95%) |
| `mid_vol_allocation_trend` | Fracción en vol media con tendencia confirmada |
| `mid_vol_allocation_no_trend` | Fracción en vol media sin tendencia |
| `high_vol_allocation` | Fracción en régimen de alta volatilidad |
| `rebalance_threshold` | Desviación mínima para disparar un rebalanceo (0.10 = 10%) |
| `uncertainty_size_mult` | Multiplicador de tamaño en período de incertidumbre de régimen |

### Sección `risk`

| Parámetro | Valor calibrado | Descripción |
|-----------|----------------|-------------|
| `initial_equity` | 546.14 | Capital base del portafolio |
| `target_equity` | 655.37 | Objetivo (+20% en 120 días) |
| `max_risk_per_trade` | 0.01 | Riesgo máximo por operación (1% = $5.46) |
| `max_exposure` | 0.92 | Exposición total máxima (92% = $502.45) |
| `max_single_position` | 0.20 | Peso máximo por instrumento (20% = $109.23) |
| `min_position_size_usd` | 100.0 | Mínimo requerido por eToro |
| `max_concurrent` | 5 | Posiciones simultáneas máximas |
| `daily_dd_halt` | 0.03 | Circuit breaker diario (3% = $16.38) |
| `weekly_dd_halt` | 0.07 | Circuit breaker semanal (7% = $38.23) |
| `max_dd_from_peak` | 0.10 | Peak DD máximo antes de lock (10% = $54.61) |
| `default_stop_loss_pct` | 0.05 | Stop loss por defecto (5% bajo apertura) |
| `urgent_stops` | (5 entradas) | Stop loss urgentes para posiciones sin stop activo |

### Sección `monitoring`

| Parámetro | Descripción |
|-----------|-------------|
| `dashboard_refresh_seconds` | Frecuencia de refresco del dashboard Rich (5s en Live) |
| `alert_rate_limit_minutes` | Cooldown entre alertas del mismo tipo (15 min) |
| `log_dir` | Directorio de archivos de log (`logs/`) |
| `log_max_bytes` | Tamaño máximo por archivo de log (10 MB) |
| `log_backup_count` | Archivos de backup rotativo (30 ≈ 30 días) |
| `email_to` | Destinatario para alertas por email (opcional) |
| `webhook_url` | URL de Discord/Slack para webhooks (opcional) |

---

## Preguntas Frecuentes (FAQ)

### ¿Por qué usa el algoritmo Forward en lugar de Viterbi?

El algoritmo de Viterbi encuentra la secuencia de estados más probable globalmente, lo cual requiere ver **toda la secuencia hasta el final**. Esto introduce sesgo de anticipación (look-ahead bias): el régimen en `T` cambiaría si añades datos de `T+100`.

El **algoritmo Forward** computa la probabilidad marginal de cada estado en `T` dado únicamente las observaciones `0..T`. Es causal y replicable en tiempo real. El test `test_look_ahead.py` verifica matemáticamente esta propiedad.

### ¿Cómo elige el HMM el número de estados?

Se entrena un modelo independiente para cada valor de `n_candidates` (por defecto [3, 4, 5, 6, 7]). El mejor modelo se selecciona por **BIC mínimo**:

```
BIC = -2 × log(L) × N + k × log(N)
```

donde `k` es el número de parámetros libres (depende del tipo de covarianza y número de estados) y `N` es el número de observaciones. Menor BIC = mejor balance entre ajuste y complejidad.

### ¿Por qué se rechazan algunas operaciones?

El `RiskManager` aplica 16 validaciones en serie. Las causas más comunes de rechazo:

| Motivo | Descripción |
|--------|-------------|
| `STOP_LOSS_AUSENTE` | La señal no tiene `stopLossRate` válido (obligatorio en eToro) |
| `CIRCUIT_BREAKER` | Drawdown diario/semanal/peak superó el umbral; trading pausado |
| `DUPLICADO_60S` | Mismo instrumento solicitado en menos de 60 segundos |
| `CASH_INSUFICIENTE` | Saldo libre menor que el mínimo de $100 |
| `MAX_CONCURRENT` | Ya hay 5 posiciones abiertas (límite configurado) |
| `CORRELACION_EXCESIVA` | El instrumento correlaciona >85% con una posición existente |
| `SPREAD_EXCESIVO` | Spread bid-ask mayor al 0.5% (eToro en horas de baja liquidez) |
| `LOCK_FILE_ACTIVO` | Peak DD >10% — requiere intervención manual para reanudar |

### ¿Cómo activar el trading en vivo (cuenta real)?

1. Verificar que has probado en **Demo** al menos 2 semanas.
2. Verificar que el backtest OOS cumple el target (≥4.66%/mes promedio).
3. En `config/settings.yaml`, cambiar `environment: "real"`.
4. Al arrancar, el bot pedirá confirmación explícita por consola: escribir `CONFIRMO`.
5. **Nunca** poner credenciales de la cuenta real en variables de entorno de un servidor compartido.

---

## Disclaimer

Este software se proporciona **únicamente con fines educativos y de investigación**. No constituye asesoramiento financiero ni garantía de beneficios.

- El trading con instrumentos financieros implica **riesgo de pérdida del capital invertido**.
- Los resultados pasados del backtest **no garantizan resultados futuros**.
- Se recomienda **encarecidamente** operar en cuenta Demo durante al menos 30 días antes de usar capital real.
- El autor no se hace responsable de pérdidas derivadas del uso de este software.

**Usa siempre paper trading primero. Gestiona el riesgo ante todo.**

---

## Estructura del Proyecto

```
regime-trader/
├── config/
│   └── settings.yaml          # Configuración central
├── core/
│   ├── hmm_engine.py          # Motor HMM + algoritmo Forward
│   ├── regime_strategies.py   # Estrategias por régimen + MACD + Donchian
│   ├── signal_generator.py    # Generación de señales por instrumento
│   └── risk_manager.py        # 16 validaciones + Circuit Breakers
├── broker/
│   ├── etoro_client.py        # Cliente REST eToro (retry + backoff)
│   ├── order_executor.py      # Ejecución de órdenes + dry_run
│   └── position_tracker.py    # Polling 30s + snapshot JSON
├── data/
│   ├── feature_engineering.py # 14 features normalizados
│   └── market_data.py         # Candles + rates + spread
├── backtest/
│   ├── backtester.py          # Walk-forward IS=252d / OOS=126d
│   ├── performance.py         # Métricas + benchmarks
│   └── stress_test.py         # Crash injection + gap risk + regime scramble
├── monitoring/
│   ├── logger.py              # JSON logs rotativos (4 archivos)
│   ├── alerts.py              # Alertas + email + webhook
│   └── dashboard.py           # Dashboard Rich (Live 5s)
├── tests/
│   ├── conftest.py
│   ├── test_look_ahead.py     # Verificación ausencia de look-ahead bias
│   ├── test_etoro_integration.py  # Tests API con mocks
│   ├── test_dry_run.py        # Flujo E2E sin órdenes reales
│   ├── test_risk_stress.py    # Estrés calibrado a $546.14
│   └── test_error_recovery.py # Recuperación desde snapshot
├── models/                    # Modelos HMM entrenados (.pkl)
├── logs/                      # Archivos de log rotativos
├── main.py                    # Entry point + CLI
└── requirements.txt
```

---

## Dependencias principales

```
hmmlearn>=0.3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
ta>=0.11
requests>=2.31
rich>=13.0
schedule>=1.2
pyyaml>=6.0
pytest>=8.0
```
