# regime-trader — Whale Strategy

> **Bot de trading algorítmico para eToro** basado en detección de regímenes de volatilidad con Hidden Markov Models (HMM) e integración de la operativa Whale Capital (Ballenera + Corto Plazo).

---

## Tabla de Contenidos

1. [Filosofía del Sistema](#1-filosofía-del-sistema)
2. [Arquitectura Técnica](#2-arquitectura-técnica)
3. [Lógica de Puntos de Compra y Venta](#3-lógica-de-puntos-de-compra-y-venta)
4. [Instalación y Configuración](#4-instalación-y-configuración)
5. [Configuración YAML Detallada](#5-configuración-yaml-detallada)
6. [Despliegue Persistente 24/7](#6-despliegue-persistente-247)
7. [Referencia Completa de Comandos](#7-referencia-completa-de-comandos)
8. [Estructura del Proyecto](#8-estructura-del-proyecto)
9. [Preguntas Frecuentes](#9-preguntas-frecuentes)
10. [Disclaimer](#10-disclaimer)

---

## 1. Filosofía del Sistema

> **Gestión de riesgos > Generación de señales**

El sistema parte de una premisa central: en trading algorítmico, **sobrevivir a las drawdowns es más importante que maximizar el retorno**. Por eso el `RiskManager` posee **poder de veto absoluto** sobre cualquier señal. Ninguna orden llega al broker sin pasar las 16 validaciones de riesgo en serie.

El HMM **no predice precios**. Clasifica el régimen de volatilidad del mercado para determinar cuánto capital exponer en cada momento. La estrategia óptima depende del régimen actual, no del régimen futuro.

Sobre este motor se integra la **capa Whale Capital**, que aplica un filtro de Fuerza Relativa (RS) para identificar líderes de mercado y detecta patrones de acumulación institucional (Cup & Handle, Doble Suelo, Darvas Box) antes de generar señales de entrada. La capa Whale tiene **prioridad** sobre la lógica HMM pura: si detecta una oportunidad Ballenera con RS ≥ 1.05, la señal Whale se procesa primero.

---

## 2. Arquitectura Técnica

### 2.1 Flujo de Datos (cada ciclo de 30 segundos)

```
┌──────────────────────────────────────────────────────────────────┐
│                          FUENTES DE DATOS                        │
│                                                                  │
│  eToro REST API  ──► Precios bid/ask en vivo (cada 30s)          │
│  yfinance        ──► Velas OHLCV diarias históricas (1×día)      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                         │
│  14 indicadores normalizados con z-score rolling 252d:           │
│  ret_1d, vol_20, vol_60, rsi_14, macd_signal, bb_width,          │
│  ema50_slope, ema200_slope, atr_pct, volume_ratio,               │
│  donchian_pct, gap_pct, intraday_range, trend_strength           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        HMM ENGINE                                │
│  GaussianHMM con selección automática por BIC                    │
│  n_candidates: [3,4,5,6,7] → elige el de menor BIC              │
│  Algoritmo Forward (causal, sin look-ahead bias)                 │
│  Output: RegimeState {label, probability, is_confirmed}          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
        ┌──────────────────┐  ┌─────────────────────────┐
        │   CAPA WHALE     │  │    CAPA HMM (fallback)  │
        │  (prioridad)     │  │                         │
        │  RS Filter       │  │  StrategyOrchestrator   │
        │  PatternDetector │  │  ├── MACDFilter          │
        │  (C&H, DB,       │  │  └── DonchianBreakout   │
        │   Darvas, S/D,   │  │                         │
        │   Engulfing)     │  │                         │
        └────────┬─────────┘  └───────────┬─────────────┘
                 │                        │
                 └──────────┬─────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │    SIGNAL GENERATOR     │
              │  Signal {direction,     │
              │   confidence, entry,    │
              │   stop_loss, size_usd}  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   RISK MANAGER (veto)   │
              │   16 checks en serie    │
              │   ──────────────────    │
              │   0. Lock file          │
              │   1. Circuit Breaker    │
              │   2. Flicker HMM        │
              │   3. Leverage           │
              │   4. Stop loss válido   │
              │   5. Duplicado 60s      │
              │   6. Spread bid-ask     │
              │   7. Trades diarios     │
              │   8. Concurrent max     │
              │   9. Cash disponible    │
              │  10. Exposición total   │
              │  11. Cap 20% posición   │
              │  12. Mínimo $100        │
              │  13. Gap risk           │
              │  14. Correlación 85%    │
              │  15. Quality score      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    ORDER EXECUTOR       │
              │  dry_run=True → skip    │
              │  Envía stop obligatorio │
              │  UUID trace por trade   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   POSITION TRACKER      │
              │  Polling 30s via API    │
              │  state_snapshot.json    │
              │  Reconcilia ghosts      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  DASHBOARD + ALERTAS    │
              │  PySide6 GUI o Rich CLI │
              │  4 logs JSON rotativos  │
              │  Email + Webhook        │
              └─────────────────────────┘
```

### 2.2 Motor HMM (`core/hmm_engine.py`)

El `HMMEngine` es el núcleo de detección de mercado. Su funcionamiento es:

**Entrenamiento (BIC):**
```
Para cada n ∈ [3, 4, 5, 6, 7]:
    Entrenar GaussianHMM(n_components=n, covariance_type="full", n_init=15)
    Calcular BIC = -2 × log(L) + k × log(N)
    donde k = parámetros libres del modelo

Seleccionar modelo con BIC mínimo
Asignar etiquetas por volatilidad ascendente:
    ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"]  (si n=5)
```

**Inferencia en tiempo real (Forward Algorithm):**
```python
# En cada tick de 30 segundos:
regime_state = hmm.predict_regime_filtered(features[-1])
# → Sin look-ahead bias: usa solo datos hasta T
# → Filtro de estabilidad: requiere stability_bars=2 barras consecutivas
# → Flicker detection: alerta si >4 cambios en ventana de 20 barras
```

**Etiquetas de régimen y comportamiento:**

| Etiqueta | Volatilidad | Exposición Capital | Estrategia |
|---|---|---|---|
| `CRASH` | Extrema | 0% — sin trades | Sin operaciones |
| `STRONG_BEAR` | Muy alta | 0% — sin trades | Sin operaciones |
| `BEAR` | Alta | 80% | Defensiva |
| `NEUTRAL` | Media | 85–100% | MACD + Donchian |
| `BULL` | Media-baja | 100% | Whale + HMM |
| `STRONG_BULL` | Baja | 100% | Whale prioritario |
| `EUPHORIA` | Muy baja | 95% | Whale con salida vigilada |

**Validación OOS:** El motor realiza un holdout del 20% final de los datos de entrenamiento. Si `OOS_log_likelihood / IS_log_likelihood < 0.5`, emite warning de posible overfitting. No bloquea el modelo, pero queda registrado en logs.

### 2.3 Orquestador de Estrategias (`core/regime_strategies.py`)

El `StrategyOrchestrator` selecciona la estrategia correcta según el régimen:

```
RegimeState → StrategyOrchestrator → Strategy.generate_signal()
                                            │
                                     ┌──────┴──────┐
                                     │             │
                               MACDFilter   DonchianBreakoutFilter
                                     │             │
                               TechnicalConfirmation
                               (STRONG / MODERATE / WEAK)
```

**Filtro MACD:** Confirma momentum. MACD line > signal line = alcista.

**Filtro Donchian:** Confirma breakout. Precio actual > máximo de N barras (55d) = breakout válido.

**Combinación:** `STRONG` si ambos confirman; `MODERATE` si uno confirma; `WEAK` si ninguno pero el régimen lo permite. La capa Whale tiene filtros propios de volumen y patrón que reemplazan esta lógica cuando está activa.

---

## 3. Lógica de Puntos de Compra y Venta

### 3.1 Filtro de Fuerza Relativa (RS) — Primera Línea de Defensa

Antes de evaluar cualquier patrón, el sistema descarta rezagados:

```
RS(t, N) = (Precio_activo(t) / Precio_activo(t−N))
           ─────────────────────────────────────────
           (Precio_SPY(t) / Precio_SPY(t−N))

donde N = 252 barras (1 año bursátil)
```

| Resultado RS | Clasificación | Acción |
|---|---|---|
| RS ≥ 1.05 | **Market Leader** | Elegible para Ballenera |
| 0.75 < RS < 1.05 | Neutral | Elegible para Corto Plazo |
| RS ≤ 0.75 | **Rezagado** | **Descarte inmediato** — sin señal |

> El descarte de rezagados ocurre **antes** de calcular indicadores técnicos, ahorrando tiempo de cómputo y previniendo capital atrapado en activos débiles (concepto de coste de oportunidad).

### 3.2 Condiciones de Entrada — Estrategia Ballenera

La operativa Ballenera aplica a activos con 1–5 años en bolsa, capitalización $300M–$50B y RS ≥ 1.05.

#### Patrón Cup & Handle (Acumulación Institucional)

Estructura de 5 fases que tarda 3–18 meses en formarse:

```
Precio
  │
  │    ←── Pivote izquierdo (ATH previo)          ←── Pivote derecho ≥ 70% del pivote izq.
  │   /                                           /
  │  /          Handle (consolida en            /
  │ /           upper 70% del cup)          ___/
  │/            ──────────────────        /
  │                                 ────/
  │                    Cup
  └──────────────────────────────────────────────── Tiempo
```

**Algoritmo de detección:**
```python
# Condiciones verificadas en orden:
1. cup_depth = (pivot - cup_low) / pivot
   Válido si: 0.15 ≤ cup_depth ≤ 0.60

2. cup_duration: 30 ≤ barras ≤ 365

3. pivot_right ≥ 0.90 × pivot_left  (asimetría tolerable)

4. handle_floor = cup_low + 0.70 × (pivot - cup_low)
   Handle SOLO incluye barras por encima del floor
   (consolida en upper 70%, no cae al fondo del cup)

5. handle_depth ≤ 0.20 × (pivot - cup_low)
   (corrección máxima del handle: 20%)

6. Volumen en breakout ≥ 1.5× media 50 barras
```

**Entrada:** Cuando el precio cierra por encima del máximo del handle.
**Stop:** 5% por debajo de la entrada (máximo absoluto).
**Tamaño:** 20% del equity actual (interés compuesto — opera sobre equity real de ese momento).

#### Patrón Doble Suelo

```
# Condiciones:
1. Dos mínimos locales con diferencia < 5%
2. Separación ≥ 20 barras entre los mínimos
3. Neckline = máximo entre los dos mínimos
4. Entrada al cierre por encima de la neckline
5. Volumen ≥ 1.5× media en el día de breakout
```

#### Corrección RS + Trampolín

```
# Captura la inertia post-incubación:
1. RS durante corrección del índice (20 días): activo cae MENOS que SPY
2. rs_during_correction = (asset_ret_20d) / (index_ret_20d)
3. Si activo mantiene precio mientras mercado corrige → señal de fortaleza relativa
4. Entrada cuando el mercado rebota (activo ya lleva ventaja)
```

### 3.3 Condiciones de Entrada — Estrategia Corto Plazo

Aplica a activos en tendencia alcista (close > EMA50 > EMA200) con buen RS.

#### Darvas Box

```
# Consolidación lateral en tendencia alcista:
1. Rango del box ≤ 10% en las últimas N barras
2. Mínimo 10 barras de consolidación
3. La acción está en uptrend (price > EMA50 > EMA200)
4. Entrada al breakout del box high con volumen ≥ 1.5×
```

#### Compresión Oferta/Demanda

```
# Patrón de resistencia destruida:
1. Máximos horizontales (resistencia) ≥ 3 toques en niveles similares (<2%)
2. Mínimos ascendentes simultáneos
3. El comprador va eliminando la pared de vendedores
4. Breakout confirm: volumen ≥ 1.5× y cierre > resistencia
```

#### Vela Engulfing en Corrección

```
# El activo muestra fortaleza antes que el índice:
1. Índice SPY en caída (retorno 5 días < -1%)
2. Vela alcista engloba la vela bajista anterior del activo
3. Cuerpo del engulfing > 0.5% del precio
4. Confirma que el activo lidera la recuperación
```

### 3.4 Condiciones de Salida y Gestión de Posiciones

El bot supervisa **todas las posiciones abiertas**, incluyendo las que el usuario abrió manualmente. Si una posición activa cumple una señal de salida, el bot la cierra para proteger el capital, independientemente de si fue apertura manual o automática.

#### Señales de Salida

| Tipo | Condición | Urgencia | Acción |
|---|---|---|---|
| **Stop Loss** | Precio ≤ entry_price × (1 − 5%) | CRÍTICA | Cierre inmediato |
| **Cambio de Tendencia** | EMA50 cruza por debajo de EMA200 (death cross) | ALTA | Cierre en próximo tick |
| **Euforia / Climático** | Gap alcista ≥ 5% en ATH + volumen ≥ 2× media, después de uptrend ≥ 20% | ALTA | Venta en techo |
| **Distribución** | Lateral en ATH + gap bajista ≥ 3% + volumen ≥ 2× media | ALTA | Salida institucional detectada |
| **Régimen CRASH/STRONG_BEAR** | HMM clasifica régimen extremo | ALTA | Cierre preventivo |

#### Interés Compuesto en el Sizing

```python
# Cada nueva operación usa el equity ACTUAL (no el inicial):
position_usd = equity_actual × 0.20   # Ballenera: 20%
position_usd = equity_actual × 0.025  # Corto Plazo: 2.5%–15%

# compounding_factor = equity_actual / equity_inicial
# Si equity creció 10% → las posiciones nuevas son 10% más grandes
# Se propaga automáticamente en cada tick (update_equity cada 30s)
```

---

## 4. Instalación y Configuración

### 4.1 Requisitos Previos

- Python **3.9+** (se usa `zoneinfo` de stdlib para zonas horarias)
- Cuenta eToro con acceso a la API pública v1
- Credenciales: `ETORO_API_KEY` y `ETORO_USER_KEY`
- Conexión a internet estable (polling cada 30 segundos)

### 4.2 Instalación Local

```bash
# 1. Clonar el repositorio
git clone <repo-url> regime-trader
cd regime-trader

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. (Opcional) Dashboard gráfico PySide6
pip install PySide6
```

### 4.3 Configuración de Credenciales con `.env`

Crea el archivo `.env` en la raíz del proyecto:

```bash
# .env — NUNCA subir al control de versiones
ETORO_API_KEY=tu_api_key_aqui
ETORO_USER_KEY=tu_user_key_aqui
```

> ⚠️ **Seguridad crítica:** El archivo `.env` contiene credenciales reales. Verifica que `.gitignore` lo excluye antes de cualquier `git add`.

```bash
# Verificar que .env está ignorado:
cat .gitignore | grep .env
# Debe mostrar: .env
```

El proyecto carga automáticamente `.env` mediante `python-dotenv`. En servidores de producción, configura las variables directamente en el entorno del sistema operativo en lugar de usar un archivo `.env`.

### 4.4 Configurar `settings.yaml`

Edita `config/settings.yaml` con los valores de tu cuenta:

```yaml
broker:
  environment: "demo"          # Empieza SIEMPRE en "demo"
  expected_cid: TU_CID_AQUI    # Tu CID numérico de eToro

risk:
  initial_equity: TU_EQUITY    # Capital actual en USD
```

Para encontrar tu CID: en eToro → Configuración → API → tu `realCid` numérico.

### 4.5 Flujo de Arranque Recomendado

```bash
# Paso 1: Verificar tests
python -m pytest tests/ -v

# Paso 2: Entrenar modelo HMM
python main.py --train-only

# Paso 3: Validar con backtest
python main.py --backtest

# Paso 4: Dry-run (sin órdenes reales)
python main.py --dry-run

# Paso 5: Demo (órdenes en cuenta demo eToro)
# → settings.yaml: environment: "demo"
python main.py

# Paso 6: Real (después de mínimo 2 semanas en demo)
# → settings.yaml: environment: "real"
python main.py
# El bot pedirá escribir "CONFIRMO" antes de arrancar
```

---

## 5. Configuración YAML Detallada

### Sección `broker`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `environment` | string | `"demo"` \| `"real"`. En `"real"` exige confirmación por consola |
| `base_url` | string | URL base API eToro (no modificar) |
| `expected_cid` | int | CID numérico de tu cuenta (health check al arrancar) |
| `polling_interval_seconds` | int | Intervalo de sincronización con el broker (default: 30) |
| `active_instruments` | list[int] | IDs de instrumentos eToro a monitorear |
| `instrument_symbols` | dict | Mapeo `{instrument_id: yahoo_ticker}` para datos históricos |

```yaml
# Ejemplo de mapeo de instrumentos:
instrument_symbols:
  4238:  "VOO"     # Vanguard S&P 500 ETF
  14328: "MELI"    # MercadoLibre
  9408:  "PLTR"    # Palantir Technologies
  6218:  "APP"     # AppLovin Corp
  2488:  "DAVE"    # Dave Inc.
```

> **Nota técnica:** eToro no expone su endpoint de candles históricas en la API pública v1. El sistema descarga los datos históricos de **Yahoo Finance** vía `yfinance`, usando el mapeo `instrument_symbols`.

### Sección `hmm`

| Parámetro | Default | Descripción |
|---|---|---|
| `n_candidates` | [3,4,5,6,7] | Número de estados HMM a evaluar con BIC |
| `n_init` | 15 | Inicializaciones aleatorias por modelo (más = más estable) |
| `covariance_type` | `"full"` | Tipo de covarianza: `"full"` \| `"diag"` \| `"spherical"` |
| `stability_bars` | 2 | Barras consecutivas para confirmar cambio de régimen |
| `flicker_window` | 20 | Ventana para medir tasa de parpadeo (barras) |
| `flicker_threshold` | 4 | Cambios máximos en ventana antes de alerta de inestabilidad |
| `min_confidence` | 0.55 | Probabilidad mínima del estado para aceptar el régimen |
| `retrain_every_days` | 7 | Días entre re-entrenamientos automáticos (lunes) |
| `model_path` | `models/hmm_model.pkl` | Ruta del modelo serializado |

### Sección `strategy`

| Parámetro | Default | Descripción |
|---|---|---|
| `low_vol_allocation` | 1.0 | Exposición 100% en régimen BULL/EUPHORIA |
| `mid_vol_allocation_trend` | 1.0 | Exposición 100% en NEUTRAL con tendencia |
| `mid_vol_allocation_no_trend` | 0.85 | Exposición 85% en NEUTRAL sin tendencia |
| `high_vol_allocation` | 0.80 | Exposición 80% en BEAR (no cierra todo) |
| `rebalance_threshold` | 0.15 | Deriva máxima antes de rebalancear (15%) |
| `uncertainty_size_mult` | 0.85 | Reductor de tamaño en estado de incertidumbre |

### Sección `risk`

| Parámetro | Valor | Descripción |
|---|---|---|
| `initial_equity` | 560.05 | Capital base actual — actualizar si cambia el portafolio |
| `target_equity` | 655.37 | Objetivo (+20% en 120 días) |
| `max_risk_per_trade` | 0.01 | Riesgo máximo por operación (1% del equity) |
| `max_exposure` | 0.92 | Exposición total máxima (92% del equity) |
| `max_single_position` | 0.20 | Peso máximo por instrumento (20%) |
| `min_position_size_usd` | 100.0 | Mínimo de eToro por operación |
| `max_concurrent` | 5 | Posiciones simultáneas máximas |
| `max_daily_trades` | 10 | Operaciones máximas por día |
| `daily_dd_reduce` | 0.02 | Drawdown diario 2% → reduce tamaños 50% |
| `daily_dd_halt` | 0.03 | Drawdown diario 3% → pausa total |
| `weekly_dd_reduce` | 0.05 | Drawdown semanal 5% → reduce tamaños 50% |
| `weekly_dd_halt` | 0.07 | Drawdown semanal 7% → pausa total |
| `max_dd_from_peak` | 0.10 | Drawdown desde pico 10% → lock permanente |
| `default_stop_loss_pct` | 0.05 | Stop loss por defecto: 5% bajo entrada |

> **Capital calibrado:** Todos los umbrales monetarios se derivan de `initial_equity × pct`. Si el capital cambia, actualiza solo `risk.initial_equity` en `settings.yaml` — el resto se recalcula automáticamente.

### Sección `whale`

| Parámetro | Default | Descripción |
|---|---|---|
| `enabled` | true | Activa/desactiva la capa Whale completa |
| `reference_index` | `"SPY"` | Índice de referencia para calcular RS |
| `rs_window` | 252 | Ventana RS en barras (1 año bursátil) |
| `rs_leader_threshold` | 1.05 | RS mínimo para clasificar como Market Leader |
| `rs_laggard_threshold` | 0.75 | RS máximo antes de descarte inmediato |
| `max_stop_pct` | 0.05 | Stop loss absoluto máximo (5%) |
| `ballenera_position_pct` | 0.20 | Tamaño Ballenera: 20% del equity |
| `corto_plazo_min_pct` | 0.025 | Tamaño mínimo Corto Plazo |
| `corto_plazo_max_pct` | 0.15 | Tamaño máximo Corto Plazo |
| `cup_min_depth_pct` | 0.15 | Profundidad mínima del cup (15%) |
| `cup_max_depth_pct` | 0.60 | Profundidad máxima del cup (60%) |
| `volume_breakout_mult` | 1.5 | Volumen de breakout ≥ 1.5× media 50d |
| `volume_climactic_mult` | 2.0 | Volumen climático para señal de salida |
| `min_confidence` | 0.45 | Confianza mínima para emitir señal Whale |

### Sección `monitoring`

| Parámetro | Descripción |
|---|---|
| `dashboard_refresh_seconds` | Frecuencia de refresco del dashboard (30s) |
| `alert_rate_limit_minutes` | Cooldown entre alertas del mismo tipo (15 min) |
| `log_dir` | Directorio de logs (`logs/`) |
| `log_max_bytes` | Tamaño máximo por archivo de log (10 MB) |
| `log_backup_count` | Archivos de backup rotativo (30 ≈ 30 días) |

**Alertas opcionales** — agregar a `monitoring` si se desean:
```yaml
monitoring:
  email_to: "tu@email.com"
  email_from: "bot@tudominio.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  webhook_url: "https://discord.com/api/webhooks/..."  # Discord o Slack
```

---

## 6. Despliegue Persistente 24/7

El bot requiere ejecución continua para no perderse ciclos de mercado. A continuación se describen tres métodos de despliegue según el entorno.

### 6.1 Método A: PM2 (Node.js Process Manager) — Recomendado para VPS

PM2 es ideal para servidores Linux con acceso a Node.js. Gestiona reinicios automáticos, logs y monitoreo.

```bash
# 1. Instalar PM2 globalmente
npm install -g pm2

# 2. Crear archivo de configuración del proceso
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'regime-trader',
    script: 'main.py',
    interpreter: '/ruta/a/tu/.venv/bin/python',
    cwd: '/ruta/al/proyecto/regime-trader',
    env: {
      ETORO_API_KEY: 'tu_api_key',
      ETORO_USER_KEY: 'tu_user_key',
    },
    restart_delay: 5000,    // esperar 5s antes de reiniciar
    max_restarts: 10,       // máximo 10 reinicios por hora
    autorestart: true,
    watch: false,           // no recargar al cambiar archivos
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    error_file: 'logs/pm2-error.log',
    out_file: 'logs/pm2-out.log',
  }]
}
EOF

# 3. Iniciar el bot
pm2 start ecosystem.config.js

# 4. Guardar configuración para reinicio del servidor
pm2 save
pm2 startup     # genera el comando de systemd, ejecutarlo como indica

# Comandos útiles:
pm2 status                  # ver estado
pm2 logs regime-trader      # ver logs en tiempo real
pm2 restart regime-trader   # reiniciar
pm2 stop regime-trader      # detener
pm2 delete regime-trader    # eliminar proceso
```

### 6.2 Método B: systemd (Linux nativo) — Recomendado para producción robusta

`systemd` es el gestor de servicios nativo de Linux. Ideal para servidores de producción.

```bash
# 1. Crear el archivo de servicio
sudo nano /etc/systemd/system/regime-trader.service
```

```ini
[Unit]
Description=Regime Trader Bot — eToro Whale Strategy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tu_usuario
WorkingDirectory=/ruta/al/proyecto/regime-trader
Environment="ETORO_API_KEY=tu_api_key"
Environment="ETORO_USER_KEY=tu_user_key"
ExecStart=/ruta/al/.venv/bin/python main.py
Restart=on-failure
RestartSec=10s
StandardOutput=append:/ruta/al/proyecto/regime-trader/logs/service.log
StandardError=append:/ruta/al/proyecto/regime-trader/logs/service-error.log

[Install]
WantedBy=multi-user.target
```

```bash
# 2. Activar y arrancar el servicio
sudo systemctl daemon-reload
sudo systemctl enable regime-trader
sudo systemctl start regime-trader

# Comandos útiles:
sudo systemctl status regime-trader     # ver estado
sudo journalctl -u regime-trader -f     # logs en tiempo real
sudo systemctl restart regime-trader    # reiniciar
sudo systemctl stop regime-trader       # detener
```

### 6.3 Método C: Docker — Recomendado para portabilidad

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs models

CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  regime-trader:
    build: .
    restart: unless-stopped
    environment:
      - ETORO_API_KEY=${ETORO_API_KEY}
      - ETORO_USER_KEY=${ETORO_USER_KEY}
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./state_snapshot.json:/app/state_snapshot.json
    stdin_open: true    # necesario para confirmación "CONFIRMO" en entorno real
    tty: true
```

```bash
# Construir y arrancar
docker compose up -d

# Ver logs
docker compose logs -f regime-trader

# Detener
docker compose down
```

> ⚠️ **Nota Docker + entorno real:** El bot solicita `CONFIRMO` por consola al arrancar en `environment: "real"`. Con Docker necesitas `-it` o `stdin_open: true` en la primera ejecución, o cambiar el método de confirmación a variable de entorno.

### 6.4 Recomendaciones de Hardware para el Servidor

| Recurso | Mínimo | Recomendado |
|---|---|---|
| CPU | 1 vCore | 2 vCores |
| RAM | 512 MB | 1 GB |
| Disco | 5 GB | 10 GB |
| Red | 10 Mbps | 100 Mbps |
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |

Proveedores económicos para VPS: **DigitalOcean** (Droplet $6/mes), **Hetzner** (CX11 €4/mes), **Contabo**.

---

## 7. Referencia Completa de Comandos

### Comandos principales

| Comando | Descripción |
|---|---|
| `python main.py` | Arranca el bot en modo normal (usa `environment` de `settings.yaml`) |
| `python main.py --dry-run` | Loop completo sin enviar órdenes reales — para validación |
| `python main.py --train-only` | Solo entrena el modelo HMM y guarda `.pkl`, luego sale |
| `python main.py --backtest` | Ejecuta walk-forward backtest IS=252d / OOS=126d |
| `python main.py --stress-test` | Pruebas de estrés: crash injection, gap risk, regime scramble |
| `python main.py --compare` | Compara performance vs buy&hold, SMA200 y estrategia random |
| `python main.py --dashboard` | Muestra dashboard Rich sin arrancar el loop de trading |

### Comandos de testing

```bash
# Ejecutar toda la suite de tests
python -m pytest tests/ -v

# Test específico de no look-ahead bias (matemáticamente garantizado)
python -m pytest tests/test_look_ahead.py -v

# Tests de Circuit Breaker calibrados al equity actual
python -m pytest tests/test_risk_stress.py -v

# Test de flujo completo sin API (mocks)
python -m pytest tests/test_dry_run.py -v

# Tests de integración eToro (con mocks)
python -m pytest tests/test_etoro_integration.py -v

# Tests de recuperación desde snapshot
python -m pytest tests/test_error_recovery.py -v

# Test único por nombre
python -m pytest tests/test_risk_stress.py::TestCircuitBreaker::test_daily_halt_triggered_at_3pct -v

# Con cobertura
python -m pytest tests/ --cov=core --cov=broker --cov=data -v
```

### Comandos de mantenimiento

```bash
# Ver logs en tiempo real (requiere tail en Unix o PowerShell)
# Windows PowerShell:
Get-Content logs/main.log -Wait -Tail 50

# Linux / macOS:
tail -f logs/main.log

# Logs de operaciones únicamente:
tail -f logs/trades.log

# Logs de alertas:
tail -f logs/alerts.log

# Logs de cambios de régimen HMM:
tail -f logs/regime.log

# Eliminar modelo HMM para reentrenamiento limpio
del models\hmm_model.pkl            # Windows
rm models/hmm_model.pkl             # Linux/macOS

# Reanudar bot tras peak drawdown lockout (requiere intervención manual)
del trading_halted.lock             # Windows
rm trading_halted.lock              # Linux/macOS
```

### Motivos de rechazo del RiskManager

| Check | Código | Descripción |
|---|---|---|
| 0 | `LOCK_FILE` | Archivo `trading_halted.lock` presente — intervención manual requerida |
| 1 | `CIRCUIT_BREAKER` | Drawdown superó umbral diario/semanal/peak |
| 2 | `FLICKER` | HMM inestable — demasiados cambios de régimen en ventana reciente |
| 3 | `LEVERAGE` | Señal pide leverage > 1 (eToro siempre 1×) |
| 4 | `STOP_LOSS_AUSENTE` | Sin `stopLossRate` válido — obligatorio en eToro |
| 5 | `DUPLICADO_60S` | Mismo instrumento solicitado en < 60 segundos |
| 6 | `SPREAD_EXCESIVO` | Spread bid-ask > 0.5% (baja liquidez) |
| 7 | `MAX_DAILY_TRADES` | Límite de 10 operaciones diarias alcanzado |
| 8 | `MAX_CONCURRENT` | 5 posiciones simultáneas abiertas |
| 9 | `CASH_INSUFICIENTE` | Saldo libre < $100 |
| 10 | `MAX_EXPOSURE` | Exposición total superaría 92% del equity |
| 11 | `CAP_20PCT` | Posición superaría 20% del equity |
| 12 | `MIN_100USD` | Tamaño calculado < $100 (mínimo eToro) |
| 13 | `GAP_RISK` | Gap overnight > 5% detectado en el instrumento |
| 14 | `CORRELACION` | Correlación > 85% con posición existente |
| 15 | `QUALITY` | Score de calidad de señal demasiado bajo |

---

## 8. Estructura del Proyecto

```
regime-trader/
│
├── config/
│   └── settings.yaml              # Configuración central — única fuente de verdad
│
├── core/
│   ├── hmm_engine.py              # Motor HMM: BIC, Forward algorithm, estabilidad
│   ├── regime_strategies.py       # Signal dataclass, MACD, Donchian, StrategyOrchestrator
│   ├── signal_generator.py        # Capa Whale (prioridad) + HMM fallback
│   ├── risk_manager.py            # CircuitBreaker, 16 validaciones, PortfolioState
│   ├── ballenero_strategy.py      # BalleneroOrchestrator: Ballenera + Corto Plazo
│   ├── pattern_detector.py        # Cup&Handle, Doble Suelo, Darvas, S/D, Engulfing
│   └── whale_filters.py           # RS filter, TrendFilter, VolumeAnalyzer
│
├── broker/
│   ├── etoro_client.py            # Cliente REST eToro v1: retry exponencial, health check
│   ├── order_executor.py          # Ejecución con dry_run, stop adjustment (close+reopen)
│   └── position_tracker.py        # Polling 30s, state_snapshot.json, reconciliación
│
├── data/
│   ├── feature_engineering.py     # 14 features OHLCV normalizados con z-score 252d
│   └── market_data.py             # Candles yfinance, rates live, spread, índice SPY
│
├── backtest/
│   ├── backtester.py              # Walk-forward IS=252d / OOS=126d
│   ├── performance.py             # Sharpe, Sortino, Calmar, vs benchmarks
│   └── stress_test.py             # Crash injection, gap risk, regime scramble
│
├── monitoring/
│   ├── logger.py                  # 4 logs JSON rotativos: main/trades/alerts/regime
│   ├── alerts.py                  # AlertManager: 10 tipos, rate-limited, email+webhook
│   ├── dashboard.py               # Dashboard Rich CLI (run_live: 5s refresh)
│   ├── ui_manager.py              # GUI PySide6: DataBridge, cards, DashboardApp
│   ├── styles.qss                 # Qt Style Sheets: paleta dark blue-purple
│   └── MICROCOPY_GUIDE.md         # Guía UX: traducciones de etiquetas HMM
│
├── tests/
│   ├── conftest.py                # Fixtures: credenciales fake, equity mock
│   ├── test_look_ahead.py         # Verificación matemática sin look-ahead bias
│   ├── test_etoro_integration.py  # Tests API con mocks HTTP
│   ├── test_dry_run.py            # Flujo E2E completo sin órdenes reales
│   ├── test_risk_stress.py        # Circuit Breaker calibrado a $560.05
│   └── test_error_recovery.py     # Recuperación desde snapshot y settings
│
├── models/
│   └── hmm_model.pkl              # Modelo HMM entrenado (generado por --train-only)
│
├── logs/                          # Archivos de log rotativos (generados en runtime)
│   ├── main.log
│   ├── trades.log
│   ├── alerts.log
│   └── regime.log
│
├── state_snapshot.json            # Snapshot del estado al apagar (auto-generado)
├── main.py                        # Entry point: CLI, startup, MainLoop
└── requirements.txt
```

---

## 9. Preguntas Frecuentes

### ¿Por qué usa el algoritmo Forward en lugar de Viterbi?

El algoritmo de Viterbi encuentra la secuencia de estados más probable globalmente, lo que **requiere ver toda la secuencia hasta el final**. Esto introduce look-ahead bias: el régimen en `T` cambiaría si añades datos de `T+100`, lo que no es reproducible en tiempo real.

El **algoritmo Forward** computa la probabilidad marginal de cada estado en `T` dado únicamente las observaciones `0..T`. Es causal y matemáticamente verificado por `test_look_ahead.py`:

```
Invariante: predict(data[0:T]) == predict(data[0:T+100]) para todo T
```

### ¿Por qué no funciona WebSocket para actualizaciones en tiempo real?

eToro no expone WebSocket en su API pública v1. Todo el estado del portafolio proviene de polling REST cada 30 segundos. Por eso `PositionTracker` usa `schedule` para disparar `_sync_positions()` periódicamente.

### ¿Qué pasa si el bot se apaga inesperadamente?

Al recibir `SIGINT` o `SIGTERM`, el bot ejecuta `_shutdown()` que guarda `state_snapshot.json`. Al reiniciar, `load_snapshot()` restaura el estado interno (`_peak_equity`, `_entry_regimes`, `_last_order_ts`) y reconcilia contra la API live para detectar posiciones cerradas o abiertas externamente.

### ¿Cómo reanudar el bot después de un lockout por peak drawdown?

```bash
# El lock se activa automáticamente cuando DD desde pico > 10%
# Para reanudar (solo después de revisar y entender la causa):
rm trading_halted.lock     # Linux/macOS
del trading_halted.lock    # Windows
```

### ¿Qué hacer si hay un error al cargar el modelo HMM?

```bash
# Eliminar el modelo existente y reentrenar desde cero:
rm models/hmm_model.pkl
python main.py --train-only
```

### ¿Por qué las velas históricas no vienen de eToro?

El endpoint `/history/candles/*` de eToro retorna 404 en la API pública v1. El sistema descarga las velas de **Yahoo Finance** vía `yfinance`, usando el mapeo `broker.instrument_symbols` en `settings.yaml`.

### ¿Cómo añadir nuevos instrumentos?

1. En `settings.yaml → broker.active_instruments`, añadir el instrument ID de eToro.
2. En `settings.yaml → broker.instrument_symbols`, añadir el mapeo `{id: "TICKER_YAHOO"}`.
3. Si es elegible para Ballenera (1–5 años en bolsa, cap $300M–$50B), añadirlo también en `whale.target_instruments`.
4. Reiniciar el bot. El HMM se re-entrenará automáticamente si el modelo es más antiguo que `retrain_every_days`.

---

## 10. Disclaimer

Este software se proporciona **únicamente con fines educativos y de investigación**. No constituye asesoramiento financiero ni garantía de beneficios.

- El trading con instrumentos financieros implica **riesgo de pérdida del capital invertido**.
- Los resultados pasados del backtest **no garantizan resultados futuros**.
- Se recomienda **encarecidamente** operar en cuenta Demo durante al menos 30 días antes de usar capital real.
- El autor no se hace responsable de pérdidas derivadas del uso de este software.

**Usa siempre paper trading primero. Gestiona el riesgo ante todo.**
