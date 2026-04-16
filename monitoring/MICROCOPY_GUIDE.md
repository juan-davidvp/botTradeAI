# Guía de Microcopy — AgentBotTrade Dashboard

## Principio rector

> El usuario no necesita entender el algoritmo. Necesita saber **si su dinero está seguro** y **si el bot está haciendo algo por él**.

Cada término técnico fue reemplazado por una frase que responde a una de estas tres preguntas emocionales: ¿Estoy ganando? ¿Estoy protegido? ¿El bot está trabajando?

---

## Traducciones de términos clave

### 1. Estado del mercado (HMM Regimes)

| Término técnico | Copy en la UI | Justificación |
|---|---|---|
| `STRONG_BULL` | "Impulso alcista fuerte — El bot opera activamente" | "Alcista fuerte" es comprensible para cualquier inversor básico. El subtítulo confirma que el bot está actuando. |
| `EUPHORIA` | "Mercado en zona de euforia — El bot opera con precaución" | "Euforia" es un concepto que los medios financieros popularizaron (burbuja). La "precaución" comunica que el bot no se deja llevar. |
| `WEAK_BULL` | "Tendencia alcista moderada — El bot busca oportunidades" | "Moderada" reduce la expectativa y evita frustración si no hay trades. |
| `NEUTRAL` | "Mercado sin dirección clara — El bot espera una señal más fuerte" | Explica *por qué* no hay acción, eliminando la angustia del silencio. |
| `BEAR` | "Mercado bajista — El bot reduce la exposición" | "Exposición" es más claro que "riesgo" para un usuario semi-experto. |
| `STRONG_BEAR` | "Caída fuerte — El bot protege tu capital" | La posesión "tu capital" genera confianza. El bot trabaja *para* el usuario. |
| `CRASH` | "Caída extrema — El bot ha pausado operaciones automáticamente" | "Automáticamente" tranquiliza: es un feature, no un bug. |

---

### 2. Gestión de riesgo (Drawdown)

| Término técnico | Copy en la UI | Justificación |
|---|---|---|
| `Drawdown diario: $X` | "Caída de hoy: Protección activa: -$X" | "Protección activa" invierte el sentido: el límite *protege*, no amenaza. |
| `Drawdown semanal: $X` | "Caída semanal: Atención: -$X" | "Atención" escala gradualmente la alarma sin entrar en pánico. |
| `DD from peak: $X` | "Caída máxima histórica: Alerta: -$X" | El histórico necesita el nivel "Alerta" para distinguirse del diario. |
| `DD Limit: $16.38 (3%)` | "Límite: $16.38 (3%)" | El porcentaje acompañado del monto evita que el usuario haga la matemática. |

---

### 3. Circuit Breaker

| Estado técnico | Copy en la UI | Justificación |
|---|---|---|
| `OK` | "Operación normal" | Simple. No hay que confirmar lo que ya funciona; basta con nombrarlo. |
| `REDUCE_50` | "Riesgo reducido al 50% preventivamente" | "Preventivamente" es clave: no es que algo salió mal, sino que el bot actúa antes. |
| `HALT` | "Pausa de seguridad activada" | "Pausa" es reversible; "parada" suena a emergencia. |
| `LOCKED` | "Bloqueado — requiere intervención manual" | La acción requerida (intervención) está explícita; el usuario sabe qué hacer. |

---

### 4. Estado de confirmación del régimen

| Estado técnico | Copy en la UI | Justificación |
|---|---|---|
| `is_confirmed = True` | "✓ Señal confirmada — el bot puede operar" | El checkmark refuerza visualmente la certeza. |
| `is_confirmed = False` | "⏳ Señal en validación…" | Los tres puntos suspensivos indican proceso activo, no error. |

---

### 5. Sección de posiciones

| Columna técnica | Nombre en la UI | Justificación |
|---|---|---|
| `instrumentID` | "Instrumento" | Traduce el ID numérico al nombre del activo cuando es posible (ej: "VOO (S&P 500)"). |
| `openRate` | "Precio entrada" | "Entrada" es el término que usa cualquier plataforma de retail trading. |
| `current_price` | "Precio actual" | Directo. No hay mejor alternativa. |
| `P&L%` | "Rendimiento" | "Rendimiento" (+/-) es menos técnico que "P&L" (Profit & Loss). |
| `stopLossRate` | "Stop loss" | Mantenido en inglés: es el único término que el usuario básico ya conoce por nombre. |
| `isNoStopLoss` | "⚠ Sin protección" | El ícono de advertencia más el texto directo comunican urgencia sin términos de trading. |
| `holding_days` | "Días abierto" | Natural. "Holding" es anglicismo innecesario en este contexto. |

---

## Principios de color semántico aplicados

| Color | Significado | Casos de uso |
|---|---|---|
| `#10B981` (verde) | Capital seguro, progreso | Ganancias, circuit breaker OK, régimen alcista |
| `#F59E0B` (ámbar) | Atención preventiva | Drawdown moderado, REDUCE_50, señal no confirmada |
| `#EF4444` (rojo) | Acción requerida | Drawdown crítico, sin stop loss, HALT/LOCKED |
| `#448AFF` (azul) | Información, neutralidad | Valores base, labels de sección |
| `#7B8BB2` (gris-azul) | Datos secundarios | Fechas, unidades, contexto |

---

## Regla de oro para futuros copys

**Antes de escribir un término técnico, responder:**
1. ¿El usuario puede leer esto y saber si debe hacer algo?
2. ¿El tono genera confianza o ansiedad innecesaria?
3. ¿Podría confundirse con otro término similar?

Si alguna respuesta es negativa, reescribir.
