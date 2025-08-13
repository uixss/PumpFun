# PumpFun


## 🔍 Casos de uso típicos
- **Descubrimiento temprano**: enterarte cuando surgen tokens nuevos con señales mínimas de tracción.
- **Prioridad operativa**: enfocar tu tiempo en lo que supera tus umbrales (market cap, crecimiento, sentimiento).
- **Contexto express**: un resumen digerible con links directos a la operación/visualización.
- **Histórico básico**: mantener un rastro de qué apareció y cuándo, para revisión posterior.

---

## 🚀 Características ampliadas
1. **Escucha en tiempo real** (WebSocket): nuevos tokens conforme salen.
2. **Enriquecimiento de contexto**:
   - Precio de SOL desde dos fuentes (intenta CoinGecko y, si falla, Jupiter).
   - Market cap aproximado **en SOL y USD**.
   - Sentimiento a partir de la descripción (positivo/neutral/negativo).
   - Tendencias rápidas: top por cap, crecimientos relevantes, palabras frecuentes.
3. **Mensajería a Telegram**:
   - **Autocorte** de mensajes largos (evita que Telegram rechace envíos).
   - **Botones de acción**: _Trading en Pump.Fun_ y _Ver en Dexscreener_.
4. **Resiliencia** básica:
   - Reintentos ante desconexiones de WS.
   - Cacheo de precio SOL con TTL.
   - Modo SSL “relajado” opcional para redes con inspección.
5. **Bajo mantenimiento**:
   - Sin infraestructura compleja: corre en una PC o servidor simple.
   - Sin base de datos obligatoria; la info clave viaja por Telegram.

---

## 🧭 Estrategia de monitoreo (sugerida)
### 1) Objetivos
- **Cobertura**: enterarse de lo relevante sin convertirse en “ruido blanco”.
- **Prioridad**: resaltar señales que valen tu atención en ese momento.
- **Toma de acción**: con un clic estás en la página de trading/visualización.

### 2) Señales mínimas para “mirar”
- **Market cap (SOL)**: mayor a un umbral (p. ej. **≥ 50 SOL**).
- **Crecimiento relativo** hacia arriba en las últimas mediciones (p. ej. **> +10%** vs. registro previo).
- **Sentimiento** no negativo (neutral/positivo).
- **Descripción no vacía** y sin “palabras vacías” (mejora la señal).

> Estos umbrales se adaptan a tu apetito de riesgo. Empezá **alto** (menos ruido) y ajustá.

### 3) Niveles de alerta
- **Info** 🟡: cumple 1–2 señales (p. ej., cap aceptable **o** sentimiento ok). _Acción_: “observar”.
- **Watchlist** 🟠: cumple 2–3 señales (cap + crecimiento **o** cap + sentimiento). _Acción_: revisar ficha.
- **Action** 🔴: cumple 3–4 señales (cap + crecimiento + sentimiento + keywords útiles). _Acción_: evaluar trade.

### 4) Triaging (qué mirar primero)
1. **Action** 🔴 arriba del todo.
2. **Watchlist** 🟠 con cap más alto primero.
3. Recién después los **Info** 🟡 si tenés tiempo.

---

## 🧰 Tácticas para reducir ruido
- **Umbrales dinámicos** según franja horaria o volatilidad (subir cap mínimo en horas calientes).
- **Cooldown** por token (ej.: no repetir alerta del mismo token en menos de 30–60 min).
- **Stopwords ampliadas** (lista de palabras a ignorar en descripciones — evita marketing genérico).
- **Whitelist opcional** de autores/comunidades que te interesen más.
- **Límites de mensajes por hora** para evitar spam (p. ej. tope “suave” de 10–15).

---

## 📏 KPIs recomendados
- **Tiempo detección → alerta** (latencia): objetivo < **10–20 s**.
- **% de tokens con “acción” que revisás** (engagement).
- **Falsos positivos** percibidos vs. útiles (calidad de señal).
- **Top X tokens** por cap/crecimiento en el día (resumen ejecutivo diario).

---

## 🧪 Checklist operativo (día 1)
1. Validar **Telegram** (que llegan mensajes al chat correcto).
2. Definir **umbrales** iniciales (cap mínimo, crecimiento, sentimiento).
3. Probar **botones**: abren correctamente Pump.Fun y Dexscreener.
4. Forzar **cortes de red** breves (ver que reconecta solo).
5. Ajustar frecuencia de mensajes si hay saturación.

---

## 🧭 Playbook de decisión (ejemplo)
- **Action 🔴**: abrir link, revisar liquidez, holders, velocidad de cambio, comunidad. Si match con tu “perfil”, _entry_ pequeña y stop por tiempo; si no, watch.
- **Watchlist 🟠**: dejar alerta “mental” o apuntar en tu plan del día; si repite señal, sube de prioridad.
- **Info 🟡**: ignorar salvo que el tema/keyword sea de tu tesis.

> Recomendación general: **no perseguir FOMO**. Priorizá procesos repetibles, no impulsos.

---

## 🧱 Limitaciones (claras y honestas)
- **No** es señal de compra/venta, es un **radar**.
- Market cap en USD es **aproximado** (depende de precio SOL y datos públicos).
- El sentimiento de texto es **superficial** (sirve para priorizar, no para decidir solo).
- No sustituye debida diligencia (liquidez, distribución, lockups, reputación).

---

## 🔐 Seguridad & gobernanza
- Guardar **tokens de Telegram** fuera de repos públicos (variables de entorno).
- Acceso a la máquina/servidor con **usuarios limitados**.
- Definir **responsable** de operación (quién reacciona a las alertas).
- Políticas de **borrado** o rotación de logs si aplica.
