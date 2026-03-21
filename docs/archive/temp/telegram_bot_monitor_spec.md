## Especificació del Telegram Bot Monitor — CryptoBot

### Context general

És un bot de Telegram que monitora entre 2 i 10 bots de trading de BTC/USDT en mode demo. Cada bot té un portfolio independent de $10,000 inicials i opera de manera autònoma. Els bots no s'inicien el mateix dia: alguns porten setmanes funcionant i d'altres s'acaben d'afegir. El bot de Telegram ha de ser llegible, clar i còmode d'usar des del mòbil. La prioritat és la llegibilitat per davant de la completesa.

---

### Comanda `/status`

Respon amb un únic missatge de text formatat en monospace (entre triple backtick). Mostra una taula llista de tots els bots actius amb tres columnes: nom del bot, retorn percentual acumulat des de l'inici, i alpha vs Buy & Hold (diferència entre el retorn del bot i el retorn que hauria tingut comprant BTC en el mateix moment en què el bot es va activar i mantenint-lo fins ara). Cada fila porta un emoji de semàfor: 🟢 si l'alpha és positiu, 🔴 si és negatiu, 🟡 si és marginalment negatiu (per exemple, entre -1% i 0%). El missatge no inclou capital total ni millor/pitjor bot, simplement la llista.

Si hi ha alguna alerta activa (bot inactiu, dada sense actualitzar, error), el bot envia un segon missatge separat amb fons de warning (si la plataforma ho suporta) o simplement amb l'emoji ⚠️ seguit del text de l'alerta. Aquests missatges d'alerta els pot enviar el bot de manera proactiva sense que l'usuari hagi escrit cap comanda.

---

### Comanda `/portfolio`

Respon amb un missatge que conté botons inline de Telegram (InlineKeyboardMarkup). Hi ha un botó per cada bot actiu (per exemple: "RF_v3", "XGB_v2", "PPO", "SAC", "PatchTST"). Quan l'usuari prem un botó, el mateix missatge s'edita (via `editMessageText` + `editMessageReplyMarkup`) per mostrar el detall del bot seleccionat, mantenint els botons visibles a sota. Això simula el comportament de pestanyes sense crear missatges nous.

El detall de cada bot mostra, en format monospace: nom del bot i data d'inici, capital actual en dòlars, retorn percentual acumulat, alpha vs Buy & Hold, max drawdown, Sharpe ratio, win rate, i la posició actual (quantitat de BTC i percentatge, i quantitat de USDT i percentatge). Si el bot té alguna alerta activa (per exemple, inactivitat), s'afegeix una línia ⚠️ al final.

---

### Comanda `/compare`

Respon amb dos elements:

Primer, una imatge generada dinàmicament (via matplotlib o similar) que mostra un gràfic de línies amb l'evolució del retorn acumulat de cada bot des del seu dia 0. L'eix X és el temps (dates), l'eix Y és el percentatge de retorn. Cada bot té una línia de color diferent. Hi ha una línia adicional discontínua en gris que representa el Buy & Hold de BTC indexat des del dia d'inici de cada bot (nota: com que cada bot té una data d'inici diferent, la línia de B&H es calcula per a cada bot de manera independent, però al gràfic es representa una sola línia de referència que és la mitjana o la del bot més antic, o bé el gràfic es normalitza per mostrar tots des del dia 0 = 0%). Sota el gràfic hi ha una llegenda horitzontal compacta amb el color i el nom de cada bot.

Segon, just a sota de la imatge, un missatge de text en monospace amb una llista ordenada de major a menor retorn, mostrant per cada bot: nom, retorn actual, i retorn del B&H corresponent.

---

### Comanda `/trades`

Respon amb un missatge de text en monospace. Mostra les últimes N operacions de tots els bots combinades, ordenades per hora descendent (la més recent primer). Per cada operació: hora, nom del bot en negreta, tipus (🟢 BUY o 🔴 SELL), quantitat de BTC, preu d'execució en dòlars, P&L de l'operació en dòlars, i confiança del senyal en percentatge. Al final del missatge, un resum de la jornada: nombre total d'operacions i P&L net del dia en dòlars.

---

### Comanda `/health`

Respon amb un missatge de text en monospace. Llista totes les fonts de dades i components del sistema amb el seu estat. Cada línia porta ✅ si tot va bé o ⚠️ si hi ha algun problema. Les línies que cal verificar: OHLCV (temps des de l'última actualització), Fear & Greed Index (temps + valor actual), connexió a TimescaleDB, MLflow tracking, latència de l'API de Binance, i un estat per cada bot (🟢 nominal o ⚠️ amb descripció del problema). Al final: resum comptant quants bots estan OK i quants avisos hi ha actius, i el temps fins al proper check automàtic.

---

### Alertes proactives (push, sense comanda)

El bot envia missatges automàtics en els casos següents, sense que l'usuari hagi demanat res:

- Un bot porta més de X hores sense executar cap operació (llindar configurable, per exemple 4h durant horari actiu).
- Una font de dades no s'ha actualitzat en el temps esperat (per exemple, OHLCV amb més de 5 minuts de retard).
- El max drawdown d'un bot supera un llindar configurable (per exemple, -10%).
- Hi ha un error no controlat en qualsevol component del sistema.

Cada alerta és un missatge curt, directe, i amb context suficient per saber exactament què ha passat i quin bot o component és el responsable.

---

### Notes tècniques

Les respostes de text usen format monospace (triple backtick o parse_mode HTML amb `<pre>`) per mantenir l'alineació de les taules. Els botons inline de `/portfolio` usen `callback_data` per identificar quin bot s'ha seleccionat i el handler edita el missatge existent en lloc de crear-ne un de nou. El gràfic de `/compare` es genera com a imatge PNG i s'envia com a `sendPhoto`. Tots els valors numèrics es formategen amb precisió consistent: retorns amb un decimal i símbol `%`, dòlars amb separador de milers i dos decimals. Les dates d'inici dels bots es mostren en format `DD/MM`.
