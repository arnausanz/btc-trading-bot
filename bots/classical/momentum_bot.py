# bots/classical/momentum_bot.py
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class MomentumBot(BaseBot):
    """
    Momentum Bot: Rate of Change (ROC) + confirmació de volum + MACD.

    Estat de l'art per BTC/USDT (1h):
    ─────────────────────────────────
    El BTC és un dels actius amb major persistència de momentum documentada.
    Les estratègies de momentum funcionen especialment bé en bull markets i
    en breakouts. Els millors resultats s'obtenen combinant:
      1. Rate of Change (ROC) — mesura la velocitat del canvi de preu
      2. Confirmació de volum — el volum ha de confirmar el moviment
      3. MACD histogram — detecta acceleració de momentum
      4. Filtre RSI — evita entrades en zones de sobrecompra extrema

    Lògica de senyals:
    ──────────────────
    BUY:  ROC > roc_threshold  AND  volum > avg*vol_mult
          AND  MACD histogram > 0  AND  RSI < rsi_max_entry

    SELL: MACD histogram creua a negatiu  OR  ROC < -roc_threshold*0.5
          OR  RSI > 80  (si en posició)

    Per al BTC (1h), paràmetres validats en literatura recent:
    ──────────────────────────────────────────────────────────
    - roc_period: 12-24h (14 per defecte — captura momentum intraday-swing)
    - roc_threshold: 1.0-3.0% (1.5% per defecte — moviment significatiu)
    - volume_multiplier: 1.2-2.0x (1.3 per defecte — confirma sense ser massa restrictiu)
    - MACD(12, 26, 9) — estàndard; histograma positiu confirma acceleració
    - rsi_max_entry: 70-75 (72 per defecte — no perseguir sobrecompra)
    """

    def __init__(self, config_path: str = "config/models/momentum.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._in_position = False

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "volume", "rsi_14"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()

        roc_period      = self.config["roc_period"]
        roc_threshold   = self.config["roc_threshold"]
        vol_window      = self.config["volume_window"]
        vol_mult        = self.config["volume_multiplier"]
        macd_fast       = self.config["macd_fast"]
        macd_slow       = self.config["macd_slow"]
        macd_sig        = self.config["macd_signal"]
        rsi_max_entry   = self.config["rsi_max_entry"]
        trade_size      = self.config["trade_size"]

        close  = features["close"]
        volume = features["volume"]
        rsi    = features["rsi_14"].iloc[-1]

        # ── Rate of Change (ROC) ─────────────────────────────────────────────
        # ROC = (preu_actual - preu_N_periodes_enrere) / preu_N_periodes_enrere × 100
        shifted       = close.shift(roc_period)
        roc           = (close - shifted) / shifted.replace(0, shifted.mean()) * 100
        current_roc   = roc.iloc[-1]

        # ── Confirmació de volum ─────────────────────────────────────────────
        # El volum ha de superar la mitjana per confirmar que el moviment
        # no és un fals breakout de baix volum
        avg_vol        = volume.rolling(window=vol_window).mean().iloc[-1]
        current_vol    = volume.iloc[-1]
        vol_confirmed  = (current_vol > vol_mult * avg_vol) if avg_vol > 0 else False

        # ── MACD (calculat dinàmicament) ─────────────────────────────────────
        ema_fast    = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow    = close.ewm(span=macd_slow, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_sig, adjust=False).mean()
        macd_hist   = macd_line - signal_line

        current_hist = macd_hist.iloc[-1]
        prev_hist    = macd_hist.iloc[-2]

        current_price = close.iloc[-1]

        # ── Senyal BUY ───────────────────────────────────────────────────────
        # Tots els filtres han de coincidir: ROC fort + volum + MACD positiu + RSI ok
        if (
            not self._in_position
            and current_roc > roc_threshold
            and vol_confirmed
            and current_hist > 0
            and rsi < rsi_max_entry
        ):
            self._in_position = True
            # Confidence: com major el ROC i confirmació de volum, més confiança
            confidence = min(1.0, current_roc / (roc_threshold * 3))
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=trade_size,
                confidence=confidence,
                reason=(
                    f"Momentum BUY: ROC={current_roc:.2f}% > {roc_threshold}%, "
                    f"Vol={current_vol/avg_vol:.1f}x avg, "
                    f"MACD_hist={current_hist:.2f}, RSI={rsi:.1f}, "
                    f"Price={current_price:.0f}"
                ),
            )

        # ── Senyal SELL ──────────────────────────────────────────────────────
        # Sortida en 3 condicions (OR):
        # 1. MACD histogram creua a negatiu → momentum s'ha girat
        # 2. ROC reverteix significativament → tendència s'ha exhaurit
        # 3. RSI sobrecompra extrema → risc de pull-back sever
        macd_crossed_down = (prev_hist >= 0 and current_hist < 0)
        roc_reversed      = (current_roc < -roc_threshold * 0.5)
        rsi_extreme       = (rsi > 80)

        if self._in_position and (macd_crossed_down or roc_reversed or rsi_extreme):
            self._in_position = False

            reason_parts = []
            if macd_crossed_down:
                reason_parts.append(f"MACD crossed negative (hist={current_hist:.2f})")
            if roc_reversed:
                reason_parts.append(f"ROC reversed ({current_roc:.2f}%)")
            if rsi_extreme:
                reason_parts.append(f"RSI extreme ({rsi:.1f})")

            if macd_crossed_down:
                confidence = min(1.0, abs(current_hist) / 50)
            elif rsi_extreme:
                confidence = min(1.0, rsi / 100)
            else:
                confidence = 0.6

            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=confidence,
                reason=f"Momentum SELL: {', '.join(reason_parts)}",
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason=f"No signal: ROC={current_roc:.2f}%, RSI={rsi:.1f}",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info(
            f"MomentumBot initialized | "
            f"roc_period={self.config['roc_period']}, "
            f"roc_threshold={self.config['roc_threshold']}%, "
            f"vol_mult={self.config['volume_multiplier']}x"
        )
