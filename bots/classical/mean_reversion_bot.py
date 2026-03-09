# bots/classical/mean_reversion_bot.py
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


class MeanReversionBot(BaseBot):
    """
    Mean Reversion Bot: Z-score del preu + RSI extrems + filtre de volum.

    Estat de l'art per BTC/USDT (1h):
    ─────────────────────────────────
    El BTC mostra reversió a la mitjana en mercats laterals i després de
    sobreextensions. Els millors resultats s'obtenen combinant:
      1. Z-score del preu vs. mitjana mòbil (sobreextensió estadística)
      2. RSI extrem com a confirmació (sobrevenuda genuïna vs. trending)
      3. Filtre de volum per evitar "catching a falling knife" en panic sells

    Lògica de senyals:
    ──────────────────
    BUY:  Z-score < -zscore_entry  AND  RSI < rsi_oversold  AND  volum normal
    SELL: Z-score > zscore_exit    OR   RSI > rsi_overbought  (si en posició)

    El Z-score mesura quantes desviacions estàndard s'aparta el preu actual
    de la seva mitjana mòbil. Un Z < -1.8 indica sobreextensió estadística
    que tendeix a revertir, especialment en períodes de baixa volatilitat.

    Paràmetres recomanats per BTC (1h, validats en literatura):
    ────────────────────────────────────────────────────────────
    - zscore_window: 20-50 candles (30 per defecte — equilibri reactiu/estable)
    - zscore_entry: 1.5-2.5σ (1.8 per defecte — senyal clar sense soroll)
    - rsi_oversold: 25-35 (30 per defecte — sobrevenuda real)
    - volume_filter_multiplier: 2.0-3.0x (2.5 per defecte — evita panics)
    """

    def __init__(self, config_path: str = "config/models/mean_reversion.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)
        self._in_position = False

    def observation_schema(self) -> ObservationSchema:
        return ObservationSchema(
            features=["close", "high", "low", "volume", "rsi_14"],
            timeframes=[self.config["timeframe"]],
            lookback=self.config["lookback"],
        )

    def on_observation(self, observation: dict) -> Signal:
        timeframe = self.config["timeframe"]
        features: pd.DataFrame = observation[timeframe]["features"].copy()

        window        = self.config["zscore_window"]
        z_entry       = self.config["zscore_entry"]
        z_exit        = self.config["zscore_exit"]
        rsi_oversold  = self.config["rsi_oversold"]
        rsi_overbought = self.config["rsi_overbought"]
        vol_mult      = self.config["volume_filter_multiplier"]
        trade_size    = self.config["trade_size"]

        close   = features["close"]
        volume  = features["volume"]
        rsi     = features["rsi_14"].iloc[-1]

        # ── Z-score del preu vs. mitjana mòbil ──────────────────────────────
        rolling_mean = close.rolling(window=window).mean()
        rolling_std  = close.rolling(window=window).std()
        zscore       = (close - rolling_mean) / rolling_std.replace(0, np.nan)
        zscore       = zscore.fillna(0)
        current_z    = zscore.iloc[-1]

        # ── Filtre de volum ──────────────────────────────────────────────────
        # No compra si el volum és molt alt respecte la mitjana:
        # volum alt en caiguda = panic sell → riscos d'una caiguda continuada
        avg_vol     = volume.rolling(window=window).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        normal_vol  = (current_vol < vol_mult * avg_vol) if avg_vol > 0 else True

        current_price = close.iloc[-1]

        # ── Senyal BUY ───────────────────────────────────────────────────────
        if (
            not self._in_position
            and current_z < -z_entry
            and rsi < rsi_oversold
            and normal_vol
        ):
            self._in_position = True
            # Confidence: com més extrema la sobreextensió, més confiança
            confidence = min(1.0, abs(current_z) / (z_entry * 2))
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.BUY,
                size=trade_size,
                confidence=confidence,
                reason=(
                    f"MeanReversion BUY: Z={current_z:.2f} < -{z_entry}, "
                    f"RSI={rsi:.1f} < {rsi_oversold}, "
                    f"Vol={current_vol/avg_vol:.1f}x avg, Price={current_price:.0f}"
                ),
            )

        # ── Senyal SELL ──────────────────────────────────────────────────────
        # Surt quan el preu ha revertit cap a la mitjana (Z prop de 0)
        # o quan el RSI indica sobrecompra
        z_reverted  = current_z > z_exit
        rsi_hot     = rsi > rsi_overbought

        if self._in_position and (z_reverted or rsi_hot):
            self._in_position = False

            if z_reverted:
                confidence = min(1.0, current_z / (z_exit + 1.0))
                reason = (
                    f"MeanReversion SELL (reversion): Z={current_z:.2f} > {z_exit}, "
                    f"RSI={rsi:.1f}, Price={current_price:.0f}"
                )
            else:
                confidence = min(1.0, rsi / 100)
                reason = (
                    f"MeanReversion SELL (RSI hot): Z={current_z:.2f}, "
                    f"RSI={rsi:.1f} > {rsi_overbought}, Price={current_price:.0f}"
                )

            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.SELL,
                size=1.0,
                confidence=confidence,
                reason=reason,
            )

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=Action.HOLD,
            size=0.0,
            confidence=1.0,
            reason=f"No signal: Z={current_z:.2f}, RSI={rsi:.1f}",
        )

    def on_start(self) -> None:
        self._in_position = False
        logger.info(
            f"MeanReversionBot initialized | "
            f"window={self.config['zscore_window']}, "
            f"z_entry={self.config['zscore_entry']}, "
            f"rsi_oversold={self.config['rsi_oversold']}"
        )
