# bots/gate/gate_bot.py
"""
GateBot — Bot de swing trading amb 5 portes seqüencials

Segueix el patró BaseBot + ObservationSchema + from_config().
S'integra al DemoRunner com qualsevol altre bot.

Timeframes: 4H (decisions) + 1D (context P1/P2)
  - on_observation() cridat cada 4H (timeframe primari)
  - P1 i P2 s'evaluen 1 cop/dia (quan tanca nova candle diària)
  - P3, P4, P5 s'evaluen cada 4H

V1 long-only:
  - STRONG_BULL / WEAK_BULL → entrades llargues
  - RANGING → entrades amb R:R més alt
  - BEAR / UNCERTAIN → cap nova entrada; gestió de posicions obertes

Persistència de posicions: gate_positions (TimescaleDB)
Near-miss logging: gate_near_misses (TimescaleDB)
"""
from __future__ import annotations

import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action
from core.db.demo_repository import DemoRepository

from bots.gate.gates.p1_regime import P1Regime, P1Result
from bots.gate.gates.p2_health import P2Health
from bots.gate.gates.p3_structure import P3Structure
from bots.gate.gates.p4_momentum import P4Momentum
from bots.gate.gates.p5_risk import P5Risk
from bots.gate.near_miss_logger import NearMissLogger, GateSnapshot

logger = logging.getLogger(__name__)


class GateBot(BaseBot):
    """
    Bot principal del Gate System.
    Orquestra les 5 portes seqüencials.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)

        # ── Inicialitzar les 5 portes ─────────────────────────────────────
        self._p1 = P1Regime(config)
        self._p2 = P2Health(config)
        self._p3 = P3Structure(config)
        self._p4 = P4Momentum(config)
        self._p5 = P5Risk(config)

        # ── Features per timeframe (del YAML) ────────────────────────────
        self._features_4h: list[str] = config.get("features_4h", [])
        self._features_1d: list[str] = config.get("features_1d", [])
        self._lookback: int          = int(config.get("lookback", 300))

        # ── Estat intern ─────────────────────────────────────────────────
        self._open_positions: list[dict] = []   # posicions obertes (en memòria + BD)
        self._p1_result: P1Result | None = None
        self._p2_multiplier: float       = 1.0
        self._last_daily_ts: datetime | None = None  # última candle diària evaluada

        # ── PnL setmanal per P5 drawdown check ───────────────────────────
        self._week_start_value: float = 10_000.0
        self._week_start_ts: datetime | None = None

        # ── DB i logger ──────────────────────────────────────────────────
        self._repo = DemoRepository()
        self._near_miss_logger = NearMissLogger(self._repo)

    # ──────────────────────────────────────────────────────────────────────
    # BaseBot interface
    # ──────────────────────────────────────────────────────────────────────

    def observation_schema(self) -> ObservationSchema:
        """
        Declara timeframes=[4h,1d] i la unió deduplificada de features.
        L'ObservationBuilder carregarà ambdós DataFrames independentment.
        """
        all_features = list(dict.fromkeys(self._features_4h + self._features_1d))
        return ObservationSchema(
            features   = all_features,
            timeframes = ["4h", "1d"],
            lookback   = self._lookback,
            extras     = {"external": self.config.get("external", {})},
        )

    def on_start(self) -> None:
        """
        Cridat una vegada en iniciar. Carrega models i restaura posicions.
        """
        self._p1.load_models()

        # Restaurar posicions obertes des de la BD
        saved_positions = self._repo.get_open_gate_positions(self.bot_id)
        self._open_positions = saved_positions
        if saved_positions:
            logger.info(
                f"[{self.bot_id}] {len(saved_positions)} posicions obertes restaurades."
            )

    def on_observation(self, observation: dict) -> Signal:
        """
        Cridat cada 4H (timeframe primari). Executa el pipeline de les 5 portes.

        observation['4h']['features']:  DataFrame 4H (lookback files)
        observation['4h']['current_price']: preu actual
        observation['1d']['features']:  DataFrame 1D (lookback files)
        observation['portfolio']:       {'USDT': ..., 'BTC': ...}
        """
        now       = datetime.now(timezone.utc)
        portfolio = observation.get("portfolio", {})
        usdt_bal  = float(portfolio.get("USDT", 0.0))
        btc_bal   = float(portfolio.get("BTC", 0.0))

        obs_4h = observation["4h"]
        obs_1d = observation["1d"]
        df_4h  = obs_4h["features"]
        df_1d  = obs_1d["features"]
        current_price = float(obs_4h["current_price"])
        ts_4h         = obs_4h["timestamp"]
        ts_1d         = obs_1d["timestamp"]

        # ── ATR 14 del 4H (necessari per P3 i P5) ────────────────────────
        atr_14 = float(df_4h["atr_14"].iloc[-1]) if "atr_14" in df_4h.columns else 1.0
        atr_pct = self._p5.atr_percentile(df_4h)

        # ── PnL setmanal (per P5 drawdown check) ─────────────────────────
        portfolio_value = usdt_bal + btc_bal * current_price
        if self._week_start_ts is None or (now - self._week_start_ts).days >= 7:
            self._week_start_value = portfolio_value
            self._week_start_ts    = now
        weekly_pnl_pct = (portfolio_value - self._week_start_value) / self._week_start_value

        # ── P1 + P2: re-evaluar 1 cop/dia ────────────────────────────────
        if self._is_new_daily_candle(ts_1d):
            self._last_daily_ts = ts_1d
            self._p1_result     = self._p1.evaluate(df_1d)
            self._p2_multiplier = self._p2.evaluate(df_1d, self._p1_result.regime) \
                if self._p1_result.is_open else 0.0
            logger.info(
                f"[{self.bot_id}] Daily eval: P1={self._p1_result.regime} "
                f"conf={self._p1_result.confidence:.2f}  P2={self._p2_multiplier:.2f}"
            )

        # Si P1 no carregat encara, retornar hold
        if self._p1_result is None:
            return self._hold("P1 model not loaded yet")

        # ── Gestionar posicions obertes (cada 4H) ─────────────────────────
        exit_signal = self._manage_open_positions(
            df_4h=df_4h,
            df_1d=df_1d,
            current_price=current_price,
            atr_14=atr_14,
            atr_pct=atr_pct,
            regime=self._p1_result.regime,
            p2_multiplier=self._p2_multiplier,
            now=now,
        )
        if exit_signal is not None:
            return exit_signal

        # ── P1: la porta permet entrades? ────────────────────────────────
        if not self._p1_result.allows_long():
            return self._hold(f"P1 closed: {self._p1_result.regime}")

        # ── P2: multiplier > 0? ───────────────────────────────────────────
        if self._p2_multiplier <= 0.0:
            return self._hold("P2 multiplier=0 (emergency)")

        # ── P3: hi ha un nivell accionable? ──────────────────────────────
        p3_result = self._p3.evaluate(
            df_4h=df_4h,
            df_1d=df_1d,
            current_price=current_price,
            atr_14=atr_14,
            regime=self._p1_result.regime,
        )

        if not p3_result.has_actionable_level:
            return self._hold("P3 closed: no actionable level")

        # ── Inicialitzar snapshot per near-miss logger ────────────────────
        snapshot = GateSnapshot(
            bot_id        = self.bot_id,
            timestamp     = now,
            p1_regime     = self._p1_result.regime,
            p1_confidence = self._p1_result.confidence,
            p2_multiplier = self._p2_multiplier,
            p3_level_type = p3_result.best_level.level_type if p3_result.best_level else None,
            p3_level_strength = p3_result.best_level.strength if p3_result.best_level else None,
            p3_risk_reward    = p3_result.risk_reward,
            p3_volume_ratio   = p3_result.volume_ratio,
        )

        # ── P4: el momentum confirma l'entrada? ───────────────────────────
        p4_result = self._p4.evaluate(df_4h, self._p1_result.regime)

        snapshot.p4_d1_ok    = p4_result.signals_detail.get("d1_accel", False)
        snapshot.p4_d2_ok    = p4_result.signals_detail.get("d1_reversal", False)
        snapshot.p4_rsi_ok   = p4_result.signals_detail.get("rsi2_oversold", False)
        snapshot.p4_macd_ok  = p4_result.signals_detail.get("macd_cross", False)
        snapshot.p4_score    = p4_result.confidence
        snapshot.p4_triggered = p4_result.triggered

        if not p4_result.triggered:
            self._near_miss_logger.log(snapshot)
            return self._hold(
                f"P4 closed: {p4_result.signals_active}/{p4_result.signals_total} signals"
            )

        # ── P5: podem entrar? Amb quina mida? ────────────────────────────
        entry_price  = current_price
        stop_price   = p3_result.stop_level  or (entry_price * 0.97)
        target_price = p3_result.target_level or (entry_price * 1.03)

        p5_entry = self._p5.evaluate_entry(
            usdt_balance     = usdt_bal,
            entry_price      = entry_price,
            stop_price       = stop_price,
            target_price     = target_price,
            regime           = self._p1_result.regime,
            p2_multiplier    = self._p2_multiplier,
            p4_confidence    = p4_result.confidence,
            n_open_positions = len(self._open_positions),
            weekly_pnl_pct   = weekly_pnl_pct,
        )

        snapshot.p5_veto_reason  = p5_entry.veto_reason
        snapshot.p5_position_size = p5_entry.size_fraction

        if p5_entry.vetoed:
            self._near_miss_logger.log(snapshot)
            return self._hold(f"P5 veto: {p5_entry.veto_reason}")

        # ── EXECUTAR TRADE ────────────────────────────────────────────────
        snapshot.executed = True
        self._near_miss_logger.log(snapshot)

        # Crear posició i persistir a BD
        position = {
            "opened_at":    now,
            "entry_price":  entry_price,
            "stop_level":   stop_price,
            "target_level": target_price,
            "highest_price": entry_price,
            "size_usdt":    p5_entry.position_usdt,
            "regime":       self._p1_result.regime,
            "decel_counter": 0,
        }
        pos_id = self._repo.save_gate_position(self.bot_id, position)
        position["id"] = pos_id
        self._open_positions.append(position)

        logger.info(
            f"[{self.bot_id}] BUY @ {entry_price:.2f} | "
            f"stop={stop_price:.2f} target={target_price:.2f} "
            f"rr={p3_result.risk_reward:.2f} | "
            f"size={p5_entry.size_fraction:.3f} | "
            f"regime={self._p1_result.regime} P4={p4_result.signals_active}/4"
        )

        return Signal(
            bot_id     = self.bot_id,
            timestamp  = now,
            action     = Action.BUY,
            size       = p5_entry.size_fraction,
            confidence = p4_result.confidence,
            reason     = (
                f"gate_buy | regime={self._p1_result.regime} | "
                f"p4={p4_result.signals_active}/4 | rr={p3_result.risk_reward:.2f} | "
                f"p5_size={p5_entry.size_fraction:.3f}"
            ),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Gestió de posicions obertes
    # ──────────────────────────────────────────────────────────────────────

    def _manage_open_positions(
        self,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        current_price: float,
        atr_14: float,
        atr_pct: float,
        regime: str,
        p2_multiplier: float,
        now: datetime,
    ) -> Signal | None:
        """
        Avalua cada posició oberta. Si cal sortir, retorna el Signal de SELL.
        Actualitza el trailing stop si el preu ha avançat.
        Retorna None si cap posició requereix sortida imminent.
        """
        if not self._open_positions:
            return None

        # d2 actual per desacceleració
        d2_current = 0.0
        prev_close = None
        if "close" in df_4h.columns and len(df_4h) >= 5:
            smoothed = df_4h["close"].ewm(span=3, adjust=False).mean()
            d2_current = float(smoothed.pct_change().diff().iloc[-1])
            prev_close = float(df_4h["close"].iloc[-2])

        # P1 compatible amb posicions llargues
        p1_compatible = not (self._p1_result and
                             self._p1_result.invalidates_long_position())

        # P3 oberta (per estancament): re-avaluar si l'estructura de suport segueix vàlida
        p3_open = False
        if self._open_positions:
            p3_check = self._p3.evaluate(
                df_4h=df_4h,
                df_1d=df_1d,
                current_price=current_price,
                atr_14=atr_14,
                regime=regime,
            )
            p3_open = p3_check.has_actionable_level

        for pos in self._open_positions[:]:  # còpia per poder modificar
            result = self._p5.evaluate_position(
                position       = pos,
                current_price  = current_price,
                atr_14         = atr_14,
                atr_percentile = atr_pct,
                d2_current     = d2_current,
                regime         = regime,
                p1_compatible  = p1_compatible,
                p2_multiplier  = p2_multiplier,
                p3_open        = p3_open,
                prev_close     = prev_close,
            )

            # Actualitzar stop i counter
            updates: dict = {}
            if result.new_stop is not None:
                pos["stop_level"] = result.new_stop
                pos["highest_price"] = max(pos["highest_price"], current_price)
                updates["stop_level"]    = result.new_stop
                updates["highest_price"] = pos["highest_price"]

            # Actualitzar decel_counter
            if d2_current < 0:
                pos["decel_counter"] = pos.get("decel_counter", 0) + 1
            else:
                pos["decel_counter"] = 0
            updates["decel_counter"] = pos["decel_counter"]

            if updates and pos.get("id"):
                self._repo.update_gate_position(pos["id"], updates)

            # Sortida completa
            if result.should_exit:
                self._open_positions.remove(pos)
                if pos.get("id"):
                    self._repo.delete_gate_position(pos["id"])
                logger.info(
                    f"[{self.bot_id}] SELL @ {current_price:.2f} | "
                    f"reason={result.exit_reason} | "
                    f"entry={pos['entry_price']:.2f}"
                )
                return Signal(
                    bot_id     = self.bot_id,
                    timestamp  = now,
                    action     = Action.SELL,
                    size       = 1.0,   # vendre tot el BTC
                    confidence = 1.0,
                    reason     = f"gate_exit | {result.exit_reason}",
                )

            # Reducció al 50% per estancament
            if result.reduce_half:
                logger.info(
                    f"[{self.bot_id}] SELL 50% @ {current_price:.2f} | "
                    f"stagnation | entry={pos['entry_price']:.2f}"
                )
                return Signal(
                    bot_id     = self.bot_id,
                    timestamp  = now,
                    action     = Action.SELL,
                    size       = 0.5,
                    confidence = 0.5,
                    reason     = "gate_reduce_half | stagnation",
                )

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _is_new_daily_candle(self, ts_1d) -> bool:
        """Retorna True si és una nova candle diària (diferent de l'última evaluada)."""
        if self._last_daily_ts is None:
            return True
        if hasattr(ts_1d, 'date') and hasattr(self._last_daily_ts, 'date'):
            return ts_1d.date() != self._last_daily_ts.date()
        return ts_1d != self._last_daily_ts

    def _hold(self, reason: str) -> Signal:
        return Signal(
            bot_id    = self.bot_id,
            timestamp = datetime.now(timezone.utc),
            action    = Action.HOLD,
            size      = 0.0,
            confidence= 0.0,
            reason    = reason,
        )
