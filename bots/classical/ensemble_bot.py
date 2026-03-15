# bots/classical/ensemble_bot.py
"""
EnsembleBot — meta-bot que combina senyals de múltiples sub-bots.

Polítiques disponibles
----------------------
majority_vote (v1, per a la demo)
    BUY  si més del 50 % dels sub-bots diuen BUY.
    SELL si més del 50 % dels sub-bots diuen SELL.
    HOLD en qualsevol altre cas.

weighted (futur)
    Cada sub-bot té un pes proporcional al seu Sharpe dels darrers N dies.

stacking (futur)
    Model ML de 2a capa entrenat sobre les prediccions dels sub-bots.

Afegir sub-bots mentre la demo corre
--------------------------------------
Edita ``config/models/ensemble.yaml`` (secció ``sub_bots``) i reinicia el
DemoRunner.  L'EnsembleBot no és en si un gestor de cartera — té el seu propi
PaperExchange independent.  Pots afegir sub-bots individuals a ``config/demo.yaml``
per executar-los en paral·lel amb l'ensemble sense interferències.
"""
import importlib
import logging
from datetime import datetime, timezone
from typing import Any

import yaml

from core.interfaces.base_bot import BaseBot, ObservationSchema
from core.models import Signal, Action

logger = logging.getLogger(__name__)


def _load_sub_bot(config_path: str) -> BaseBot:
    """
    Carrega un sub-bot des del seu config_path.

    Utilitza els camps ``module`` + ``class_name`` del YAML per a càrrega dinàmica
    (classical bots), o detecta el tipus per ``category`` per a ML/RL.

    Args:
        config_path: Ruta al YAML del sub-bot (ex: ``config/models/trend.yaml``).

    Returns:
        Instància inicialitzada del bot.

    Raises:
        ValueError: Si no es pot determinar el tipus de bot.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    category = cfg.get("category", "classic").upper()

    # ── ML supervisat ──────────────────────────────────────────────────────────
    if category == "ML":
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)

    # ── RL ────────────────────────────────────────────────────────────────────
    if category == "RL":
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)

    # ── Classical / Ensemble: càrrega genèrica via module + class_name ────────
    module_path = cfg.get("module")
    class_name  = cfg.get("class_name")
    if module_path and class_name:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(config_path=config_path)

    raise ValueError(
        f"[EnsembleBot] No es pot carregar sub-bot: {config_path}. "
        "El YAML ha de tenir 'module' + 'class_name' o 'category: ML/RL'."
    )


class EnsembleBot(BaseBot):
    """
    Meta-bot que agrega els senyals de múltiples sub-bots en una sola decisió.

    Config mínima (config/models/ensemble.yaml)::

        bot_id: ensemble_v1
        policy: majority_vote      # majority_vote | weighted (futur)
        trade_size: 0.5            # fracció del capital per ordre
        sub_bots:
          - config/models/trend.yaml
          - config/models/mean_reversion.yaml
          - ...

    Paràmetres
    ----------
    config_path : str
        Ruta al YAML de l'EnsembleBot.
    """

    def __init__(self, config_path: str = "config/models/ensemble.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        super().__init__(bot_id=config["bot_id"], config=config)

        self._policy: str     = config.get("policy", "majority_vote")
        self._trade_size: float = config.get("trade_size", 0.5)
        self._in_position: bool = False

        # Instantiate sub-bots
        self.sub_bots: list[BaseBot] = []
        sub_bot_paths: list[str] = config.get("sub_bots", [])
        if not sub_bot_paths:
            logger.warning("[EnsembleBot] No s'han definit sub_bots a la config!")
        for path in sub_bot_paths:
            try:
                bot = _load_sub_bot(path)
                self.sub_bots.append(bot)
                logger.info(f"[EnsembleBot] Sub-bot carregat: {bot.bot_id} ({path})")
            except Exception as exc:
                logger.error(f"[EnsembleBot] Error carregant sub-bot {path}: {exc}")

        if not self.sub_bots:
            raise RuntimeError(
                f"[EnsembleBot] Cap sub-bot carregat. Comprova 'sub_bots' al YAML."
            )

        logger.info(
            f"[EnsembleBot] Inicialment amb {len(self.sub_bots)} sub-bots | "
            f"política: {self._policy}"
        )

    # ── ObservationSchema ─────────────────────────────────────────────────────

    def observation_schema(self) -> ObservationSchema:
        """
        Retorna la unió dels schemas de tots els sub-bots.

        - features: unió de totes les features requerides
        - timeframes: unió de tots els timeframes
        - lookback: màxim dels lookbacks individuals
        - extras: merge dels dicts extras
        """
        all_features:    set[str]      = set()
        all_timeframes:  set[str]      = set()
        max_lookback:    int           = 0
        merged_extras:   dict[str, Any] = {}

        for bot in self.sub_bots:
            try:
                schema = bot.observation_schema()
                all_features.update(schema.features)
                all_timeframes.update(schema.timeframes)
                max_lookback = max(max_lookback, schema.lookback)
                merged_extras.update(schema.extras)
            except Exception as exc:
                logger.warning(
                    f"[EnsembleBot] No s'ha pogut obtenir schema de {bot.bot_id}: {exc}"
                )

        # Fallback si cap sub-bot ha retornat res
        if not all_features:
            all_features = {"close", "rsi_14"}
        if not all_timeframes:
            all_timeframes = {"1h"}
        if max_lookback == 0:
            max_lookback = 200

        return ObservationSchema(
            features=sorted(all_features),
            timeframes=sorted(all_timeframes),
            lookback=max_lookback,
            extras=merged_extras,
        )

    # ── Voting ────────────────────────────────────────────────────────────────

    def _majority_vote(self, signals: list[Signal]) -> tuple[Action, float, str]:
        """
        Aplica la política majority_vote sobre la llista de senyals.

        Returns:
            (action, confidence, reason)
        """
        n_buy  = sum(1 for s in signals if isinstance(s.action, Action) and s.action == Action.BUY)
        n_sell = sum(1 for s in signals if isinstance(s.action, Action) and s.action == Action.SELL)
        n_hold = len(signals) - n_buy - n_sell
        total  = len(signals)
        threshold = total / 2  # estricte: > 50%

        buy_sub  = [s.bot_id for s in signals if isinstance(s.action, Action) and s.action == Action.BUY]
        sell_sub = [s.bot_id for s in signals if isinstance(s.action, Action) and s.action == Action.SELL]

        if n_buy > threshold:
            return (
                Action.BUY,
                n_buy / total,
                f"Majority vote BUY {n_buy}/{total} | {', '.join(buy_sub)}",
            )
        if n_sell > threshold:
            return (
                Action.SELL,
                n_sell / total,
                f"Majority vote SELL {n_sell}/{total} | {', '.join(sell_sub)}",
            )
        return (
            Action.HOLD,
            max(n_hold, max(n_buy, n_sell)) / total,
            f"No majoria — BUY:{n_buy} SELL:{n_sell} HOLD:{n_hold}/{total}",
        )

    # ── on_observation ────────────────────────────────────────────────────────

    def on_observation(self, observation: dict) -> Signal:
        """
        Crida cada sub-bot amb la mateixa observació i aplica la política de vot.

        Nota: cada sub-bot manté el seu propi estat intern (``_in_position``, etc.)
        independentment de l'EnsembleBot.  L'EnsembleBot té el seu propi estat
        ``_in_position`` per controlar si ha comprat o no.
        """
        signals: list[Signal] = []
        for bot in self.sub_bots:
            try:
                sig = bot.on_observation(observation)
                signals.append(sig)
            except Exception as exc:
                logger.warning(
                    f"[EnsembleBot] Error en sub-bot {bot.bot_id}: {exc}",
                    exc_info=True,
                )

        if not signals:
            logger.error("[EnsembleBot] Cap sub-bot ha generat senyal. Retornant HOLD.")
            return Signal(
                bot_id=self.bot_id,
                timestamp=datetime.now(timezone.utc),
                action=Action.HOLD,
                size=0.0,
                confidence=0.0,
                reason="Error: cap sub-bot disponible",
            )

        if self._policy == "majority_vote":
            action, confidence, reason = self._majority_vote(signals)
        else:
            # Política no implementada → fallback majority_vote
            logger.warning(
                f"[EnsembleBot] Política '{self._policy}' no implementada. "
                "Usant majority_vote."
            )
            action, confidence, reason = self._majority_vote(signals)

        # Gestió d'estat: no comprar si ja estem en posició, no vendre si no ho estem
        if action == Action.BUY and self._in_position:
            action   = Action.HOLD
            reason   = f"[HOLD] Ja en posició | {reason}"
            confidence = 1.0
        elif action == Action.SELL and not self._in_position:
            action   = Action.HOLD
            reason   = f"[HOLD] Sense posició oberta | {reason}"
            confidence = 1.0
        else:
            if action == Action.BUY:
                self._in_position = True
            elif action == Action.SELL:
                self._in_position = False

        size = self._trade_size if action == Action.BUY else (1.0 if action == Action.SELL else 0.0)

        return Signal(
            bot_id=self.bot_id,
            timestamp=datetime.now(timezone.utc),
            action=action,
            size=size,
            confidence=confidence,
            reason=reason,
        )

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def on_start(self) -> None:
        self._in_position = False
        for bot in self.sub_bots:
            if hasattr(bot, "on_start"):
                bot.on_start()
        logger.info(
            f"[EnsembleBot] on_start — {len(self.sub_bots)} sub-bots actius | "
            f"política: {self._policy}"
        )

    def on_stop(self) -> None:
        for bot in self.sub_bots:
            if hasattr(bot, "on_stop"):
                bot.on_stop()
        logger.info("[EnsembleBot] on_stop")
