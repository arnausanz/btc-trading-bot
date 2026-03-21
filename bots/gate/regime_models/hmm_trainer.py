# bots/gate/regime_models/hmm_trainer.py
"""
HMM Trainer — Descoberta de règims de mercat (Porta 1, Fase 1)

Entrena un GaussianHMM (hmmlearn) sobre 3 observacions diàries:
  - daily_return:       retorn diari del preu
  - normalized_atr14:  ATR14 / close (volatilitat relativa)
  - volume_sma_ratio:  volume / SMA(20) del volume (activitat relativa)

Selecciona K (nombre d'estats) de 2 a 6 per BIC mínim.
Per cada K: 10 inicialitzacions aleatòries → màxima log-verosimilitud per estabilitat.

Decodificació Viterbi → seqüència d'estats [0, 2, 1, 0, 3, ...]

Mapeig automàtic HMM state → RegimeState:
  Ordena estats per mean_return → STRONG_BULL (max return) ... STRONG_BEAR (min return)
  Refinament per ADX i volatilitat per distingir WEAK de STRONG.

Guarda: {model, state_mapping, k, bic_scores} a models/gate_hmm.pkl
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Màpings possibles de K → etiquetes de règim (per index d'ordre de retorn)
# STRONG_BULL = idx amb major mean_return, STRONG_BEAR = menor
_REGIME_NAMES = ["STRONG_BEAR", "WEAK_BEAR", "RANGING", "WEAK_BULL", "STRONG_BULL"]


class HMMTrainResult(NamedTuple):
    model: GaussianHMM
    scaler: StandardScaler
    state_mapping: dict[int, str]    # {hmm_state_id: regime_name}
    k: int
    bic_scores: dict[int, float]
    labels: np.ndarray               # Viterbi sequence per tot el dataset


class HMMTrainer:
    """Entrena i selecciona el millor GaussianHMM per a descoberta de règims."""

    def __init__(
        self,
        k_range: tuple[int, int] = (2, 6),
        n_init: int = 10,
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.k_min, self.k_max = k_range
        self.n_init       = n_init
        self.n_iter       = n_iter
        self.random_state = random_state

    def fit(self, df_daily: pd.DataFrame) -> HMMTrainResult:
        """
        Entrena HMMs per K = k_min..k_max, selecciona per BIC.

        Args:
            df_daily: DataFrame diari amb columnes ['close', 'atr_14', 'volume'].
                     Mínim 500 files (~2 anys).
        """
        # ── Preparar observacions ─────────────────────────────────────────
        obs = self._prepare_observations(df_daily)
        logger.info(f"HMM training: {len(obs)} observations, K={self.k_min}..{self.k_max}")

        # ── Escalar observacions ──────────────────────────────────────────
        scaler = StandardScaler()
        obs_scaled = scaler.fit_transform(obs)

        # ── Cercar millor K per BIC ───────────────────────────────────────
        bic_scores: dict[int, float] = {}
        best_k      = self.k_min
        best_bic    = np.inf
        best_model: GaussianHMM | None = None

        for k in range(self.k_min, self.k_max + 1):
            model, loglik = self._train_best_init(obs_scaled, k)
            if model is None:
                logger.warning(f"  K={k}: tots els inits han fallat, skipping")
                continue

            bic = self._bic(k, n_obs=obs_scaled.shape[1], loglik=loglik, T=len(obs_scaled))
            bic_scores[k] = bic
            logger.info(f"  K={k}: loglik={loglik:.1f}  BIC={bic:.1f}")

            if bic < best_bic:
                best_bic   = bic
                best_k     = k
                best_model = model

        if best_model is None:
            raise RuntimeError("HMM: cap model ha convergit. Comprova les dades.")

        logger.info(f"Best K={best_k} (BIC={best_bic:.1f})")

        # ── Decodificació Viterbi → etiquetes ─────────────────────────────
        _, raw_labels = best_model.decode(obs_scaled, algorithm="viterbi")

        # ── Mapeig HMM states → RegimeState ──────────────────────────────
        state_mapping = self._map_states(raw_labels, obs, best_k)

        return HMMTrainResult(
            model=best_model,
            scaler=scaler,
            state_mapping=state_mapping,
            k=best_k,
            bic_scores=bic_scores,
            labels=raw_labels,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_observations(df: pd.DataFrame) -> np.ndarray:
        """
        Construeix la matriu d'observacions [T×3] per a l'HMM:
          col 0: daily_return       (pct_change del close)
          col 1: normalized_atr14   (atr_14 / close)
          col 2: volume_sma_ratio   (volume / SMA_20 del volume)
        """
        ret      = df["close"].pct_change().fillna(0.0)
        norm_atr = (df["atr_14"] / df["close"]).fillna(0.0)
        vol_ma   = df["volume"].rolling(20, min_periods=1).mean()
        vol_ratio = (df["volume"] / vol_ma).fillna(1.0)
        obs = np.column_stack([ret.values, norm_atr.values, vol_ratio.values])
        return obs.astype(float)

    def _train_best_init(
        self, obs_scaled: np.ndarray, k: int
    ) -> tuple[GaussianHMM | None, float]:
        """
        Entrena n_init models independents i retorna el de màxima log-verosimilitud.
        """
        best_model: GaussianHMM | None = None
        best_loglik = -np.inf

        for seed in range(self.n_init):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="full",
                    n_iter=self.n_iter,
                    random_state=self.random_state + seed,
                    verbose=False,
                )
                model.fit(obs_scaled)
                loglik = model.score(obs_scaled)
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_model  = model
            except Exception as e:
                logger.debug(f"  K={k} seed={seed}: {e}")
                continue

        return best_model, best_loglik

    @staticmethod
    def _bic(k: int, n_obs: int, loglik: float, T: int) -> float:
        """
        BIC = -2×loglik + nparams×log(T)
        nparams: transicions + means + covariances (full)
        """
        n_params = (
            k * (k - 1)                          # transicions
            + k * n_obs                          # means
            + k * n_obs * (n_obs + 1) // 2       # covariances (full)
        )
        return -2.0 * loglik + n_params * np.log(T)

    @staticmethod
    def _map_states(labels: np.ndarray, obs: np.ndarray, k: int) -> dict[int, str]:
        """
        Mapeja els estat HMM → RegimeState per mean_return (obs col 0).

        Ordena per retorn mitjà:
          - Estat amb major retorn → STRONG_BULL (o BULL si K<4)
          - Estat amb menor retorn → STRONG_BEAR (o BEAR)
          - Intermitjos → RANGING / WEAK variants

        El mapeig és semàntic: noms del _REGIME_NAMES llista alineats amb l'ordre
        dels estats per retorn, agafant els K centrals.
        """
        # Retorn mitjà per estat
        mean_returns: dict[int, float] = {}
        for state in range(k):
            mask = labels == state
            if mask.sum() > 0:
                mean_returns[state] = float(obs[mask, 0].mean())
            else:
                mean_returns[state] = 0.0

        # Ordenar estats de menor a major retorn
        sorted_states = sorted(mean_returns.keys(), key=lambda s: mean_returns[s])

        # Distribuir noms: agafar els K últims de _REGIME_NAMES (de BEAR a BULL)
        # Per K=2: [STRONG_BEAR, STRONG_BULL]
        # Per K=3: [STRONG_BEAR, RANGING, STRONG_BULL]
        # Per K=4: [STRONG_BEAR, WEAK_BEAR, WEAK_BULL, STRONG_BULL]
        # Per K=5: tot el _REGIME_NAMES (RANGING al centre)
        # Per K=6: UNCERTAIN per als duplicats del centre
        regime_labels_available = _REGIME_NAMES.copy()

        # Per K < 5, triem els noms dels extrems + el/s central/s
        if k == 2:
            names_to_use = ["STRONG_BEAR", "STRONG_BULL"]
        elif k == 3:
            names_to_use = ["STRONG_BEAR", "RANGING", "STRONG_BULL"]
        elif k == 4:
            names_to_use = ["STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL"]
        elif k == 5:
            names_to_use = _REGIME_NAMES
        else:  # k >= 6: UNCERTAIN per als estats extres del centre
            names_to_use = _REGIME_NAMES + ["UNCERTAIN"] * (k - 5)

        state_mapping = {}
        for idx, state in enumerate(sorted_states):
            if idx < len(names_to_use):
                state_mapping[state] = names_to_use[idx]
            else:
                state_mapping[state] = "UNCERTAIN"

        logger.info(f"State mapping: {state_mapping}")
        logger.info(f"Mean returns:  {mean_returns}")
        return state_mapping

    # ──────────────────────────────────────────────────────────────────────
    # Persistència
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def save(result: HMMTrainResult, path: str | Path) -> None:
        """Guarda el resultat complet de l'entrenament HMM."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":         result.model,
                "scaler":        result.scaler,
                "state_mapping": result.state_mapping,
                "k":             result.k,
                "bic_scores":    result.bic_scores,
            }, f)
        logger.info(f"HMM saved → {path}")

    @staticmethod
    def load(path: str | Path) -> dict:
        """Carrega un model HMM guardat prèviament."""
        with open(path, "rb") as f:
            return pickle.load(f)
