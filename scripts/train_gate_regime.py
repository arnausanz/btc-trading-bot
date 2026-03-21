#!/usr/bin/env python3
# scripts/train_gate_regime.py
"""
Pipeline complet d'entrenament del Gate System (Porta 1)

Passos:
  1. Carregar OHLCV diari + Fear & Greed + funding rate via FeatureBuilder
  2. Calcular les 14 features de P1
  3. Entrenar HMM K=2..6, seleccionar per BIC, decodificar Viterbi → labels
  4. Mapejar HMM states → RegimeState
  5. Walk-forward XGBoost (5 folds) + Optuna
  6. Validar (accuracy OOS > 55%, mlogloss std < 15%)
  7. Guardar models/gate_hmm.pkl + models/gate_xgb_regime.pkl

Ús:
  cd /path/to/btc-trading-bot
  python scripts/train_gate_regime.py
  python scripts/train_gate_regime.py --config config/models/gate.yaml
  python scripts/train_gate_regime.py --n-trials 100 --no-optuna
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Afegir el directori arrel al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processing.feature_builder import FeatureBuilder
from bots.gate.regime_models.hmm_trainer import HMMTrainer
from bots.gate.regime_models.xgb_classifier import XGBRegimeClassifier, P1_FEATURES
from bots.gate.gates.p1_regime import _linear_slope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_gate_regime")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Gate System P1 regime models")
    p.add_argument("--config",    default="config/models/gate.yaml", help="YAML config path")
    p.add_argument("--n-trials",  type=int, default=50, help="Optuna trials")
    p.add_argument("--no-optuna", action="store_true", help="Skip Optuna, use default params")
    p.add_argument("--symbol",    default="BTC/USDT")
    return p.parse_args()


def load_daily_data(symbol: str, config: dict) -> pd.DataFrame:
    """
    Carrega el DataFrame diari via FeatureBuilder.
    Inclou: OHLCV + Fear&Greed + funding_rate.
    """
    external = config.get("external", {})
    # Assegurem que les dades externes necessàries estan actives
    ext_cfg = {
        "fear_greed":   external.get("fear_greed", True),
        "funding_rate": external.get("funding_rate", True),
    }
    fb = FeatureBuilder(
        symbol=symbol,
        timeframe="1d",
        external=ext_cfg,
        select=None,
    )
    df = fb.build()
    logger.info(f"Daily data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_p1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa les 14 features de P1 sobre el DataFrame diari complet.
    Retorna un DataFrame alineat amb P1_FEATURES (mateixa longitud que df).
    """
    close  = df["close"]
    volume = df["volume"]

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ema_ratio_50_200
    ema_ratio = ema50 / ema200

    # ema_slope_50 i ema_slope_200 (rolling window)
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Pendent lineal en finestra mòbil, normalitzat per l'últim valor."""
        slopes = series.rolling(window).apply(
            lambda x: _linear_slope(x) / (x.iloc[-1] if x.iloc[-1] != 0 else 1.0),
            raw=False,
        )
        return slopes

    ema50_slope  = rolling_slope(ema50, 10)
    ema200_slope = rolling_slope(ema200, 20)

    # adx_14
    adx = df["adx_14"] if "adx_14" in df.columns else pd.Series(25.0, index=df.index)

    # atr_14_percentile: rolling percentile (252 dies)
    atr_series = df["atr_14"] if "atr_14" in df.columns else pd.Series(0.0, index=df.index)
    atr_pct = atr_series.rolling(252, min_periods=30).apply(
        lambda x: float((x[:-1] < x[-1]).mean() * 100.0), raw=True
    )

    # funding_rate
    if "funding_rate" in df.columns:
        fr       = df["funding_rate"]
        fr_3d    = fr.rolling(3, min_periods=1).mean()
        fr_7d    = fr.rolling(7, min_periods=1).mean()
    else:
        fr_3d = fr_7d = pd.Series(0.0, index=df.index)

    # volume_sma_ratio i volume_trend
    vol_ma     = volume.rolling(20, min_periods=1).mean()
    vol_ratio  = volume / vol_ma
    vol_trend  = rolling_slope(vol_ma, 20)

    # fear_greed
    fg_col = "fear_greed_value" if "fear_greed_value" in df.columns else \
             "fear_greed" if "fear_greed" in df.columns else None
    fg_series = df[fg_col] if fg_col else pd.Series(50.0, index=df.index)
    fg_7d_change = fg_series - fg_series.shift(7)

    # rsi_14_daily
    rsi14 = df["rsi_14"] if "rsi_14" in df.columns else pd.Series(50.0, index=df.index)

    # returns
    ret5d  = close.pct_change(5)
    ret20d = close.pct_change(20)

    features = pd.DataFrame({
        "ema_ratio_50_200":    ema_ratio,
        "ema_slope_50":        ema50_slope,
        "ema_slope_200":       ema200_slope,
        "adx_14":              adx,
        "atr_14_percentile":   atr_pct,
        "funding_rate_3d":     fr_3d,
        "funding_rate_7d":     fr_7d,
        "volume_sma_ratio":    vol_ratio,
        "volume_trend":        vol_trend,
        "fear_greed":          fg_series,
        "fear_greed_7d_change": fg_7d_change,
        "rsi_14_daily":        rsi14,
        "returns_5d":          ret5d,
        "returns_20d":         ret20d,
    }, index=df.index)

    features = features[P1_FEATURES]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    logger.info(f"P1 features computed: {len(features)} rows x {len(features.columns)} cols")
    return features


def validate_results(metrics: dict) -> bool:
    """Verifica els criteris d'èxit. Retorna True si tot passa."""
    ok = True
    mean_acc = metrics["mean_accuracy"]
    mll_std  = metrics["mlogloss_std"]

    if mean_acc < 0.55:
        logger.warning(f"Accuracy OOS {mean_acc:.3f} < 0.55 (objectiu no assolit)")
        ok = False
    else:
        logger.info(f"Accuracy OOS {mean_acc:.3f} ≥ 0.55 ✓")

    if mll_std > 0.15:
        logger.warning(f"mlogloss std {mll_std:.4f} > 0.15 (inestable entre folds)")
        ok = False
    else:
        logger.info(f"mlogloss std {mll_std:.4f} ≤ 0.15 ✓")

    return ok


def main() -> None:
    args = parse_args()

    # ── Carregar config ───────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_paths = config.get("model_paths", {})
    hmm_path = Path(model_paths.get("hmm", "models/gate_hmm.pkl"))
    xgb_path = Path(model_paths.get("xgb", "models/gate_xgb_regime.pkl"))

    # ── Pas 1: carregar dades ─────────────────────────────────────────────
    logger.info("=== PAS 1: Carregar dades diàries ===")
    df_daily = load_daily_data(args.symbol, config)
    if len(df_daily) < 500:
        logger.error(f"Dades insuficients: {len(df_daily)} files < 500. Carrega més historial.")
        sys.exit(1)

    # ── Pas 2: calcular features P1 ───────────────────────────────────────
    logger.info("=== PAS 2: Computar les 14 features de P1 ===")
    X = compute_p1_features(df_daily)
    df_hmm = df_daily.loc[X.index]  # alinear amb les features calculades

    # ── Pas 3: entrenar HMM ───────────────────────────────────────────────
    logger.info("=== PAS 3: Entrenar HMM (selecció K per BIC) ===")
    trainer = HMMTrainer(k_range=(2, 6), n_init=10, n_iter=200)
    hmm_result = trainer.fit(df_hmm)

    logger.info(f"HMM: K={hmm_result.k}  BIC scores={hmm_result.bic_scores}")
    logger.info(f"HMM: state_mapping={hmm_result.state_mapping}")

    # ── Pas 4: obtenir les etiquetes Viterbi alineades ────────────────────
    logger.info("=== PAS 4: Etiquetes Viterbi → RegimeState ===")
    # Mapejar raw HMM labels a noms de règim
    y_raw   = hmm_result.labels[-len(X):]  # alinear amb les features
    y_named = np.array([hmm_result.state_mapping.get(s, "UNCERTAIN") for s in y_raw])
    n_classes = hmm_result.k

    # Estadística de distribució d'etiquetes
    unique, counts = np.unique(y_named, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  {u}: {c} ({c/len(y_named)*100:.1f}%)")

    # ── Pas 5: entrenar XGBoost ───────────────────────────────────────────
    logger.info("=== PAS 5: Walk-forward XGBoost + Optuna ===")
    n_trials = 1 if args.no_optuna else args.n_trials
    clf = XGBRegimeClassifier(n_folds=5, n_optuna_trials=n_trials)
    metrics = clf.fit(X, y_named, n_classes=n_classes)

    # ── Pas 6: validació ──────────────────────────────────────────────────
    logger.info("=== PAS 6: Validació ===")
    passed = validate_results(metrics)
    if not passed:
        logger.warning(
            "Alguns criteris d'èxit no s'han assolit. "
            "El model es guarda igualment però revisa els resultats."
        )

    # ── Pas 7: guardar models ─────────────────────────────────────────────
    logger.info("=== PAS 7: Guardar models ===")
    HMMTrainer.save(hmm_result, hmm_path)
    clf.save(xgb_path)

    logger.info("=" * 50)
    logger.info(f"HMM  → {hmm_path}")
    logger.info(f"XGB  → {xgb_path}")
    logger.info(f"K={hmm_result.k} | Accuracy OOS={metrics['mean_accuracy']:.3f}")
    logger.info("Entrenament complet.")


if __name__ == "__main__":
    main()
