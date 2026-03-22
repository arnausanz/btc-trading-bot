#!/usr/bin/env python3
"""
scripts/diagnose_demo.py
────────────────────────
Diagnòstic complet del DemoRunner sense arrencar-lo.

Comprova, pas a pas, per quina raó falla la càrrega de cada bot:
  1. Imports de paquets ML
  2. discover_configs (working directory)
  3. _build_ml_registry (imports dinàmics dels models)
  4. Càrrega de cada fitxer .pkl
  5. Instanciació de cada bot (MLBot, EnsembleBot…)

Executa des de l'arrel del projecte:
    python scripts/diagnose_demo.py
    python scripts/diagnose_demo.py --config config/demo.yaml
"""
import argparse
import importlib
import logging
import os
import pickle
import sys
import traceback

import yaml

# ── Colors ANSI ───────────────────────────────────────────────────────────────
OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
INFO = "\033[94m·\033[0m"

logging.basicConfig(level=logging.WARNING)  # silencia els loggers interns


def section(title: str) -> None:
    print(f"\n\033[1m{'─'*60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'─'*60}\033[0m")


def ok(msg: str)   -> None: print(f"  {OK}  {msg}")
def fail(msg: str) -> None: print(f"  {FAIL}  {msg}")
def warn(msg: str) -> None: print(f"  {WARN}  {msg}")
def info(msg: str) -> None: print(f"  {INFO}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Entorn
# ══════════════════════════════════════════════════════════════════════════════
def check_environment() -> None:
    section("1. Entorn")
    info(f"Python:           {sys.version}")
    info(f"Executable:       {sys.executable}")
    info(f"Working dir:      {os.getcwd()}")
    info(f"Script dir:       {os.path.dirname(os.path.abspath(__file__))}")

    # Comprova que estem a l'arrel del projecte
    if os.path.isdir("config/models") and os.path.isfile("config/demo.yaml"):
        ok("Working directory és l'arrel del projecte")
    else:
        fail("Working directory NO és l'arrel del projecte!")
        fail(f"  Esperat: carpeta amb config/models/ i config/demo.yaml")
        fail(f"  Actual:  {os.getcwd()}")
        print("\n  → Executa des de l'arrel: cd /ruta/projecte && python scripts/diagnose_demo.py")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Paquets Python
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_PACKAGES = [
    ("sklearn",    "scikit-learn"),
    ("xgboost",    "xgboost"),
    ("lightgbm",   "lightgbm"),
    ("catboost",   "catboost"),
    ("torch",      "torch (PyTorch)"),
    ("tqdm",       "tqdm"),
    ("numpy",      "numpy"),
    ("pandas",     "pandas"),
    ("yaml",       "pyyaml"),
]


def check_packages() -> dict[str, bool]:
    section("2. Paquets Python")
    results: dict[str, bool] = {}
    for pkg, label in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{label:<25} v{ver}")
            results[pkg] = True
        except Exception as e:
            fail(f"{label:<25} ERROR: {type(e).__name__}: {e}")
            results[pkg] = False
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. discover_configs
# ══════════════════════════════════════════════════════════════════════════════
def check_discover_configs() -> dict[str, str]:
    section("3. discover_configs('ML')")
    try:
        from core.config_utils import discover_configs
        ml_configs = discover_configs("ML")
        if ml_configs:
            ok(f"Trobats {len(ml_configs)} configs ML:")
            for stem, path in ml_configs.items():
                info(f"  {stem:<20} → {path}")
        else:
            fail("No s'ha trobat cap config ML!")
            fail("  Causa probable: working directory incorrecte o YAML sense 'category: ML'")
        return ml_configs
    except Exception as e:
        fail(f"Error important a discover_configs: {e}")
        traceback.print_exc()
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 4. _build_ml_registry
# ══════════════════════════════════════════════════════════════════════════════
def check_ml_registry() -> dict:
    section("4. _build_ml_registry (imports dinàmics)")
    try:
        from core.config_utils import discover_configs
        import importlib as _imp
        registry: dict = {}
        for stem, path in discover_configs("ML").items():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            mt       = cfg.get("model_type")
            mod_path = cfg.get("module")
            cls_name = cfg.get("class_name")
            if not (mt and mod_path and cls_name):
                warn(f"{stem}: sense module/class_name — saltat")
                continue
            try:
                mod = _imp.import_module(mod_path)
                cls = getattr(mod, cls_name)
                registry[mt] = cls
                ok(f"{mt:<20} → {mod_path}.{cls_name}")
            except Exception as e:
                fail(f"{mt:<20} → IMPORT FAILED: {e}")
        if not registry:
            fail("_MODEL_REGISTRY buida — cap bot ML podrà carregar-se!")
        return registry
    except Exception as e:
        fail(f"Error important: {e}")
        traceback.print_exc()
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Fitxers de model (.pkl)
# ══════════════════════════════════════════════════════════════════════════════
MODEL_FILES = [
    "models/random_forest_v1.pkl",
    "models/xgboost_v1.pkl",
    "models/gate_hmm.pkl",
    "models/gate_xgb_regime.pkl",
]


def check_model_files() -> None:
    section("5. Fitxers de model (.pkl / .pt / .zip)")
    for path in MODEL_FILES:
        if not os.path.exists(path):
            fail(f"{path}  →  NO EXISTEIX")
            continue
        size_kb = os.path.getsize(path) / 1024
        if path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                ok(f"{path}  ({size_kb:.0f} KB)  →  {keys}")
            except Exception as e:
                fail(f"{path}  ({size_kb:.0f} KB)  →  PICKLE ERROR: {e}")
        else:
            ok(f"{path}  ({size_kb:.0f} KB)  →  existeix")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Instanciació de cada bot enabled a demo.yaml
# ══════════════════════════════════════════════════════════════════════════════
_ML_TYPES = {"random_forest", "xgboost", "lightgbm", "catboost", "gru", "patchtst", "tft"}
_RL_TYPES = {
    "ppo", "sac",
    "ppo_professional", "sac_professional",
    "td3_professional", "td3_multiframe",
}


def _load_bot(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config.get("model_type", "")

    if model_type in _ML_TYPES:
        from bots.ml.ml_bot import MLBot
        return MLBot(config_path=config_path)

    if model_type in _RL_TYPES:
        from bots.rl.rl_bot import RLBot
        return RLBot(config_path=config_path)

    module_path = config.get("module")
    class_name  = config.get("class_name")
    if module_path and class_name:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(config_path=config_path)

    raise ValueError(
        f"No es pot detectar el tipus de bot per '{config_path}'. "
        "El YAML ha de tenir 'module' + 'class_name' o un model_type reconegut."
    )


def check_bot_loading(demo_config_path: str) -> None:
    section(f"6. Instanciació de bots ({demo_config_path})")
    with open(demo_config_path) as f:
        demo_cfg = yaml.safe_load(f)

    bots_cfg = demo_cfg.get("bots", [])
    enabled  = [b for b in bots_cfg if b.get("enabled", True)]
    info(f"Bots enabled: {len(enabled)} de {len(bots_cfg)}")

    loaded = []
    for entry in enabled:
        cp = entry["config_path"]
        try:
            bot = _load_bot(cp)
            ok(f"{bot.bot_id:<20} ← {cp}")
            loaded.append(bot.bot_id)
        except Exception as e:
            fail(f"{'?':<20} ← {cp}")
            # Mostrar traceback complet per veure exactament on falla
            lines = traceback.format_exc().strip().split("\n")
            for line in lines:
                print(f"              {line}")

    print()
    if loaded:
        ok(f"Bots carregats ({len(loaded)}): {', '.join(loaded)}")
    else:
        fail("Cap bot carregat!")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnòstic del DemoRunner")
    parser.add_argument(
        "--config", default="config/demo.yaml",
        help="Ruta al demo.yaml (default: config/demo.yaml)"
    )
    args = parser.parse_args()

    print("\n\033[1m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1m║          DIAGNÒSTIC DEMO RUNNER — BTC Trading Bot        ║\033[0m")
    print("\033[1m╚══════════════════════════════════════════════════════════╝\033[0m")

    check_environment()
    pkg_ok = check_packages()
    check_discover_configs()
    check_ml_registry()
    check_model_files()
    check_bot_loading(args.config)

    print("\n\033[1m─────────────────────────────────────────────────────────────\033[0m")
    print("\033[1m  Fi del diagnòstic. Revisa les ✗ per trobar el problema.\033[0m")
    print("\033[1m─────────────────────────────────────────────────────────────\033[0m\n")


if __name__ == "__main__":
    main()
