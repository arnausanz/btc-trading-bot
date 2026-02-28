# scripts/run_demo.py
import logging
import sys

sys.path.append(".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from core.engine.demo_runner import DemoRunner

if __name__ == "__main__":
    runner = DemoRunner(config_path="config/demo.yaml")
    runner.run()