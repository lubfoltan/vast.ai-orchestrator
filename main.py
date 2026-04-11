"""X-Ray Training Orchestrator — entry point."""

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log", encoding="utf-8"),
    ],
)

from gui import run_app

if __name__ == "__main__":
    run_app()
