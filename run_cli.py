"""Headless CLI runner — starts the full training pipeline without the GUI.

Usage
-----
    py run_cli.py                          # use everything from user_config.json
    py run_cli.py --no-split               # override: pipeline splits dataset (pre_split_data=False)
    py run_cli.py --train 0.7 --val 0.15 --test 0.15
    py run_cli.py --epochs 50 --seed 42
    py run_cli.py --dry-run                # print the train.py command only, no renting

All base settings are read from user_config.json. CLI flags override individual values.
"""

import argparse
import json
import logging
import os
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrator.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "user_config.json")


def _load_json() -> dict:
    if not os.path.isfile(_CONFIG_PATH):
        sys.exit(f"[ERROR] user_config.json not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Vast.ai training pipeline from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Split options
    split = p.add_mutually_exclusive_group()
    split.add_argument(
        "--split", action="store_true", default=None,
        help="Force pipeline to split the dataset (pre_split_data=False).",
    )
    split.add_argument(
        "--no-split", dest="split", action="store_false",
        help="Use pre-split train/val/test subfolders (pre_split_data=True).",
    )

    p.add_argument("--train", type=float, metavar="RATIO",
                   help="Train split ratio, e.g. 0.7")
    p.add_argument("--val",   type=float, metavar="RATIO",
                   help="Validation split ratio, e.g. 0.15")
    p.add_argument("--test",  type=float, metavar="RATIO",
                   help="Test split ratio, e.g. 0.15")
    p.add_argument("--seed",  type=int,   help="Random seed")

    p.add_argument("--epochs",     type=int,   help="Number of training epochs")
    p.add_argument("--lr",         type=float, help="Learning rate")
    p.add_argument("--batch-size", type=int,   help="Batch size")
    p.add_argument("--model",      type=str,
                   choices=["ResNet-50", "DenseNet-121", "EfficientNet-B0", "ConvNeXt", "Custom CNN"],
                   help="Model architecture")

    p.add_argument("--data-path",   type=str, help="Override data_path")
    p.add_argument("--output-path", type=str, help="Override output_path")

    p.add_argument("--max-price",   type=float, help="Max price per hour (USD)")
    p.add_argument("--min-gpu-ram", type=float, help="Min GPU RAM (GB)")

    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the train.py command that would be run, then exit. No instance is rented.",
    )
    return p


def _merge(raw: dict, args: argparse.Namespace) -> dict:
    """Merge CLI overrides into the raw JSON dict (in-place copy)."""
    cfg = dict(raw)

    if args.split is not None:
        # --split  → pre_split_data=False (pipeline splits)
        # --no-split → pre_split_data=True (already split)
        cfg["pre_split_data"] = not args.split

    if args.train is not None:
        cfg["train_split"] = str(args.train)
    if args.val is not None:
        cfg["val_split"] = str(args.val)
    if args.test is not None:
        cfg["test_split"] = str(args.test)

    if args.seed is not None:
        cfg["seed"] = str(args.seed)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["learning_rate"] = str(args.lr)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.model is not None:
        cfg["model"] = args.model

    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.output_path is not None:
        cfg["output_path"] = args.output_path
    if args.max_price is not None:
        cfg["max_price"] = str(args.max_price)
    if args.min_gpu_ram is not None:
        cfg["min_gpu_ram"] = str(args.min_gpu_ram)

    return cfg


def _build_experiment_config(cfg: dict):
    from config import ExperimentConfig

    MODEL_MAP = {
        "ResNet-50":      "resnet50",
        "DenseNet-121":   "densenet121",
        "EfficientNet-B0": "efficientnet_b0",
        "ConvNeXt":       "convnext",
        "Custom CNN":     "custom_cnn",
    }

    ec = ExperimentConfig()
    ec.api_key       = cfg.get("api_key", "")
    ec.ssh_key_path  = cfg.get("ssh_key_path", "")
    ec.data_path     = cfg.get("data_path", "")
    ec.test_data_path = cfg.get("test_data_path", "")
    ec.output_path   = cfg.get("output_path", "")
    ec.custom_script_path = cfg.get("custom_script_path", "")

    ec.task_type     = cfg.get("task_type", "Classification").lower()
    ec.use_builtin   = bool(cfg.get("use_builtin", False))
    ec.pre_split_data = bool(cfg.get("pre_split_data", False))

    model_label      = cfg.get("model", "ResNet-50")
    ec.model_name    = MODEL_MAP.get(model_label, "resnet50")
    ec.optimizer     = cfg.get("optimizer", "AdamW")

    try:
        ec.learning_rate = float(cfg.get("learning_rate", 0.001))
    except (ValueError, TypeError):
        ec.learning_rate = 0.001

    ec.batch_size = int(cfg.get("batch_size", 32))
    ec.epochs     = int(cfg.get("epochs", 50))

    try:
        ec.train_split = float(cfg.get("train_split", 0.7))
    except (ValueError, TypeError):
        ec.train_split = 0.7
    try:
        ec.val_split = float(cfg.get("val_split", 0.15))
    except (ValueError, TypeError):
        ec.val_split = 0.15
    try:
        ec.test_split = float(cfg.get("test_split", 0.15))
    except (ValueError, TypeError):
        ec.test_split = 0.15
    try:
        ec.seed = int(cfg.get("seed", 42))
    except (ValueError, TypeError):
        ec.seed = 42

    try:
        ec.min_gpu_ram = float(cfg.get("min_gpu_ram", 8.0))
    except (ValueError, TypeError):
        ec.min_gpu_ram = 8.0
    try:
        ec.max_price = float(cfg.get("max_price", 1.0))
    except (ValueError, TypeError):
        ec.max_price = 1.0
    try:
        ec.max_samples_per_class = int(cfg.get("max_samples_per_class", 0))
    except (ValueError, TypeError):
        ec.max_samples_per_class = 0

    return ec


def _print_summary(ec) -> None:
    print()
    print("=" * 62)
    print("  TRAINING CONFIG SUMMARY")
    print("=" * 62)
    print(f"  Task            : {ec.task_type}")
    print(f"  Model           : {ec.model_name}")
    print(f"  Optimizer       : {ec.optimizer}  lr={ec.learning_rate}")
    print(f"  Batch / Epochs  : {ec.batch_size} / {ec.epochs}")
    print(f"  Seed            : {ec.seed}")
    print(f"  Data path       : {ec.data_path}")
    if ec.test_data_path:
        print(f"  Test data path  : {ec.test_data_path}")
    if ec.pre_split_data:
        print(f"  Split mode      : pre-split (train/val/test subfolders)")
    else:
        tr, vl, ts = ec.train_split, ec.val_split, ec.test_split
        print(f"  Split ratio     : train={tr}  val={vl}  test={ts}")
    print(f"  Max price       : ${ec.max_price}/h   min GPU RAM: {ec.min_gpu_ram} GB")
    print(f"  Output path     : {ec.output_path}")
    print("=" * 62)
    print()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    raw = _load_json()
    merged = _merge(raw, args)
    ec = _build_experiment_config(merged)

    _print_summary(ec)

    if args.dry_run:
        print("[DRY RUN] train.py command that would run on the remote instance:")
        print()
        print(ec.build_train_command())
        print()
        return

    # Validate ratios when pipeline does the splitting
    if not ec.pre_split_data and not ec.use_builtin:
        total = round(ec.train_split + ec.val_split + ec.test_split, 6)
        if abs(total - 1.0) > 1e-4:
            sys.exit(
                f"[ERROR] Split ratios must sum to 1.0 "
                f"(train={ec.train_split} + val={ec.val_split} + test={ec.test_split} = {total})"
            )

    # Validate paths
    if not ec.api_key:
        sys.exit("[ERROR] api_key is empty in user_config.json")
    if not ec.use_builtin and not ec.data_path:
        sys.exit("[ERROR] data_path is empty in user_config.json")

    from orchestrator import Orchestrator

    done_event = threading.Event()
    error_holder = [None]

    def _log(msg: str) -> None:
        logger.info(msg)

    def _run():
        try:
            orch = Orchestrator(ec, log_cb=_log)
            orch.run()
        except Exception as exc:
            error_holder[0] = exc
        finally:
            done_event.set()

    t = threading.Thread(target=_run, daemon=False)
    t.start()

    try:
        t.join()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. The remote instance may still be running.")
        print("    Check vast.ai console to terminate it manually.")
        sys.exit(1)

    if error_holder[0]:
        print(f"\n[ERROR] Pipeline failed: {error_holder[0]}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
