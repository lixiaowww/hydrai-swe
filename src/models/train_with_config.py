from __future__ import annotations

from pathlib import Path
import argparse
import os

from neuralhydrology import nh_run
from neuralhydrology.utils.config import Config


def train_with_config(config_path: str) -> None:
    cfg = Config(Path(config_path))
    nh_run.start_training(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralHydrology with a specific config file")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    train_with_config(args.config)



