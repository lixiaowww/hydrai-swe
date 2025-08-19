from __future__ import annotations

"""
Generate per-fold NeuralHydrology configs for forward-chaining CV using the prepared evaluation dataset.
This script slices dates per fold and writes config files under src/neuralhydrology/cv_configs.
It does not trigger training by itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import os
import yaml


@dataclass
class FoldWindow:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def make_forward_windows(csv_path: str, n_folds: int = 5) -> List[FoldWindow]:
    df = pd.read_csv(csv_path, parse_dates=['date']).sort_values('date')
    dates = df['date'].dt.date.unique()
    if len(dates) < n_folds + 2:
        n_folds = max(2, min(3, len(dates) - 2))
    step = len(dates) // (n_folds + 1)
    windows: List[FoldWindow] = []
    for i in range(1, n_folds + 1):
        train_end = dates[step * i - 1]
        test_end = dates[step * (i + 1) - 1] if i < n_folds else dates[-1]
        train_start = dates[0]
        val_start = dates[step * i] if step * i < len(dates) else train_end
        val_end = val_start
        test_start = dates[step * i] if step * i < len(dates) else train_end
        windows.append(FoldWindow(
            train_start=str(train_start),
            train_end=str(train_end),
            val_start=str(val_start),
            val_end=str(val_end),
            test_start=str(test_start),
            test_end=str(test_end),
        ))
    return windows


def write_fold_configs(windows: List[FoldWindow], out_dir: str = "src/neuralhydrology/cv_configs") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for idx, w in enumerate(windows, 1):
        cfg = {
            'experiment_name': f'hydrai_swe_cv_fold{idx}',
            'run_dir': 'runs/{experiment_name}_{start_time}',
            'device': 'cpu',
            'dataset': 'generic',
            'data_dir': str(Path('src/neuralhydrology/data').resolve()),
            'train_basin_file': str(Path('src/neuralhydrology/data/basins.txt').resolve()),
            'validation_basin_file': str(Path('src/neuralhydrology/data/basins.txt').resolve()),
            'test_basin_file': str(Path('src/neuralhydrology/data/basins.txt').resolve()),
            'train_start_date': pd.to_datetime(w.train_start).strftime('%d/%m/%Y'),
            'train_end_date': pd.to_datetime(w.train_end).strftime('%d/%m/%Y'),
            'validation_start_date': pd.to_datetime(w.val_start).strftime('%d/%m/%Y'),
            'validation_end_date': pd.to_datetime(w.val_end).strftime('%d/%m/%Y'),
            'test_start_date': pd.to_datetime(w.test_start).strftime('%d/%m/%Y'),
            'test_end_date': pd.to_datetime(w.test_end).strftime('%d/%m/%Y'),
            'model': 'lstm',
            'hidden_size': 64,
            'dynamic_inputs': ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year'],
            'target_variables': ['streamflow_m3s'],
            'forcings': ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 'day_of_year', 'month', 'year'],
            'epochs': 5,
            'batch_size': 16,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'seq_length': 30,
            'predict_last_n': 1,
            'head': 'regression',
            'loss': 'MSE',
            'target_noise_std': 0.0,
            'initial_forget_bias': 3.0,
        }
        out_path = os.path.join(out_dir, f'fold_{idx}.yml')
        with open(out_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        paths.append(out_path)
    return paths


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate NeuralHydrology CV configs')
    parser.add_argument('--eval_csv', required=True, help='Path to timeseries_eval.csv')
    parser.add_argument('--out', default='src/neuralhydrology/cv_configs')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    wins = make_forward_windows(args.eval_csv, n_folds=args.folds)
    paths = write_fold_configs(wins, out_dir=args.out)
    print('Generated configs:')
    for p in paths:
        print(' -', p)



