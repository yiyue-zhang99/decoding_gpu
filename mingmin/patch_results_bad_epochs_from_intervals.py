"""Synchronize cached Mingmin FFT bad-epoch flags with intervals CSV files.

The pickle trial order is identified by its ``trial_index`` field, so this
script maps the CSV ``is_bad_epoch`` flag by trial index rather than CSV row.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def interval_path(interval_dir: Path, key: str) -> Path:
    path = interval_dir / f"{key}_intervals.csv"
    if path.exists():
        return path

    # Historical filename for sub30/session3.
    path = interval_dir / f"{key}.2_intervals.csv"
    if path.exists():
        return path
    raise FileNotFoundError(f"No intervals CSV for {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--interval-dir", type=Path, required=True)
    args = parser.parse_args()

    changed = 0
    checked = 0
    for result_path in sorted(args.results_dir.glob("subj*_sess*.pkl")):
        match = re.fullmatch(r"subj(\d+)_sess(\d+)\.pkl", result_path.name)
        if match is None:
            continue
        key = f"sub{int(match.group(1)):02d}_session{int(match.group(2))}"

        with result_path.open("rb") as f:
            result = pickle.load(f)
        intervals = pd.read_csv(interval_path(args.interval_dir, key))
        flags_by_trial = dict(
            zip(
                intervals["trial_index"].astype(int),
                intervals["is_bad_epoch"].astype(bool),
                strict=True,
            )
        )
        trials = np.asarray(result["trial_index"], dtype=int)
        missing = sorted(set(trials) - set(flags_by_trial))
        if missing:
            raise ValueError(f"{result_path.name}: missing trial indices {missing}")
        replacement = np.asarray([flags_by_trial[trial] for trial in trials], dtype=bool)
        current = np.asarray(result["is_bad_epoch"], dtype=bool)
        if current.shape != replacement.shape:
            raise ValueError(
                f"{result_path.name}: is_bad_epoch shape {current.shape} does not "
                f"match trial_index shape {replacement.shape}"
            )
        if not np.array_equal(current, replacement):
            result["is_bad_epoch"] = replacement
            with result_path.open("wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            changed += 1
        checked += 1

    print(f"Checked {checked} result files; updated {changed} files.")


if __name__ == "__main__":
    main()
