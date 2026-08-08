#!/usr/bin/env python3
"""Per-trial behavioural table for the Guven mental rotation/WM dataset.

ResultMatrix column layout (0-indexed), per Data/guven_data/beh.txt:
  0: acc
  1: rt
  4: cue item category (1-6)
  5: uncue item category (1-6)
  6: cue side (1=left, 2=right)
  7: left item category (0-6)
  8: right item category (0-6)
  9: rotation angle code (1-3)
  11: rotation direction (1=CW, 2=CCW, 3=none)
All other columns are undocumented and not loaded here.

Item categories 1-6 map to orientation in degrees; rotation angle codes map
to degrees of rotation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = Path("/media/dilay/T7/guven_data/Raw_Data")
EXPORTED_BEHAVIOR_DIR = Path(
    "/home/dilay/project2/tw/Data/guven_data/beh_trialwise"
)

ACC_COL = 0
RT_COL = 1
CUE_ITEM_COL = 4
UNCUE_ITEM_COL = 5
CUE_COL = 6
LEFT_ITEM_COL = 7
RIGHT_ITEM_COL = 8
ROTATION_ANGLE_COL = 9
ROTATION_DIRECTION_COL = 11

CUE_LEFT = 1
CUE_RIGHT = 2

ITEM_CATEGORY_DEG = {1: 15, 2: 45, 3: 75, 4: 105, 5: 135, 6: 165}
ROTATION_ANGLE_DEG = {1: 0, 2: 30, 3: 60}
ROTATION_DIRECTION_LABEL = {1: "CW", 2: "CCW", 3: "none"}


def _item_deg(category: np.ndarray) -> np.ndarray:
    """Map item category codes to degrees; 0 (absent item) maps to NaN."""
    out = np.full(category.shape, np.nan)
    for code, deg in ITEM_CATEGORY_DEG.items():
        out[category == code] = deg
    return out


def result_matrix_path(subj: int, sess: int) -> Path:
    return (
        RAW_DIR / f"Ppt_{subj}" / "Behavioural" / "Experiment"
        / f"Results_Gen_2_Mental_Rotation_Experiment_EEG_{subj}_{sess}_1.mat"
    )


def load_behavior_table(subj: int, sess: int) -> pd.DataFrame:
    """Return one row per trial (1-based `trial`) with the documented columns."""
    csv_path = (
        EXPORTED_BEHAVIOR_DIR / f"sub{subj:02d}_session{sess}_beh.csv"
    )
    if csv_path.is_file():
        return pd.read_csv(csv_path)

    mat_path = result_matrix_path(subj, sess)
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"Missing Guven behaviour for subject {subj:02d}, session {sess}: "
            f"neither exported CSV {csv_path} nor raw MAT {mat_path} exists. "
            "Mount the T7 data drive or export/copy the behavioural CSV files "
            "into Data/guven_data/beh_trialwise."
        )
    result_matrix = np.asarray(
        loadmat(mat_path, simplify_cells=True)["ResultMatrix"]
    )

    cue_loc = result_matrix[:, CUE_COL].astype(int)
    assert set(np.unique(cue_loc)) <= {CUE_LEFT, CUE_RIGHT}, (
        f"unexpected cue_loc values in {mat_path.name}: {np.unique(cue_loc)}"
    )
    cue_item = result_matrix[:, CUE_ITEM_COL].astype(int)
    uncue_item = result_matrix[:, UNCUE_ITEM_COL].astype(int)
    rotation_angle_code = result_matrix[:, ROTATION_ANGLE_COL].astype(int)
    rotation_direction_code = result_matrix[:, ROTATION_DIRECTION_COL].astype(int)

    n_trials = result_matrix.shape[0]
    table = pd.DataFrame(
        {
            "subj": subj,
            "sess": sess,
            "trial": np.arange(1, n_trials + 1),
            "acc": result_matrix[:, ACC_COL].astype(int),
            "rt": result_matrix[:, RT_COL].astype(float),
            "cue_loc": cue_loc,
            "cue_item": cue_item,
            "cue_item_deg": _item_deg(cue_item),
            "uncue_item": uncue_item,
            "uncue_item_deg": _item_deg(uncue_item),
            "left_item": result_matrix[:, LEFT_ITEM_COL].astype(int),
            "right_item": result_matrix[:, RIGHT_ITEM_COL].astype(int),
            "rotation_angle_code": rotation_angle_code,
            "rotation_angle_deg": [
                ROTATION_ANGLE_DEG[code] for code in rotation_angle_code
            ],
            "rotation_direction_code": rotation_direction_code,
            "rotation_direction": [
                ROTATION_DIRECTION_LABEL[code] for code in rotation_direction_code
            ],
        }
    )
    return table


def item_theta_rad(category: np.ndarray) -> np.ndarray:
    """Doubled-angle radians for a cue/uncue item category array (1-6)."""
    degrees = _item_deg(np.asarray(category))
    if np.any(np.isnan(degrees)):
        raise ValueError("item category contains values outside 1-6")
    return np.deg2rad(2 * degrees)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/dilay/project2/tw/Data/guven_data/beh_trialwise"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for subject in args.subjects:
        for session in args.sessions:
            path = result_matrix_path(subject, session)
            if not path.is_file():
                print(f"[skip] missing {path}")
                continue
            table = load_behavior_table(subject, session)
            out_path = args.out_dir / f"sub{subject:02d}_session{session}_beh.csv"
            table.to_csv(out_path, index=False)
            tables.append(table)
            print(f"[done] {out_path}")

    if tables:
        combined = pd.concat(tables, ignore_index=True)
        combined_path = args.out_dir / "all_subjects_beh.csv"
        combined.to_csv(combined_path, index=False)
        print(f"[done] {combined_path} ({len(combined)} rows)")
