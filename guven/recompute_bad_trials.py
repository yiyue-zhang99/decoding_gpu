#!/usr/bin/env python3
"""Recompute Guven bad-epoch flags at a stricter peak-to-peak threshold,
restricted to the electrode-line channels (union of L5..L1, M, R1..R5 from
guven_tw.build_electrode_lines) instead of all EEG channels, and patch them
into the cached guven_fft wave-power pickles (fft/guven/guven_tw.py output),
without re-running ICA or the wave computation.

The cleaned EEG data in EPOCH_DIR (fft/guven/preprocess.py's output) doesn't
depend on the reject threshold or channel scope -- only the is_bad_epoch flag
derived from it does. wave_trigger_computer_uniform also runs over every
trial regardless of is_bad_epoch (bad trials are carried through, not
dropped -- see guven_tw.py's docstring), so fwMax/bwMax/fwMaxSS/bwMaxSS are
unaffected by this change. Only is_bad_epoch needs to change in each cached
pickle.

Historical 250uV/150uV all-channel baselines are read back from the
per-session CSVs this script already wrote on the previous (all-channel) run
under CSV_DIR, rather than from the pkl's current is_bad_epoch -- the pkl was
already overwritten with the all-channel 150uV result by that run, so it's
no longer a valid "old" baseline to diff against.

For every session, writes:
  - a per-session diagnostic CSV (trial_idx, max_ptp_uv_lines,
    is_bad_epoch_250uv_allch, is_bad_epoch_150uv_allch,
    is_bad_epoch_150uv_lines) under CSV_DIR
  - the patched pickle (same fwMax/bwMax/etc, is_bad_epoch =
    is_bad_epoch_150uv_lines) back to CACHE_DIR, overwriting the old one
  - a summary CSV of per-session bad-trial counts under CACHE_DIR
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import EXCLUDE_THRESHOLD_CHS  # noqa: E402 -- reuse exact channel filter
from guven_tw import build_electrode_lines  # noqa: E402

EPOCH_DIR = Path("/home/dilay/project2/tw/Data/guven_data/epoch_threshold")
CACHE_DIR = Path("/home/dilay/project2/tw/results/guven_fft")
CSV_DIR = CACHE_DIR / "bad_trials_150"

NEW_THRESHOLD_UV = 150.0


def line_channel_names(ch_names: list[str]) -> list[str]:
    """Union of channel names used by any of the 11 electrode lines."""
    lines, _ = build_electrode_lines(ch_names)
    idx_set = sorted({i for line in lines for i in line})
    return [ch_names[i] for i in idx_set]


def compute_ptp(fif_path: Path, line_chs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    picks = [c for c in line_chs if c not in EXCLUDE_THRESHOLD_CHS]
    data = epochs.get_data(picks=picks, units="uV", verbose=False)
    peak_to_peak = np.ptp(data, axis=2)  # (n_trials, n_line_ch)
    return peak_to_peak.max(axis=1), np.any(peak_to_peak > NEW_THRESHOLD_UV, axis=1)


def process_one(fif_path: Path, line_chs: list[str]) -> dict | None:
    key = fif_path.stem
    pkl_path = CACHE_DIR / f"{key}.pkl"
    if not pkl_path.exists():
        print(f"[skip] {key}: no cached pkl")
        return None

    max_ptp, bad_lines_150 = compute_ptp(fif_path, line_chs)

    old_csv_path = CSV_DIR / f"{key}_bad_trials.csv"
    if old_csv_path.exists():
        old_df = pd.read_csv(old_csv_path)
        # Old-format CSVs (all-channel run) used is_bad_epoch_250/_150; new-format
        # CSVs (this script, already rerun once for this file) already carry the
        # renamed *_allch columns -- accept either so a partial rerun is safe.
        col_250 = "is_bad_epoch_250uv_allch" if "is_bad_epoch_250uv_allch" in old_df else "is_bad_epoch_250"
        col_150 = "is_bad_epoch_150uv_allch" if "is_bad_epoch_150uv_allch" in old_df else "is_bad_epoch_150"
        bad_250_allch = old_df[col_250].to_numpy(dtype=bool)
        bad_150_allch = old_df[col_150].to_numpy(dtype=bool)
    else:
        with open(pkl_path, "rb") as fp:
            bad_250_allch = np.asarray(pickle.load(fp)["is_bad_epoch"])
        bad_150_allch = np.full_like(bad_250_allch, fill_value=False)

    assert len(bad_250_allch) == len(bad_lines_150), (
        f"{key}: trial count mismatch csv={len(bad_250_allch)} fif={len(bad_lines_150)}"
    )

    with open(pkl_path, "rb") as fp:
        result = pickle.load(fp)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trial_idx": np.arange(len(bad_lines_150)),
        "max_ptp_uv_lines": max_ptp,
        "is_bad_epoch_250uv_allch": bad_250_allch,
        "is_bad_epoch_150uv_allch": bad_150_allch,
        "is_bad_epoch_150uv_lines": bad_lines_150,
    }).to_csv(old_csv_path, index=False)

    result["is_bad_epoch"] = bad_lines_150
    with open(pkl_path, "wb") as fp:
        pickle.dump(result, fp)

    n_250, n_150_allch, n_150_lines = (
        int(bad_250_allch.sum()), int(bad_150_allch.sum()), int(bad_lines_150.sum())
    )
    print(
        f"[done] {key}: bad @250uV(all-ch)={n_250} -> @150uV(all-ch)={n_150_allch} "
        f"-> @150uV(lines)={n_150_lines} (of {len(bad_lines_150)} trials)"
    )
    return {
        "key": key,
        "n_trials": len(bad_lines_150),
        "n_bad_250uv_allch": n_250,
        "n_bad_150uv_allch": n_150_allch,
        "n_bad_150uv_lines": n_150_lines,
    }


def main() -> None:
    fif_files = sorted(EPOCH_DIR.glob("sub*_session*.fif"))

    probe = mne.read_epochs(fif_files[0], preload=False, verbose=False)
    eeg_idx = mne.pick_types(probe.info, eeg=True, eog=False, misc=False)
    eeg_ch_names = [probe.ch_names[i] for i in eeg_idx]
    line_chs = line_channel_names(eeg_ch_names)
    print(f"[line channels] {len(line_chs)}: {line_chs}")

    rows = Parallel(n_jobs=8, verbose=5)(
        delayed(process_one)(f, line_chs) for f in fif_files
    )
    rows = [r for r in rows if r is not None]

    summary_path = CACHE_DIR / "bad_trials_150_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
