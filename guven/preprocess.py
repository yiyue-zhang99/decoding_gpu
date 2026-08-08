#!/usr/bin/env python3
"""Preprocess Guven Mental Rotation/WM BrainVision files through ICA ocular cleaning.

Trigger codes (see /media/dilay/T7/guven_data/Filter_Gen_2_Mental_rotation.m for the
original MATLAB pipeline these mirror):
    100         target (epoch anchor, t=0) -- the only code this script uses
    101/102     cue                 (target + 0.75s)
    91          impulse1            (target + 1.85s)
    151/152/153/213  rotation       (target + 2.45s)
    92          impulse2            (target + 3.55s)
    201/202     probe               (target + 4.15s)
Only target(100) is consumed here (via mne.events_from_annotations) to anchor
epochs; the other codes aren't re-derived or re-checked on every run because
trigger correctness for the whole dataset (120 files, 43,200/43,201 trials)
was already verified independently and doesn't change between runs:
  - check_raw_triggers.py: zero missing/duplicate cue/impulse1/rotation/
    impulse2/probe events per trial (self-consistency check).
  - check_eeg_vs_behavior_triggers.py: every trigger value matches the
    ground-truth behavioral Trigger_Matrix exactly for 119/120 sessions.
  - check_timing_from_raw.py: only 1/43,200 trials fell outside a 50ms timing
    tolerance, and that one is a floating-point boundary artifact (probe at
    4.200s vs. 4.15s expected -- exactly the +/-0.05s cutoff), not a real
    timing problem.
The one real anomaly in the whole dataset is subject 21 session 1, which has
one extra leading EEG trial with no behavioral counterpart -- the original
MATLAB script independently hardcodes the same fix (`if sub==21 ...
pop_select(..., 'notrial', 1)`). See SKIP_TRIALS below.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


RAW_DIR = Path("/media/dilay/T7/guven_data/Raw_Data")
MONTAGE_NAME = "standard_1020"
OUT_ROOT = Path("/home/dilay/project2/tw/Data/guven_data")
EPOCH_THRESHOLD_DIR = OUT_ROOT / "epoch_threshold"
BAD_TRIALS_DIR = OUT_ROOT / "bad_trials"
# Bilateral mastoid reference (matches the original MATLAB pipeline's
# pop_reref(EEG, [1 2])).
MASTOID_CHS = ["A1", "A2"]
EXCLUDE_THRESHOLD_CHS: set[str] = set()

L_FREQ = 0.1
H_FREQ = 40.0
ICA_FIT_L_FREQ = 1.0
TARGET_SFREQ = 500.0
# Min inter-target gap across the whole dataset is 5.718s (checked empirically).
# -1.2/+4.4 covers baseline through the probe marker (~4.12-4.20s) with margin
# on both sides without spilling into the neighboring trial.
TMIN = -1.2
TMAX = 5.5
RANDOM_STATE = 42

TARGET_CODE = 100
EPOCH_EVENT_ID = {"target": 1}

# subject_session -> number of leading EEG target-trials to drop before they
# have no behavioral counterpart (confirmed via check_eeg_vs_behavior_triggers.py;
# the original MATLAB pipeline hardcodes the identical fix for this subject).
SKIP_TRIALS: dict[str, int] = {
    "sub21_session1": 1,
}


def output_key(path: Path) -> str:
    match = re.search(r"WM_(\d+)_(\d+)$", path.stem)
    if not match:
        return path.stem
    subject, session = match.groups()
    return f"sub{int(subject):02d}_session{session}"


def prepare_raw(path: Path, montage_name: str) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")

    eog_found = [ch for ch in raw.ch_names if ch.upper().startswith("EOG")]
    if eog_found:
        raw.set_channel_types({ch: "eog" for ch in eog_found}, verbose=False)

    # raw's channel names are inconsistently cased (e.g. "FCZ", "OZ"); the montage
    # expects its own exact casing (e.g. "FCz", "Oz"), so look each one up case-
    # insensitively and rename to whatever casing the montage uses.
    montage = mne.channels.make_standard_montage(montage_name)
    montage_lookup = {name.lower(): name for name in montage.ch_names}
    rename_map = {}
    for ch in raw.ch_names:
        if ch in eog_found:
            continue
        if ch.lower() in montage_lookup:
            rename_map[ch] = montage_lookup[ch.lower()]
    raw.rename_channels(rename_map)
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    raw.filter(L_FREQ, None, picks=["eeg", "eog"], verbose=False)
    raw.filter(None, H_FREQ, picks=["eeg", "eog"], verbose=False)
    return raw


def fit_ica_and_find_eog(
    raw: mne.io.BaseRaw,
    eog_chs: list[str],
    n_components: float | int | None,
) -> tuple[ICA, list[int], dict[str, list[int]]]:
    fit_raw = raw.copy().filter(ICA_FIT_L_FREQ, None, picks="eeg", verbose=False)
    ica = ICA(
        n_components=n_components,
        method="picard",
        fit_params={"extended": True},
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(fit_raw, picks="eeg", verbose=False)

    eog_component_map: dict[str, list[int]] = {}
    excluded: set[int] = set()
    for ch in eog_chs:
        if ch not in raw.ch_names:
            continue
        inds, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
        eog_component_map[ch] = [int(ind) for ind in inds]
        excluded.update(int(ind) for ind in inds)

    ica.exclude = sorted(excluded)
    return ica, ica.exclude, eog_component_map


def find_threshold_bad_epochs(epochs: mne.Epochs, eeg_reject_uv: float) -> np.ndarray:
    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, misc=False)
    eeg_picks = [
        pick for pick in eeg_picks
        if epochs.ch_names[pick] not in EXCLUDE_THRESHOLD_CHS
    ]
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found for threshold rejection.")

    data = epochs.get_data(picks=eeg_picks, units="uV", verbose=False)
    peak_to_peak = np.ptp(data, axis=2)
    return np.any(peak_to_peak > eeg_reject_uv, axis=1)


def make_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    reject_method: str,
    eeg_reject_uv: float,
) -> tuple[mne.Epochs, list[int]]:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EPOCH_EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        picks=["eeg", "eog"],
        reject=None,
        reject_by_annotation=False,
        verbose=False,
    )
    # No trial in this dataset sits close enough to a recording's start/end for
    # mne.Epochs to silently drop it for running past the boundary (checked the
    # worst-case margin across all 120 files: >3.6s to spare on every edge), so
    # epochs here always has one entry per row of `events`, in order.
    assert len(epochs) == len(events), "unexpected boundary-dropped epoch(s)"

    if reject_method == "peak_to_peak":
        bad_mask = find_threshold_bad_epochs(epochs, eeg_reject_uv)
    elif reject_method == "autoreject":
        try:
            from autoreject import AutoReject
        except ImportError as exc:
            raise RuntimeError("autoreject is not installed in this environment.") from exc
        ar = AutoReject(
            picks=mne.pick_types(epochs.info, eeg=True),
            n_jobs=1,
            random_state=RANDOM_STATE,
            n_interpolate=[0],
            verbose=False,
        )
        ar.fit(epochs)
        bad_mask = ar.get_reject_log(epochs).bad_epochs
    else:
        raise ValueError(f"Unknown reject method: {reject_method}")

    bad_trials = [int(idx) for idx in np.flatnonzero(bad_mask)]
    epochs.metadata = pd.DataFrame({"is_bad_epoch": bad_mask})
    return epochs, bad_trials


def write_bad_trials_csv(bad_trials_path: Path, bad_mask: np.ndarray) -> None:
    BAD_TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trial_idx": np.arange(len(bad_mask)),
        "is_bad_epoch": bad_mask.astype(bool),
    }).to_csv(bad_trials_path, index=False)


def process_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    key = output_key(path)
    epoch_prefix = EPOCH_THRESHOLD_DIR / key
    epochs_clean_path = epoch_prefix.with_suffix(".fif")
    bad_trials_path = BAD_TRIALS_DIR / f"{key}_bad_trials.csv"

    if epochs_clean_path.exists() and not args.overwrite:
        if not bad_trials_path.exists():
            # fif already produced by an earlier run (before this file existed) --
            # backfill the small CSV from the metadata already stored in the fif
            # instead of redoing ICA/epoch rejection.
            epochs_existing = mne.read_epochs(epochs_clean_path, preload=False, verbose="ERROR")
            bad_mask = epochs_existing.metadata["is_bad_epoch"].to_numpy()
            write_bad_trials_csv(bad_trials_path, bad_mask)
            print(f"[backfill] {key}: wrote {bad_trials_path.name} from existing fif")
            return {"file": path.name, "key": key, "status": "backfilled_bad_trials"}
        print(f"[skip] {key} already done")
        return {"file": path.name, "key": key, "status": "skipped"}

    EPOCH_THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[start] {key}")

    raw = prepare_raw(path, args.montage)
    raw.resample(args.sfreq, verbose=False)
    raw.set_eeg_reference(ref_channels=MASTOID_CHS, projection=False, verbose=False)

    target_events, _ = mne.events_from_annotations(
        raw, event_id={f"Stimulus/S{TARGET_CODE}": EPOCH_EVENT_ID["target"]}, verbose=False
    )
    n_drop_leading = SKIP_TRIALS.get(key, 0)
    if n_drop_leading:
        print(f"[{key}] dropping {n_drop_leading} spurious leading trial(s) with no behavioral counterpart")
        target_events = target_events[n_drop_leading:]

    ica, excluded, eog_component_map = fit_ica_and_find_eog(
        raw,
        eog_chs=args.eog_chs,
        n_components=args.ica_n_components,
    )

    raw_clean = raw.copy()
    ica.apply(raw_clean, exclude=excluded, verbose=False)

    epochs_clean, bad_trials = make_epochs(
        raw_clean,
        target_events,
        reject_method=args.bad_epoch_method,
        eeg_reject_uv=args.eeg_reject_uv,
    )
    epochs_clean.save(epochs_clean_path, overwrite=True, verbose=False)
    write_bad_trials_csv(bad_trials_path, epochs_clean.metadata["is_bad_epoch"].to_numpy())

    summary = {
        "file": path.name,
        "key": key,
        "status": "done",
        "n_target_events": len(target_events),
        "n_epochs_saved": len(epochs_clean),
        "ica_excluded_components": excluded,
        "ica_eog_component_map": eog_component_map,
        "bad_epoch_method": args.bad_epoch_method,
        "eeg_reject_uv": args.eeg_reject_uv,
        "bad_trial_indices_zero_based": bad_trials,
        "epochs_threshold": str(epochs_clean_path),
        "bad_trials_csv": str(bad_trials_path),
        "runtime_s": round(time.time() - t0, 3),
    }

    print(
        f"[done] {key}: epochs={len(epochs_clean)}, "
        f"ICA exclude={excluded}, {time.time() - t0:.1f}s"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess Guven mental rotation/WM BrainVision files and remove EOG-related ICA components."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional .vhdr files. Defaults to all files under --raw-dir.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--montage", type=str, default=MONTAGE_NAME)
    parser.add_argument("--sfreq", type=float, default=TARGET_SFREQ)
    parser.add_argument("--eog-chs", nargs="*", default=["EOGvl", "EOGh"])
    parser.add_argument("--ica-n-components", type=int, default=30)
    parser.add_argument(
        "--bad-epoch-method",
        choices=["peak_to_peak", "autoreject"],
        default="peak_to_peak",
        help="Bad epoch flagging method after ICA cleaning. Epochs are flagged, not dropped.",
    )
    parser.add_argument("--eeg-reject-uv", type=float, default=250.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of files to process in parallel.")
    return parser


def process_file_safely(path: Path, args: argparse.Namespace) -> dict[str, object]:
    try:
        return process_file(path, args)
    except Exception as exc:
        import traceback

        print(f"[error] {path.name}: {exc}")
        return {
            "file": path.name,
            "key": output_key(path),
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mne.set_log_level("WARNING")

    paths = args.paths or sorted(
        args.raw_dir.glob("Ppt_*/EEG/*.vhdr"),
        key=lambda path: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    if args.limit is not None:
        paths = paths[: args.limit]

    from joblib import Parallel, delayed, parallel_config

    inner_threads = max(1, (os.cpu_count() or 1) // args.n_jobs)
    print(
        f"Processing {len(paths)} file(s) with n_jobs={args.n_jobs} "
        f"(inner_max_num_threads={inner_threads})"
    )
    with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
        summary_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(process_file_safely)(path, args) for path in paths
        )

    summary_path = OUT_ROOT / "guven_threshold_preprocess_summary.csv"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
