#!/usr/bin/env python3
"""Travelling-wave FFT analysis for Mingmin sequence epochs."""

from __future__ import annotations

import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, "/home/dilay/project2/tw/travelling_waves/tw/tw_fft/func")
from wave_ana_tw_power2 import wave_trigger_computer_uniform


EPOCH_CLEAN_DIR = Path("/home/dilay/project2/tw/Data/Mingmin_data/sequency/epoch_autoreject")
BEHAVIOR_CSV_CACHE = Path("/home/dilay/project2/tw/Data/Mingmin_data/sequency/behavior_csv")
# All trials are kept for the FFT/TW computation now (see align_epochs_to_intervals);
# eeg_bad / timing-issue trials are only flagged, never dropped here. The cache dir
# name is changed from the old *_all200/_clean dirs so stale filtered-trial pickles
# are never mistaken for the new all-trials-kept schema.
if EPOCH_CLEAN_DIR.name == "epoch_auto":
    CACHE_DIR = Path("/home/dilay/project2/tw/results/mingmin_sequence_tw_auto_alltrials")
else:
    CACHE_DIR = Path("/home/dilay/project2/tw/results/mingmin_sequence_tw_clean")

TOL_S = 0.05
EXPECTED_TIMES = {
    "stim2_rel_s": 1.15,
    "impulse1_rel_s": 2.30,
    "probe1_rel_s": 2.90,
    "impulse2_rel_s": 4.45,
    "probe2_rel_s": 5.05,
}


def build_electrode_lines(ch_names):
    M  = ["8Z", "7Z", "6Z", "5Z", "4Z", "3Z", "2Z"]

    L1 = ["8L", "7L", "6L", "5L", "4L", "3Z", "2Z"]
    L2 = ["8L", "7L", "6L", "5L", "4L", "3L", "2L"]
    L3 = ["5LC", "5LB", "3LA", "2LA", "4L", "3L", "2L"]
    L4 = ["5LC", "5LB", "3LA", "2LA", "1LA", "2L"]
    L5 = ["5LC", "5LB", "4LB", "3LB", "2LB", "1LB"]

    R1 = ["8R", "7R", "6R", "5R", "4R", "3Z", "2Z"]
    R2 = ["8R", "7R", "6R", "5R", "4R", "3R", "2R"]
    R3 = ["5RC", "5RB", "3RA", "2RA", "4R", "3R", "2R"]
    R4 = ["5RC", "5RB", "3RA", "2RA", "1RA", "2R"]
    R5 = ["5RC", "5RB", "4RB", "3RB", "2RB", "1RB"]

    lines = [L5, L4, L3, L2, L1, M, R1, R2, R3, R4, R5]
    names = ["L5", "L4", "L3", "L2", "L1", "M", "R1", "R2", "R3", "R4", "R5"]
    name2idx = {ch: idx for idx, ch in enumerate(ch_names)}

    electrode_lines = []
    for name, line in zip(names, lines):
        missing = [ch for ch in line if ch not in name2idx]
        if missing:
            print(f"[build_electrode_lines] {name} missing: {missing} -> dropped")
        electrode_lines.append([name2idx[ch] for ch in line if ch in name2idx])
    return electrode_lines, names


def extract_save_name(path):
    match = re.match(r"sub(\d+)_session([0-9.]+)\.fif$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse epoch filename: {path.name}")
    subj = int(match.group(1))
    session_label = match.group(2)
    session_num = int(float(session_label))
    return subj, session_label, session_num


def response_to_number(series):
    values = []
    for item in series:
        match = re.search(r"[12]", str(item))
        values.append(float(match.group(0)) if match else np.nan)
    return np.asarray(values, dtype=float)


def timing_issue_mask(intervals):
    if "has_timing_issue" in intervals.columns:
        return intervals["has_timing_issue"].astype(bool).to_numpy()

    issue = np.zeros(len(intervals), dtype=bool)
    for col, expected in EXPECTED_TIMES.items():
        actual = pd.to_numeric(intervals[col], errors="coerce").to_numpy(dtype=float)
        issue |= np.isnan(actual) | (np.abs(actual - expected) > TOL_S)
    return issue


def align_epochs_to_intervals(intervals, n_epochs):
    """Map every saved epoch to its behavior/intervals row.

    All saved epochs are kept (nothing is dropped here); eeg_bad and
    timing-issue status are returned per epoch so callers can flag/filter
    downstream instead of losing those trials from the FFT computation.
    """
    timing_bad = timing_issue_mask(intervals)
    eeg_bad = (
        intervals["is_bad_epoch"].to_numpy(dtype=bool)
        if "is_bad_epoch" in intervals.columns
        else np.zeros(len(intervals), dtype=bool)
    )

    if n_epochs == len(intervals):
        interval_rows = np.arange(len(intervals), dtype=int)
        return interval_rows, eeg_bad, timing_bad, "all_trials"

    # Some upstream pipelines already drop eeg_bad epochs before saving the FIF.
    # Those trials are physically gone from the file, so they can't be recovered
    # here; every epoch that *is* present is therefore not eeg_bad by construction.
    clean_interval_rows = np.where(~eeg_bad)[0]
    if n_epochs == len(clean_interval_rows):
        return (
            clean_interval_rows,
            np.zeros(n_epochs, dtype=bool),
            timing_bad[clean_interval_rows],
            "bad_epochs_removed",
        )

    raise RuntimeError(
        f"saved epochs={n_epochs}, intervals={len(intervals)}, "
        f"non-bad interval rows={len(clean_interval_rows)}. Cannot align epochs to behavior."
    )


def load_session_epochs(epoch_path):
    """Load one session's epochs/behavior, keeping every trial in the FIF."""
    subj, session_label, session_num = extract_save_name(epoch_path)
    key = f"sub{subj:02d}_session{session_label}"

    csv_path = os.path.join(
        BEHAVIOR_CSV_CACHE,
        f"sub{subj:02d}_session{session_num}_behavior.csv",
    )
    intervals_path = epoch_path.with_name(f"{key}_intervals_autoreject.csv")
    beh = pd.read_csv(csv_path)
    intervals = pd.read_csv(intervals_path)

    epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
    epochs.pick("eeg")
    interval_rows, is_bad_epoch, has_timing_issue, alignment = align_epochs_to_intervals(
        intervals, len(epochs)
    )

    trial_index = intervals.iloc[interval_rows]["trial_index"].to_numpy(dtype=int)

    epochs.crop(tmin=-1.5, tmax=7.5)
    eeg_data = epochs.get_data(copy=True)
    the_data = eeg_data.transpose(1, 2, 0)  # (n_ch, n_times, n_trials)

    beh_keep = beh.iloc[trial_index - 1].reset_index(drop=True)

    return dict(
        subject=subj,
        session=session_label,
        alignment=alignment,
        ch_names=epochs.ch_names,
        sfreq=float(epochs.info["sfreq"]),
        the_data=the_data,
        trial_index=trial_index,
        is_bad_epoch=is_bad_epoch,
        has_timing_issue=has_timing_issue,
        blocktype=np.asarray(beh_keep["blockInfo"], dtype=str),
        test_early_item=np.asarray(beh_keep["testEarlyItem"], dtype=int),
        early_acc=np.asarray(beh_keep["probe1Acc"], dtype=float),
        late_acc=np.asarray(beh_keep["probe2Acc"], dtype=float),
        early_rt=np.asarray(beh_keep["probe1Rt"], dtype=float),
        late_rt=np.asarray(beh_keep["probe2Rt"], dtype=float),
        # Response keys: 1 = left, 2 = right.
        early_response=response_to_number(beh_keep["probe1Response"]),
        late_response=response_to_number(beh_keep["probe2Response"]),
    )


def run_subject_group(files, window_size=250, step=25, shuffle_reps=200):
    """Concatenate a group of a subject's sessions into one batched TW/FFT call.

    Combining sessions before calling wave_trigger_computer_uniform lets one
    call process multiple sessions' trials at once (instead of one ~400-trial
    call per session), which is where the batching benefit comes from;
    results are then split back per session for saving so downstream file
    format stays unchanged. `files` need not be all of a subject's sessions -
    see FILES_PER_GROUP below, which caps how many sessions (and therefore
    how much memory) go into one such call.
    """
    sessions = [load_session_epochs(f) for f in files]

    ch_names = sessions[0]["ch_names"]
    sfreq = sessions[0]["sfreq"]
    for s in sessions[1:]:
        if s["ch_names"] != ch_names:
            raise RuntimeError(
                f"subj{s['subject']:02d}: channel layout differs across sessions; cannot batch."
            )

    electrode_lines, _ = build_electrode_lines(ch_names)
    combined_data = np.concatenate([s["the_data"] for s in sessions], axis=2)

    fwmax, bwmax, fwssmax, bwssmax, fftf, time, ff = wave_trigger_computer_uniform(
        combined_data,
        electrode_lines,
        sfreq=sfreq,
        window_size=window_size,
        step=step,
        shuffle_reps=shuffle_reps,
        baseline_mode="surr",
        max_batch=1600,
        spatial_demean=False,
    )

    results = []
    start = 0
    for s in sessions:
        n = s["the_data"].shape[2]
        sl = slice(start, start + n)
        results.append(dict(
            fwmax=fwmax[..., sl],
            fwssmax=fwssmax[..., sl],
            bwmax=bwmax[..., sl],
            bwssmax=bwssmax[..., sl],
            fftf=None if fftf is None else fftf[..., sl],
            time=time,
            ff=ff,
            subject=s["subject"],
            session=s["session"],
            epoch_alignment=s["alignment"],
            trial_index=s["trial_index"],
            is_bad_epoch=s["is_bad_epoch"],
            has_timing_issue=s["has_timing_issue"],
            blocktype=s["blocktype"],
            test_early_item=s["test_early_item"],
            early_acc=s["early_acc"],
            late_acc=s["late_acc"],
            early_rt=s["early_rt"],
            late_rt=s["late_rt"],
            early_response=s["early_response"],
            late_response=s["late_response"],
        ))
        start += n
    return results


os.makedirs(CACHE_DIR, exist_ok=True)
mne.set_log_level("WARNING")
epoch_files = sorted(
    EPOCH_CLEAN_DIR.glob("sub*_session*.fif"),
    key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
)

subject_groups = defaultdict(list)
for f in epoch_files:
    subj, _session_label, _session_num = extract_save_name(f)
    subject_groups[subj].append(f)

# Each parallel job concatenates at most this many sessions (see
# run_subject_group). Smaller groups -> less memory per job -> more jobs can
# run at once; keep this in sync with the n_jobs passed to Parallel below.
FILES_PER_GROUP = 1
file_groups = [
    files[i:i + FILES_PER_GROUP]
    for files in subject_groups.values()
    for i in range(0, len(files), FILES_PER_GROUP)
]


def out_path_for(subj, session_label):
    return os.path.join(CACHE_DIR, f"subj{subj:02d}_sess{session_label}.pkl")


def run_subject(files):
    subj = extract_save_name(files[0])[0]
    out_paths = [out_path_for(*extract_save_name(f)[:2]) for f in files]

    if all(os.path.exists(p) for p in out_paths):
        print(f"[skip] subj{subj:02d} ({len(files)} sessions)")
        return subj, out_paths

    results = run_subject_group(files)
    for path, result in zip(out_paths, results):
        with open(path, "wb") as fp:
            pickle.dump(result, fp)

    n_trials_total = sum(r["trial_index"].shape[0] for r in results)
    print(f"[done] subj{subj:02d} ({len(files)} sessions, {n_trials_total} trials batched)")
    return subj, out_paths


file_index = Parallel(n_jobs=15, verbose=10)(
    delayed(run_subject)(files) for files in file_groups
)
