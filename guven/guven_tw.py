#!/usr/bin/env python3
"""Compute per-session traveling-wave power for the Guven mental rotation/WM dataset.

Electrode lines are copied from fft/micheal/micheal_tw.py's build_electrode_lines
(not imported -- that file runs a Parallel(...) job as a side effect of module
import, so importing it would kick off micheal's own analysis). guven's montage
is missing CP1/CP2, so lines L1/L2/R1/R2 silently drop to 6 electrodes instead
of 7 (unequal spacing for those 4/11 lines) -- left as-is per instruction, to be
hand-filtered later.

Cue side (ResultMatrix column index 6, 0-indexed -- confirmed against
trialType's documented cue_cond column, "1:left 2:right") is loaded per trial
from each session's behavioral .mat file and saved alongside the wave output
as cue_loc, but is NOT used to reorder electrode lines here -- every trial is
run through the same absolute line order (L5..L1, M, R1..R5), matching
fft/micheal/micheal_tw.py's approach. Cue-side flipping (to express line
position relative to the cued hemisphere instead of absolute left/right) is
done at plot time instead, in guven_plot_funcs.py.

Bad epochs (epochs.metadata["is_bad_epoch"], set by fft/guven/preprocess.py's
peak-to-peak threshold check) are NOT dropped here -- every trial in a session
is run through the wave computation, and is_bad_epoch is carried through to
the output so bad trials can be filtered at plotting time instead.

Each subject's 4 sessions are joblib-parallelized as 2 pairs (session1+2,
session3+4) rather than 4 separate jobs: compute_session_pair pools both
sessions' trials into one wave_trigger_computer_uniform call (halving the
number of joblib tasks and their per-task import/load overhead), then splits
the result back along the trial axis and writes one .pkl per session as
before -- the on-disk cache layout and guven_plot_funcs.load_results() are
unaffected.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat

sys.path.insert(0, "/home/dilay/project2/tw/travelling_waves/tw/tw_fft/func")
from wave_ana_tw_power2 import wave_trigger_computer_uniform  # noqa: E402

RAW_DIR = Path("/media/dilay/T7/guven_data/Raw_Data")
EPOCH_DIR = Path("/home/dilay/project2/tw/Data/guven_data/epoch_autoreject")
CACHE_DIR = Path("/home/dilay/project2/tw/results/guven_fft")

SFREQ = 500
WINDOW_SIZE = 250
STEP = 25
SHUFFLE_REPS = 1

CUE_COL = 6  # ResultMatrix column index (0-indexed); 1=left, 2=right
CUE_LEFT = 1
CUE_RIGHT = 2
RT_COL = 1  # ResultMatrix column index (0-indexed); reaction time


def build_electrode_lines(ch_names: list[str]) -> tuple[list[list[int]], list[str]]:
    M = ['Oz', 'POz', 'Pz', 'CPz', 'Cz', 'FCz', 'Fz']
    L1 = ['O1', 'PO3', 'P1', 'C1', 'FCz', 'Fz']
    L2 = ['O1', 'PO3', 'P1', 'C1', 'FC1', 'F1']
    L3 = ['PO7', 'P5', 'CP3', 'C3','FC1', 'F1']
    L4 = ['O1', 'PO3', 'P3', 'CP3', 'C3', 'FC3', 'F3']
    L5 = ['PO7', 'P5', 'CP3', 'C3', 'FC3', 'F3']
    R1 = ['O2', 'PO4', 'P2',  'C2', 'FCz', 'Fz']
    R2 = ['O2', 'PO4', 'P2',  'C2', 'FC2', 'F2']
    R3 = ['PO8', 'P6', 'CP4', 'C4', 'FC2', 'F2']
    R4 = ['O2', 'PO4', 'P4', 'CP4', 'C4', 'FC4', 'F4']
    R5 = ['PO8', 'P6', 'CP4', 'C4', 'FC4', 'F4']

    lines = [L5, L4, L3, L2, L1, M, R1, R2, R3, R4, R5]
    names = ['L5', 'L4', 'L3', 'L2', 'L1', 'M', 'R1', 'R2', 'R3', 'R4', 'R5']

    name2idx = {c: i for i, c in enumerate(ch_names)}
    electrode_lines = []
    for nm, line in zip(names, lines):
        missing = [c for c in line if c not in name2idx]
        if missing:
            print(f"[build_electrode_lines] {nm} missing: {missing} -> dropped")
        idxs = [name2idx[c] for c in line if c in name2idx]
        electrode_lines.append(idxs)
    return electrode_lines, names


def load_behavior(subj: int, sess: int) -> tuple[np.ndarray, np.ndarray]:
    mat_path = (
        RAW_DIR / f"Ppt_{subj}" / "Behavioural" / "Experiment"
        / f"Results_Gen_2_Mental_Rotation_Experiment_EEG_{subj}_{sess}_1.mat"
    )
    d = loadmat(mat_path, simplify_cells=True)
    result_matrix = d["ResultMatrix"]
    cue_loc = np.asarray(result_matrix[:, CUE_COL]).astype(int)
    assert set(np.unique(cue_loc)) <= {CUE_LEFT, CUE_RIGHT}, (
        f"unexpected cue_loc values in {mat_path.name}: {np.unique(cue_loc)}"
    )
    rt = np.asarray(result_matrix[:, RT_COL]).astype(float)
    return cue_loc, rt


def _parse_subj_sess(fif_path: Path) -> tuple[int, int]:
    match = re.search(r"sub(\d+)_session(\d+)$", fif_path.stem)
    return int(match.group(1)), int(match.group(2))


def _load_session_arrays(fif_path: Path) -> dict:
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    ch_names = epochs.copy().pick("eeg").ch_names
    the_data = epochs.get_data(picks=ch_names).transpose(1, 2, 0)  # (n_ch, n_times, n_trials)
    n_trials = the_data.shape[2]

    subj, sess = _parse_subj_sess(fif_path)
    cue_loc, rt = load_behavior(subj, sess)
    assert len(cue_loc) == n_trials, (
        f"{fif_path.name}: {n_trials} epochs but {len(cue_loc)} behavioral trials"
    )
    is_bad_epoch = epochs.metadata["is_bad_epoch"].to_numpy()

    return dict(
        subj=subj, sess=sess, ch_names=ch_names, the_data=the_data,
        epoch_tmin=float(epochs.tmin), cue_loc=cue_loc, rt=rt, is_bad_epoch=is_bad_epoch,
    )


def _pack_result(result: dict, sfreq: int, epoch_tmin: float, cue_loc, rt, is_bad_epoch, line_names) -> dict:
    return dict(
        fwMax=result["fwMax"], bwMax=result["bwMax"],
        fwMaxSS=result["fwMaxSS"], bwMaxSS=result["bwMaxSS"],
        starts=result["starts"], ff=result["ff"],
        sfreq=sfreq, epoch_tmin=epoch_tmin,
        cue_loc=cue_loc, rt=rt, is_bad_epoch=is_bad_epoch,
        line_names=line_names,
    )


def compute_one_session(fif_path: Path) -> dict:
    s = _load_session_arrays(fif_path)
    lines, line_names = build_electrode_lines(s["ch_names"])
    result = wave_trigger_computer_uniform(
        s["the_data"], lines,
        sfreq=SFREQ, window_size=WINDOW_SIZE, step=STEP, shuffle_reps=SHUFFLE_REPS,
        baseline_mode="surr", spatial_demean=False, return_dict=True,
    )
    return _pack_result(result, SFREQ, s["epoch_tmin"], s["cue_loc"], s["rt"], s["is_bad_epoch"], line_names)


def compute_session_pair(fif_path_a: Path, fif_path_b: Path) -> tuple[dict, dict]:
    """Run the wave computation once on both sessions' trials pooled together
    (fewer, bigger joblib tasks instead of one per session), then split the
    result back into one dict per session so the cache still saves as one
    file per session -- downstream loading (guven_plot_funcs.load_results) is
    unchanged.
    """
    a = _load_session_arrays(fif_path_a)
    b = _load_session_arrays(fif_path_b)
    assert a["subj"] == b["subj"], f"{fif_path_a.name} / {fif_path_b.name}: subject mismatch"

    ch_names = a["ch_names"]
    if b["ch_names"] != ch_names:
        common = [c for c in ch_names if c in set(b["ch_names"])]
        print(f"[compute_session_pair] sub{a['subj']:02d}: channel mismatch between "
              f"session{a['sess']} and session{b['sess']}, using {len(common)} common channels")
        ch_names = common
        a_data = np.asarray([a["the_data"][a["ch_names"].index(c)] for c in ch_names])
        b_data = np.asarray([b["the_data"][b["ch_names"].index(c)] for c in ch_names])
    else:
        a_data, b_data = a["the_data"], b["the_data"]

    n_a = a_data.shape[2]
    combined_data = np.concatenate([a_data, b_data], axis=2)
    lines, line_names = build_electrode_lines(ch_names)

    result = wave_trigger_computer_uniform(
        combined_data, lines,
        sfreq=SFREQ, window_size=WINDOW_SIZE, step=STEP, shuffle_reps=SHUFFLE_REPS,
        baseline_mode="surr", spatial_demean=False, return_dict=True,
    )

    split_a, split_b = {}, {}
    for key in ("fwMax", "bwMax", "fwMaxSS", "bwMaxSS"):
        split_a[key] = result[key][..., :n_a]
        split_b[key] = result[key][..., n_a:]
    shared = dict(starts=result["starts"], ff=result["ff"])
    split_a.update(shared)
    split_b.update(shared)

    dict_a = _pack_result(split_a, SFREQ, a["epoch_tmin"], a["cue_loc"], a["rt"], a["is_bad_epoch"], line_names)
    dict_b = _pack_result(split_b, SFREQ, b["epoch_tmin"], b["cue_loc"], b["rt"], b["is_bad_epoch"], line_names)
    return dict_a, dict_b


def run_one_file(fif_path: Path) -> Path:
    out_path = CACHE_DIR / f"{fif_path.stem}.pkl"
    if out_path.exists():
        print(f"[skip] {fif_path.name}")
        return out_path

    print(f"[start] {fif_path.name}")
    result = compute_one_session(fif_path)
    with open(out_path, "wb") as fp:
        pickle.dump(result, fp)
    print(f"[done] {fif_path.name}")
    return out_path


def run_session_pair(fif_path_a: Path, fif_path_b: Path) -> tuple[Path, Path]:
    out_a = CACHE_DIR / f"{fif_path_a.stem}.pkl"
    out_b = CACHE_DIR / f"{fif_path_b.stem}.pkl"
    if out_a.exists() and out_b.exists():
        print(f"[skip] {fif_path_a.name} + {fif_path_b.name}")
        return out_a, out_b

    print(f"[start] {fif_path_a.name} + {fif_path_b.name}")
    dict_a, dict_b = compute_session_pair(fif_path_a, fif_path_b)
    with open(out_a, "wb") as fp:
        pickle.dump(dict_a, fp)
    with open(out_b, "wb") as fp:
        pickle.dump(dict_b, fp)
    print(f"[done] {fif_path_a.name} + {fif_path_b.name}")
    return out_a, out_b


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fif_files = sorted(EPOCH_DIR.glob("sub*_session*.fif"))

    by_subj: dict[int, list[Path]] = {}
    for f in fif_files:
        subj, _ = _parse_subj_sess(f)
        by_subj.setdefault(subj, []).append(f)

    tasks = []
    for subj in sorted(by_subj):
        files = sorted(by_subj[subj], key=lambda p: _parse_subj_sess(p)[1])
        for i in range(0, len(files) - 1, 2):
            tasks.append(delayed(run_session_pair)(files[i], files[i + 1]))
        if len(files) % 2 == 1:
            tasks.append(delayed(run_one_file)(files[-1]))

    Parallel(n_jobs=18, verbose=10)(tasks)


if __name__ == "__main__":
    main()
