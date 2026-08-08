# %%
import argparse
import glob
import os
import pickle
import re
import sys
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat


FUNC_DIR = "/home/dilay/project2/tw/travelling_waves/tw/tw_fft/func"
if FUNC_DIR not in sys.path:
    sys.path.insert(0, FUNC_DIR)

from wave_ana_newbl import (  # noqa: E402
    build_chn_dist_lines,
    wave_trigger_computer_nonuniform_newbl,
    wave_trigger_computer_uniform_newbl,
)


def format_duration(seconds):
    seconds = float(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


def peel_to_str(x):
    while isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x.ravel()[0]
    return str(x)


def build_electrode_lines(ch_names):
    midline = ["Oz", "POz", "Pz", "CPz", "Cz", "FCz", "Fz"]

    l1 = ["O1", "PO3", "P1", "CP1", "C1", "FCz", "Fz"]
    l2 = ["O1", "PO3", "P1", "CP1", "C1", "FC1", "F1"]
    l3 = ["PO7", "P5", "CP3", "C3", "FC1", "F1"]
    l4 = ["O1", "PO3", "P3", "CP3", "C3", "FC3", "F3"]
    l5 = ["PO7", "P5", "CP3", "C3", "FC3", "F3"]

    r1 = ["O2", "PO4", "P2", "CP2", "C2", "FCz", "Fz"]
    r2 = ["O2", "PO4", "P2", "CP2", "C2", "FC2", "F2"]
    r3 = ["PO8", "P6", "CP4", "C4", "FC2", "F2"]
    r4 = ["O2", "PO4", "P4", "CP4", "C4", "FC4", "F4"]
    r5 = ["PO8", "P6", "CP4", "C4", "FC4", "F4"]

    lines = [l5, l4, l3, l2, l1, midline, r1, r2, r3, r4, r5]
    names = ["L5", "L4", "L3", "L2", "L1", "M", "R1", "R2", "R3", "R4", "R5"]
    name2idx = {ch: i for i, ch in enumerate(ch_names)}

    electrode_lines = []
    for name, line in zip(names, lines):
        missing = [ch for ch in line if ch not in name2idx]
        if missing:
            print(f"[build_electrode_lines] {name} missing: {missing} -> dropped")
        electrode_lines.append([name2idx[ch] for ch in line if ch in name2idx])
    return electrode_lines, names


def load_and_compute_one(
    mat_path,
    *,
    method="fft",
    sfreq=500,
    window_size=250,
    step=25,
    shuffle_reps=10,
    freq_lo=2.0,
    freq_hi=40.0,
    demean=False,
    symmetry_flag=True,
    max_batch=1000,
    seed=None,
):
    eeg = loadmat(mat_path, struct_as_record=False)
    data = eeg["ft_mem"].item()

    t = np.asarray(data.time).ravel()
    tmin, tmax = -1.25, 6.0
    mask = (t >= tmin) & (t <= tmax)
    eeg_crop = data.trial[:, :, mask]  # (n_trials, n_ch, n_times_crop)
    eeg_data = eeg_crop

    channel_info = data.elec.item()
    ch_names = [peel_to_str(x) for x in np.asarray(channel_info.label).ravel()]
    elecpos = np.asarray(channel_info.elecpos)

    bad_trials = np.unique(
        np.concatenate(
            [
                data.bad_trials_mem.reshape(-1),
                data.bad_trials_imp1.reshape(-1),
                data.bad_trials_imp2.reshape(-1),
                data.bad_trials_probe1.reshape(-1),
                data.bad_trials_probe2.reshape(-1),
            ]
        )
    )
    eeg_data = np.delete(eeg_data, bad_trials - 1, axis=0)

    early_rot = np.where(np.asarray(data.Results[:, 8]).ravel() > 0, 2, 1)
    late_rot = np.where(np.asarray(data.Results[:, 12]).ravel() > 0, 2, 1)
    early_rot = np.delete(early_rot, bad_trials - 1)
    late_rot = np.delete(late_rot, bad_trials - 1)

    early_acc = np.asarray(data.Results[:, 10]).ravel()
    early_rt = np.asarray(data.Results[:, 9]).ravel()
    late_acc = np.asarray(data.Results[:, 14]).ravel()
    late_rt = np.asarray(data.Results[:, 13]).ravel()
    early_acc = np.delete(early_acc, bad_trials - 1)
    early_rt = np.delete(early_rt, bad_trials - 1)
    late_acc = np.delete(late_acc, bad_trials - 1)
    late_rt = np.delete(late_rt, bad_trials - 1)

    cue_loc = np.asarray(data.Results[:, 2]).ravel()
    cue_loc = np.delete(cue_loc, bad_trials - 1)

    electrode_lines, line_names = build_electrode_lines(ch_names)
    chn_dist_lines = build_chn_dist_lines(electrode_lines, elecpos)

    the_data = eeg_data.transpose(1, 2, 0)  # (n_ch, n_times, n_trials)
    rng = None if seed is None else np.random.default_rng(seed)

    if method == "fft":
        tw = wave_trigger_computer_uniform_newbl(
            the_data,
            electrode_lines,
            sfreq=sfreq,
            window_size=window_size,
            step=step,
            shuffle_reps=shuffle_reps,
            freq_lo=freq_lo,
            freq_hi=freq_hi,
            demean=demean,
            symmetry_flag=symmetry_flag,
            max_batch=max_batch,
            rng=rng,
        )
    elif method == "nfft":
        tw = wave_trigger_computer_nonuniform_newbl(
            the_data,
            electrode_lines,
            chn_dist_lines,
            sfreq=sfreq,
            window_size=window_size,
            step=step,
            shuffle_reps=shuffle_reps,
            freq_lo=freq_lo,
            freq_hi=freq_hi,
            demean=demean,
            symmetry_flag=symmetry_flag,
            max_batch=max_batch,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return dict(
        method=method,
        fwmax=tw["fwmax"],
        bwmax=tw["bwmax"],
        ratio=tw["ratio"],
        pfw=tw["pfw"],
        pbw=tw["pbw"],
        ratio_real_dist=tw["ratio_real_dist"],
        ratio_null_dist=tw["ratio_null_dist"],
        ratio_xq=tw["ratio_xq"],
        time=tw["starts"],
        starts=tw["starts"],
        ff=tw["ff"],
        fy=tw["fy"],
        demean=demean,
        shuffle_reps=tw["shuffle_reps"],
        max_batch=tw["max_batch"],
        symmetry_flag=tw["symmetry_flag"],
        electrode_lines=electrode_lines,
        line_names=line_names,
        cue_loc=cue_loc,
        early_rot=early_rot,
        late_rot=late_rot,
        early_acc=early_acc,
        early_rt=early_rt,
        late_acc=late_acc,
        late_rt=late_rt,
    )


def run_one_file(f, out_dir, args):
    file_start = time.perf_counter()
    base = os.path.basename(f)
    sess = int(re.search(r"sess(\d+)", base).group(1))
    subj = int(re.search(r"_(\d+)\.mat$", base).group(1))
    out_path = os.path.join(out_dir, f"subj{subj:02d}_sess{sess}.pkl")

    if os.path.exists(out_path) and not args.overwrite:
        elapsed = format_duration(time.perf_counter() - file_start)
        print(f"[skip] {base} elapsed={elapsed}", flush=True)
        return subj, sess, out_path

    print(
        f"[start] {base} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    seed = None if args.seed is None else args.seed + subj * 100 + sess
    result = load_and_compute_one(
        f,
        method=args.method,
        sfreq=args.sfreq,
        window_size=args.window_size,
        step=args.step,
        shuffle_reps=args.shuffle_reps,
        freq_lo=args.freq_lo,
        freq_hi=args.freq_hi,
        demean=args.demean,
        symmetry_flag=not args.no_symmetry,
        max_batch=args.max_batch,
        seed=seed,
    )

    with open(out_path, "wb") as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = format_duration(time.perf_counter() - file_start)
    print(f"[done] {base} elapsed={elapsed}", flush=True)
    return subj, sess, out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute travelling-wave FW/BW probabilities with log-ratio null distributions."
    )
    parser.add_argument("--root", default="/home/dilay/project2/tw/Data_whole")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/micheal_newbl_<method>.",
    )
    parser.add_argument(
        "--method",
        choices=("fft", "nfft"),
        default="nfft",
        help="Use uniform FFT or nonuniform NUFFT/NFFT analysis.",
    )
    parser.add_argument("--sfreq", type=float, default=500)
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--shuffle-reps", type=int, default=10)
    parser.add_argument("--max-batch", type=int, default=1000)
    parser.add_argument("--freq-lo", type=float, default=2.0)
    parser.add_argument("--freq-hi", type=float, default=40.0)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--demean", action="store_true", default=True)
    parser.add_argument("--no-symmetry", action="store_true")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f"/home/dilay/project2/tw/results/micheal_newbl_{args.method}"
    return args


def main():
    total_start = time.perf_counter()
    wall_start = datetime.now()
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(args.root, "MemImp3_mem_whole_sess*_*.mat")))
    if args.limit_files is not None:
        mat_files = mat_files[: args.limit_files]

    print(
        (
            f"[total-start] {wall_start.strftime('%Y-%m-%d %H:%M:%S')} "
            f"files={len(mat_files)} n_jobs={args.n_jobs} method={args.method}"
        ),
        flush=True,
    )
    file_index = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one_file)(f, args.out_dir, args) for f in mat_files
    )

    index_path = os.path.join(args.out_dir, "file_index.pkl")
    with open(index_path, "wb") as fp:
        pickle.dump(file_index, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[index] {index_path}")
    total_elapsed = format_duration(time.perf_counter() - total_start)
    print(
        (
            f"[total-done] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"elapsed={total_elapsed}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
