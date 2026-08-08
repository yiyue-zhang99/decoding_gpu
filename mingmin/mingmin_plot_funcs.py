#!/usr/bin/env python3
"""Plot Mingmin sequence travelling-wave results.

The plotting style follows micheal_plot.ipynb, but this sequence dataset has
no cue location. Everything is therefore aligned by response hand only:
left-hand response trials are flipped so line indices 0..4 are contralateral
and 6..10 are ipsilateral. The main plotted quantity is contra - ipsi.
"""


import pickle
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mne.stats import permutation_cluster_1samp_test
import warnings

warnings.filterwarnings("ignore", message="No clusters found", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Provided stat_fun", category=RuntimeWarning)


RESULT_DIR = Path("/home/dilay/project2/tw/results/mingmin_fft")
FIG_DIR = RESULT_DIR / "figures"

ALPHA = 0.05
BL0, BL1 = -0.75, -0.25
BL_CORRECT = False
TMIN_BASE = -1.2
SFREQ = 500
WINDOW_SIZE = 250
MAX_RT = 1.55  

LINE_NAMES = ["L5", "L4", "L3", "L2", "L1", "M", "R1", "R2", "R3", "R4", "R5"]
FLIPPED_LINE_NAMES = ["C5", "C4", "C3", "C2", "C1", "M", "I1", "I2", "I3", "I4", "I5"]
LINE_ALIASES = {
    **{name.upper(): idx for idx, name in enumerate(LINE_NAMES)},
    **{name.upper(): idx for idx, name in enumerate(FLIPPED_LINE_NAMES)},
    "MID": 5,
    "MIDLINE": 5,
}

FREQ_BANDS = [
    (2, 6, "Theta (2-6 Hz)"),
    (8, 12, "Alpha (8-12 Hz)"),
    (14, 30, "Beta (14-30 Hz)"),
]
MEASURES = ["fw", "bw", "ratio"]
MEASURE_COLORS = {
    "fw": "#1f77b4",
    "bw": "#d55e00",
    "ratio": "#9467bd",
}
MEASURE_LABELS = {
    "fw": "FW",
    "bw": "BW",
    "ratio": "RATIO",
}
MEASURE_YLABELS = {
    "fw": "FW contra-ipsi (dB)",
    "bw": "BW contra-ipsi (dB)",
    "ratio": "log-ratio",
}


EVENT_TIMES = [0, 1.15, 2.30, 2.90, 4.45, 5.05]
EVENT_LABELS = ["stim1", "stim2", "impulse1", "probe1", "impulse2", "probe2"]
PROBE_ONSET = {"early": 2.90, "late": 5.05}
BAR_WIDTH_LONG = 0.3
BAR_WIDTH_SHORT = 0.15
STIM_BARS = [
    (0, BAR_WIDTH_LONG), (1.15, BAR_WIDTH_LONG), (2.30, BAR_WIDTH_SHORT),
    (2.90, BAR_WIDTH_LONG), (4.45, BAR_WIDTH_SHORT), (5.05, BAR_WIDTH_LONG),
]
DEFAULT_YLIMS = {"fw": (-0.1, 0.1), "bw": (-0.1, 0.1), "ratio": (-0.02, 0.02)}
LINE_POWER_YLIMS = {"fw": (-0.4, 0.4), "bw": (-0.4, 0.4), "ratio": (-0.02, 0.02)}
FIXED_RANDOM_YLIMS = {"fw": (-0.4, 0.4), "bw": (-0.4, 0.4), "ratio": (-0.1, 0.1)}

# ----------------------------- load data -----------------------------
def load_results(result_dir: Path):
    by_subj = defaultdict(dict)
    for path in sorted(result_dir.glob("subj*_sess*.pkl")):
        match = re.search(r"subj(\d+)_sess([0-9.]+)", path.name)
        if match is None:
            continue
        subj = int(match.group(1))
        sess = match.group(2)
        with path.open("rb") as f:
            by_subj[subj][sess] = pickle.load(f)
    return by_subj, sorted(by_subj)

def resolve_line_idx(line_idx, default):
    """Resolve numeric indices or line labels to 0-based line indices."""
    if line_idx is None:
        return list(default)
    if isinstance(line_idx, (str, int, np.integer)):
        line_idx = [line_idx]

    resolved = []
    for item in line_idx:
        if isinstance(item, (int, np.integer)):
            resolved.append(int(item))
            continue
        key = str(item).strip().upper()
        if key not in LINE_ALIASES:
            valid = ", ".join(FLIPPED_LINE_NAMES + LINE_NAMES)
            raise ValueError(f"Unknown line index/name {item!r}. Valid names: {valid}")
        resolved.append(LINE_ALIASES[key])
    return resolved

# ── Trial filters ──────────────────────────────────────────────────────────────
def _respond_left_mask(d, rt_type):
    if rt_type == "early":
        return d["early_response"] == 1
    if rt_type == "late":
        return d["late_response"] == 1
    raise ValueError(f"rt_type should be 'early' or 'late', got {rt_type!r}")


def _flip_by_response_hand(arr, d, rt_type):
    """Flip left-response trials so every trial uses a right-hemi reference."""
    arr = arr.copy()
    left = _respond_left_mask(d, rt_type)
    arr[:, :, :, left] = arr[::-1, :, :, left]
    return arr


def _block_mask(d, trial_filter):
    """Resolve no block filter (None/'all'), 'fixed', or 'random'."""
    if trial_filter is None or trial_filter == 'all':
        return np.ones(len(d["blocktype"]), dtype=bool)
    block = np.char.lower(np.asarray(d["blocktype"]).astype(str))
    return block == trial_filter


def _rt_valid_mask(d, max_rt=None):
    if max_rt is None:
        max_rt = MAX_RT
    return np.isfinite(d["early_rt"]) & (d["late_rt"] <= max_rt)


def _no_timing_issue_mask(d):
    n = len(d["blocktype"])
    if "has_timing_issue" not in d:
        return np.ones(n, dtype=bool)
    return ~np.asarray(d["has_timing_issue"], dtype=bool)


def _no_bad_epoch_mask(d):
    n = len(d["blocktype"])
    if "is_bad_epoch" not in d:
        return np.ones(n, dtype=bool)
    return ~np.asarray(d["is_bad_epoch"], dtype=bool)


def _resolve_trial_mask(
    d,
    trial_filter="all",
    exclude_invalid_rt=False,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Build the boolean mask used by plotting functions to select trials."""
    mask = np.ones(len(d['early_rt']), dtype=bool)
    if exclude_invalid_rt:
        mask &= _rt_valid_mask(d, max_rt=max_rt)
    mask &= _block_mask(d, trial_filter)
    if exclude_timing_issue:
        mask &= _no_timing_issue_mask(d)
    if exclude_bad_epoch:
        mask &= _no_bad_epoch_mask(d)
    return mask

# ----trial summary printing  ----

def print_trial_summary(by_subj, subj_ids, rt_type="early", max_rt=None):
    """One row per subject: raw/removed/eeg_bad/timing_bad/rt_bad counts, the
    combined (union) exclusion total, and fixed/random kept-vs-total trial counts.

    eeg_bad and timing_bad are read from each trial's own is_bad_epoch /
    has_timing_issue flags (already saved in the pkl), so no separate
    intervals.csv lookup is needed. rt_bad uses _rt_valid_mask for rt_type/max_rt.
    """
    if max_rt is None:
        max_rt = MAX_RT
    print(f"Subject trial counts after EEG bad + timing bad + RT ({rt_type}, max_rt={max_rt}) exclusions")
    print(
        "subj\torig\tkept\tremoved\tremoved%\t"
        "eeg_bad\teeg_bad%\ttiming_bad\ttiming_bad%\trt_bad\trt_bad%\t"
        "total_excl\ttotal_excl%\tfixed(kept/total)\trandom(kept/total)\t"
        "early_rt_mean\tlate_rt_mean"
    )
    total_excl_pcts = []
    kept_pcts = []
    overall_orig_total = 0
    overall_kept_total = 0
    subj_early_rt_means = []
    subj_late_rt_means = []
    for subj in subj_ids:
        total = 0
        eeg_bad_total = timing_bad_total = rt_bad_total = combined_excl_total = 0
        fixed_total = random_total = fixed_kept = random_kept = 0
        kept_early_rts = []
        kept_late_rts = []

        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            block = np.char.lower(np.asarray(d["blocktype"]).astype(str))
            total += len(block)

            no_timing = _no_timing_issue_mask(d)
            no_bad = _no_bad_epoch_mask(d)
            rt_ok = _rt_valid_mask(d, max_rt=max_rt)
            keep_all = no_timing & no_bad & rt_ok

            eeg_bad_total += int((~no_bad).sum())
            timing_bad_total += int((~no_timing).sum())
            rt_bad_total += int((~rt_ok).sum())
            combined_excl_total += int((~keep_all).sum())

            fixed_mask = block == "fixed"
            random_mask = block == "random"
            fixed_total += int(fixed_mask.sum())
            random_total += int(random_mask.sum())
            fixed_kept += int((fixed_mask & keep_all).sum())
            random_kept += int((random_mask & keep_all).sum())
            kept_early_rts.append(np.asarray(d["early_rt"])[keep_all])
            kept_late_rts.append(np.asarray(d["late_rt"])[keep_all])

        orig_total = 1680  # 420 trials/session x 4 blocks/subject
        removed = orig_total - total
        removed_pct = 0.0 if orig_total == 0 else removed / orig_total * 100
        eeg_bad_pct = 0.0 if orig_total == 0 else eeg_bad_total / orig_total * 100
        timing_bad_pct = 0.0 if orig_total == 0 else timing_bad_total / orig_total * 100
        rt_bad_pct = 0.0 if orig_total == 0 else rt_bad_total / orig_total * 100

        # total_excl = trials never saved to the pkl + trials present but flagged
        # by any of eeg_bad / timing_bad / rt_invalid (a true union, not a sum).
        total_excl = removed + combined_excl_total
        total_excl_pct = 0.0 if orig_total == 0 else total_excl / orig_total * 100
        kept_total = orig_total - total_excl
        kept_pct = 0.0 if orig_total == 0 else kept_total / orig_total * 100
        total_excl_pcts.append(total_excl_pct)
        kept_pcts.append(kept_pct)
        overall_orig_total += orig_total
        overall_kept_total += kept_total
        early_rt_mean = float(np.nanmean(np.concatenate(kept_early_rts)))
        late_rt_mean = float(np.nanmean(np.concatenate(kept_late_rts)))
        subj_early_rt_means.append(early_rt_mean)
        subj_late_rt_means.append(late_rt_mean)

        print(
            f"{subj}\t{orig_total}\t{total}\t{removed}\t{removed_pct:.2f}%\t"
            f"{eeg_bad_total}\t{eeg_bad_pct:.2f}%\t{timing_bad_total}\t{timing_bad_pct:.2f}%\t"
            f"{rt_bad_total}\t{rt_bad_pct:.2f}%\t{total_excl}\t{total_excl_pct:.2f}%\t"
            f"{fixed_kept}/{fixed_total}\t{random_kept}/{random_total}\t"
            f"{early_rt_mean:.3f}\t{late_rt_mean:.3f}"
        )

    if total_excl_pcts:
        overall_kept_pct = (
            100.0 * overall_kept_total / overall_orig_total
            if overall_orig_total > 0 else 0.0
        )
        print("\nOverall (pooled across subjects):")
        print(f"  total trials = {overall_orig_total}")
        print(f"  kept trials  = {overall_kept_total}  (kept ratio = {overall_kept_pct:.2f}%)")
        print(f"\nGroup average kept% across {len(kept_pcts)} subjects: {np.mean(kept_pcts):.2f}%")
        print(f"Group average early RT (s): {np.mean(subj_early_rt_means):.3f} ± {np.std(subj_early_rt_means, ddof=1):.3f}")
        print(f"Group average late RT (s):  {np.mean(subj_late_rt_means):.3f} ± {np.std(subj_late_rt_means, ddof=1):.3f}")

# ── Statistics helpers ─────────────────────────────────────────────────────────

def _cluster_1d(data, alpha=ALPHA, n_perm=10000):
    """Cluster test for subject x time data. Returns (sig, details)."""
    threshold = stats.t.ppf(1 - alpha / 2, df=data.shape[0] - 1)
    obs, cls, ps, _ = permutation_cluster_1samp_test(
        data[:, :, np.newaxis],
        threshold=threshold, n_permutations=n_perm,
        tail=0, n_jobs=-1, verbose=False, seed=42,
    )
    sig = np.zeros(data.shape[1], dtype=bool)
    details = []
    for i, (cl, p) in enumerate(zip(cls, ps), start=1):
        time_idx = cl[0]
        if p < alpha:
            sig[time_idx] = True
        details.append({
            'cluster_id': i,
            'p': float(p),
            'start_idx': int(time_idx[0]),
            'end_idx': int(time_idx[-1]),
            'significant': bool(p < alpha),
        })
    return sig, details


def _cluster_2d(data, alpha=ALPHA, n_perm=5000):
    """Cluster test for subject x freq x time data."""
    if data.shape[0] < 3:
        return np.zeros(data.shape[1:], dtype=bool), []

    threshold = stats.t.ppf(1 - alpha / 2, df=data.shape[0] - 1)
    obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        data,
        threshold=threshold,
        n_permutations=n_perm,
        tail=0,
        n_jobs=1,
        verbose=False,
        seed=42,
    )

    sig = np.zeros(obs.shape, dtype=bool)
    details = []
    for cluster, p_value in zip(clusters, p_values):
        cluster_mask = np.zeros(obs.shape, dtype=bool)
        cluster_mask[cluster] = True
        freq_idx, time_idx = np.where(cluster_mask)
        if p_value < alpha:
            sig |= cluster_mask
        details.append({
            "p": float(p_value),
            "significant": bool(p_value < alpha),
            "freq_start_idx": int(freq_idx.min()),
            "freq_end_idx": int(freq_idx.max()),
            "time_start_idx": int(time_idx.min()),
            "time_end_idx": int(time_idx.max()),
        })
    return sig, details

def _print_cluster_details(details, t=None, ff=None, only_significant=True):
    shown = False
    for item in details:
        if only_significant and not item["significant"]:
            continue
        if ff is not None and t is not None and "freq_start_idx" in item:
            where = (
                f"{ff[item['freq_start_idx']]:.2f}-{ff[item['freq_end_idx']]:.2f} Hz, "
                f"{t[item['time_start_idx']]:.2f}-{t[item['time_end_idx']]:.2f} s"
            )
        elif "freq_start_idx" in item:
            where = (
                f"frequency index {item['freq_start_idx']}-{item['freq_end_idx']}, "
                f"time index {item['time_start_idx']}-{item['time_end_idx']}"
            )
        elif t is not None and "start_idx" in item:
            where = f"{t[item['start_idx']]:.2f}-{t[item['end_idx']]:.2f}s"
        else:
            where = "location unavailable"
        label = "significant" if item["significant"] else "n.s."
        print(f"{where}, p={item['p']:.4g} ({label})")
        shown = True
    if not shown:
        print("No significant clusters found.")

def _draw_sig_bar(ax, mask, t, y, color="#b00020", lw=5):
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    for start, stop in zip(starts, stops):
        ax.plot(
            [t[start], t[min(stop - 1, len(t) - 1)]],
            [y, y],
            color=color,
            linewidth=lw,
            solid_capstyle="butt",
        )


def _compute_rt_lines(
    by_subj, subj_ids,
    trial_filter=None,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    early_all, late_all = [], []
    for subj in subj_ids:
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            valid = _resolve_trial_mask(
                d, trial_filter=trial_filter,
                exclude_invalid_rt=True, max_rt=max_rt,
                exclude_timing_issue=exclude_timing_issue,
                exclude_bad_epoch=exclude_bad_epoch,
            )
            early_all.append(d['early_rt'][valid])
            late_all.append(d['late_rt'][valid])
    med_early = float(np.median(np.concatenate(early_all)))
    med_late = float(np.median(np.concatenate(late_all)))
    early_line = PROBE_ONSET['early'] + med_early
    late_line = PROBE_ONSET['late'] + med_late
    return early_line, late_line


def _draw_rt_lines(ax, early_line, late_line):
    """Red dashed = median early RT; green dashed = median second (late/probe2) RT."""
    ax.axvline(early_line, color='tab:red', linestyle='--', linewidth=1.4, alpha=0.85)
    ax.axvline(late_line, color='tab:green', linestyle='--', linewidth=1.4, alpha=0.85)


def _mean_ci(y):
    mean = y.mean(axis=0)
    sem = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    ci = stats.t.ppf(0.975, df=y.shape[0] - 1) * sem
    return mean, ci


def _time_axis(d, tmin_base=TMIN_BASE, sfreq=SFREQ, window_size=WINDOW_SIZE):
    starts = np.asarray(d["time"])
    return tmin_base + (starts + window_size // 2) / sfreq



def _measure_arrays(d, 
                    baseline_correct=BL_CORRECT, 
                    bl0=BL0, bl1=BL1):
    fw = 10 * np.log10(d["fwmax"] / d["fwssmax"])
    bw = 10 * np.log10(d["bwmax"] / d["bwssmax"])
    ratio = np.log10(d["fwmax"] / d["bwmax"])

    if baseline_correct:
        t = _time_axis(d)
        bl = (t >= bl0) & (t <= bl1)
        fw -= fw[:, :, bl, :].mean(axis=2, keepdims=True)
        bw -= bw[:, :, bl, :].mean(axis=2, keepdims=True)
        ratio -= ratio[:, :, bl, :].mean(axis=2, keepdims=True)

    return {"fw": fw, "bw": bw, "ratio": ratio}



def _subject_contra_ipsi(
    by_subj,
    subj_ids,
    mode="hand",
    rt_type="early",
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    contra_idx=None,
    ipsi_idx=None,
    mid_idx=None,
    trial_filter=None,
    exclude_invalid_rt=True,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Return subject-level contra, ipsi, and midline FW/BW/ratio maps."""

    contra_idx = resolve_line_idx(contra_idx, range(5))
    ipsi_idx = resolve_line_idx(ipsi_idx, range(6, 11))
    mid_idx = resolve_line_idx(mid_idx, [5])
    if len(contra_idx) != len(ipsi_idx):
        raise ValueError("contra_idx and ipsi_idx must contain the same number of lines.")

    d0 = by_subj[subj_ids[0]][sorted(by_subj[subj_ids[0]])[0]]
    t = _time_axis(d0)
    ff = np.asarray(d0["ff"])

    out = {
        measure: {side: [] for side in ("contra", "ipsi", "mid")}
        for measure in MEASURES
    }
    used_subj = []

    for subj in subj_ids:
        sess_data = {
            measure: {side: [] for side in ("contra", "ipsi", "mid")}
            for measure in MEASURES
        }
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            keep = _resolve_trial_mask(
                d, trial_filter=trial_filter,
                exclude_invalid_rt=exclude_invalid_rt, max_rt=max_rt,
                exclude_timing_issue=exclude_timing_issue,
                exclude_bad_epoch=exclude_bad_epoch,
            )
            if not np.any(keep):
                continue
            arrays = _measure_arrays(d, baseline_correct=baseline_correct, bl0=bl0, bl1=bl1)
            for measure, arr in arrays.items():
                if mode == "hand":
                    arr = _flip_by_response_hand(arr, d, rt_type)
                elif d["cue"][0] == 2:
                    arr = arr[::-1]
                arr = arr[..., keep].mean(axis=-1)
                contra = arr[contra_idx].mean(axis=0)
                ipsi = arr[ipsi_idx].mean(axis=0)
                mid = arr[mid_idx].mean(axis=0)
                sess_data[measure]["contra"].append(contra)
                sess_data[measure]["ipsi"].append(ipsi)
                sess_data[measure]["mid"].append(mid)

        if sess_data["ratio"]["contra"]:
            used_subj.append(subj)
            for measure in MEASURES:
                for side in ("contra", "ipsi", "mid"):
                    out[measure][side].append(np.mean(sess_data[measure][side], axis=0))

    out = {
        measure: {side: np.asarray(values) for side, values in side_data.items()}
        for measure, side_data in out.items()
    }
    out["subjects"] = used_subj
    out["time"] = t
    out["ff"] = ff
    out["contra_idx"] = contra_idx
    out["ipsi_idx"] = ipsi_idx
    out["mid_idx"] = mid_idx
    return out

# --- plot helpers ---
def _decorate_axis(
    ax,
    event_labels=False,
    label_fontsize=9,
    label_rotation=28,
    label_y=-0.10,
    xlim=None,
):
    for x in EVENT_TIMES:
        ax.axvline(x, color="0.2", linestyle="--", linewidth=0.9, dashes=(5, 5), alpha=0.45)
    ax.broken_barh(
        STIM_BARS,
        (-0.05, 0.05),
        facecolors="gray",
        alpha=0.45,
        clip_on=False,
        transform=ax.get_xaxis_transform(),
    )
    if event_labels:
        for x, label in zip(EVENT_TIMES, EVENT_LABELS):
            ax.text(
                x - 0.05,
                label_y,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=label_rotation,
                fontsize=label_fontsize,
                fontweight="bold",
                color="0.1",
                va="top",
                ha="right",
                clip_on=False,
            )
    if xlim is not None:
        ax.set_xlim(*xlim)


def _band_axes(n_bands, row_height=4.2):
    # Match the Guven/Micheal band plots: compare frequency bands on one y scale.
    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, row_height), sharey=True)
    axes = np.asarray(axes).ravel()
    return fig, axes


def _subject_line_power(
    by_subj,
    subj_ids,
    rt_type="early",
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    line_idx=None,
    trial_filter='all',
    exclude_invalid_rt=True,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Return subject-level TW power averaged over electrode lines.

    The returned measure arrays have shape n_subject x freq x time. By default
    all 11 electrode lines are averaged, so this is not a contra-ipsi measure.
    """

    d0 = by_subj[subj_ids[0]][sorted(by_subj[subj_ids[0]])[0]]
    t = _time_axis(d0)
    ff = np.asarray(d0["ff"])

    out = {measure: [] for measure in MEASURES}
    used_subj = []

    for subj in subj_ids:
        sess_data = {measure: [] for measure in MEASURES}
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            keep = _resolve_trial_mask(
                d, trial_filter=trial_filter,
                exclude_invalid_rt=exclude_invalid_rt, max_rt=max_rt,
                exclude_timing_issue=exclude_timing_issue,
                exclude_bad_epoch=exclude_bad_epoch,
            )
            if not np.any(keep):
                continue

            arrays = _measure_arrays(d, baseline_correct=baseline_correct, bl0=bl0, bl1=bl1)
            for measure, arr in arrays.items():
                idx = resolve_line_idx(line_idx, range(arr.shape[0]))
                arr_sel = arr[idx]
                sess_data[measure].append(arr_sel[..., keep].mean(axis=(0, -1)))

        if len(sess_data["ratio"]) > 0:
            used_subj.append(subj)
            for measure in MEASURES:
                out[measure].append(np.mean(sess_data[measure], axis=0))

    out = {measure: np.asarray(values) for measure, values in out.items()}
    out["subjects"] = used_subj
    out["time"] = t
    out["ff"] = ff
    return out


# ── Fast / slow RT analysis ────────────────────────────────────────────────────

def compute_fast_slow(
    by_subj,
    subj_ids,
    mode="hand",
    rt_type="early",
    fmin=8,
    fmax=12,
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    contra_idx=None,
    ipsi_idx=None,
    trial_filter=None,
    exclude_invalid_rt=True,
    max_rt=None,
    freq_bands=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Compute subject-level fast/slow contra-ipsi power time series for sequence data.

    If freq_bands is provided, return arrays shaped
    (n_bands, n_subject, n_time). Otherwise use the single requested fmin/fmax
    band.
    """
    d0 = by_subj[subj_ids[0]][sorted(by_subj[subj_ids[0]])[0]]
    t = _time_axis(d0)
    ff = np.asarray(d0["ff"])
    contra_idx = resolve_line_idx(contra_idx, range(5))
    ipsi_idx = resolve_line_idx(ipsi_idx, range(6, 11))
    if len(contra_idx) != len(ipsi_idx):
        raise ValueError("contra_idx and ipsi_idx must contain the same number of lines.")

    if freq_bands is None:
        if fmin is None or fmax is None:
            freq_bands = [(2, 6), (8, 12), (14, 30)]
        else:
            freq_bands = [(fmin, fmax)]
    def make_band_lists():
        return ([[] for _ in freq_bands], [[] for _ in freq_bands], [[] for _ in freq_bands], [[] for _ in freq_bands], [[] for _ in freq_bands], [[] for _ in freq_bands])

    fw_fast_list, fw_slow_list, bw_fast_list, bw_slow_list, ratio_fast_list, ratio_slow_list = make_band_lists()
    fast_rt_all, slow_rt_all = [], []

    for subj in subj_ids:
        band_trial_data = [[] for _ in freq_bands]
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            keep = _resolve_trial_mask(
                d, trial_filter=trial_filter,
                exclude_invalid_rt=exclude_invalid_rt, max_rt=max_rt,
                exclude_timing_issue=exclude_timing_issue,
                exclude_bad_epoch=exclude_bad_epoch,
            )
            arrays = _measure_arrays(d, baseline_correct=baseline_correct, bl0=bl0, bl1=bl1)
            for measure, arr in arrays.items():
                if mode == "hand":
                    arr = _flip_by_response_hand(arr, d, rt_type)
                elif np.asarray(d["cue"]).reshape(-1)[0] == 2:
                    arr = arr[::-1]
                diff_arr = arr[contra_idx] - arr[ipsi_idx][::-1]
                arrays[measure] = diff_arr.mean(axis=0)

            rt = d[f"{rt_type}_rt"][keep]
            for band_idx, (lo, hi) in enumerate(freq_bands):
                fmask = (ff >= lo) & (ff <= hi)
                fw_arr = arrays["fw"][fmask].mean(axis=0)
                bw_arr = arrays["bw"][fmask].mean(axis=0)
                ratio_arr = arrays["ratio"][fmask].mean(axis=0)
                band_trial_data[band_idx].append((fw_arr[:, keep], bw_arr[:, keep], ratio_arr[:, keep], rt))

        all_rt = [rt for band_entries in band_trial_data for (_, _, _, rt) in band_entries]
        rt_all = np.concatenate(all_rt)
        med_rt = np.median(rt_all)

        for band_idx, band_entries in enumerate(band_trial_data):
            fw_fast_subj = []
            fw_slow_subj = []
            bw_fast_subj = []
            bw_slow_subj = []
            ratio_fast_subj = []
            ratio_slow_subj = []
            fast_rt_subj = []
            slow_rt_subj = []

            for fw_arr, bw_arr, ratio_arr, rt in band_entries:
                fast_mask = rt < med_rt
                slow_mask = rt > med_rt
                if fast_mask.any():
                    fw_fast_subj.append(fw_arr[:, fast_mask])
                    bw_fast_subj.append(bw_arr[:, fast_mask])
                    ratio_fast_subj.append(ratio_arr[:, fast_mask])
                    fast_rt_subj.append(rt[fast_mask])
                if slow_mask.any():
                    fw_slow_subj.append(fw_arr[:, slow_mask])
                    bw_slow_subj.append(bw_arr[:, slow_mask])
                    ratio_slow_subj.append(ratio_arr[:, slow_mask])
                    slow_rt_subj.append(rt[slow_mask])

            fw_fast_list[band_idx].append(np.concatenate(fw_fast_subj, axis=1).mean(axis=1))
            fw_slow_list[band_idx].append(np.concatenate(fw_slow_subj, axis=1).mean(axis=1))
            bw_fast_list[band_idx].append(np.concatenate(bw_fast_subj, axis=1).mean(axis=1))
            bw_slow_list[band_idx].append(np.concatenate(bw_slow_subj, axis=1).mean(axis=1))
            ratio_fast_list[band_idx].append(np.concatenate(ratio_fast_subj, axis=1).mean(axis=1))
            ratio_slow_list[band_idx].append(np.concatenate(ratio_slow_subj, axis=1).mean(axis=1))
            fast_rt_all.extend(np.concatenate(fast_rt_subj))
            slow_rt_all.extend(np.concatenate(slow_rt_subj))

    return (
        np.asarray([np.asarray(values) for values in fw_fast_list]),
        np.asarray([np.asarray(values) for values in fw_slow_list]),
        np.asarray([np.asarray(values) for values in bw_fast_list]),
        np.asarray([np.asarray(values) for values in bw_slow_list]),
        np.asarray([np.asarray(values) for values in ratio_fast_list]),
        np.asarray([np.asarray(values) for values in ratio_slow_list]),
        float(np.median(fast_rt_all)) if fast_rt_all else np.nan,
        float(np.median(slow_rt_all)) if slow_rt_all else np.nan,
        t,
        rt_type,
    )


def plot_fast_slow(
    fw_fast_arr,
    fw_slow_arr,
    bw_fast_arr,
    bw_slow_arr,
    ratio_fast_arr,
    ratio_slow_arr,
    fast_rt_avg,
    slow_rt_avg,
    t,
    rt_type="early",
    mode="hand",
    fmin=8,
    fmax=12,
    freq_bands=None,
    save=False,
    save_dir=None,
):
    """Plot fast vs slow time series for FW, BW, and RATIO.

    When supplied with multiband arrays, each measure is plotted as a vertical
    stack of the requested bands.
    """
    if save:
        if save_dir is None:
            save_dir = Path(FIG_DIR)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if freq_bands is None:
        if fw_fast_arr.ndim == 3 or (fw_fast_arr.ndim == 1 and fw_fast_arr.dtype == object):
            freq_bands = [(2, 6), (8, 12), (14, 30)]
        else:
            freq_bands = [(fmin, fmax)]
    band_labels = [f"{lo}-{hi} Hz" for lo, hi in freq_bands]

    probe_time = PROBE_ONSET.get(rt_type, 0.0)
    fast_time = probe_time + fast_rt_avg
    slow_time = probe_time + slow_rt_avg

    measure_configs = [
        ("FW", fw_fast_arr, fw_slow_arr, "#1f77b4", "#08306b", "FW contra-ipsi (dB)"),
        ("BW", bw_fast_arr, bw_slow_arr, "#ff7f0e", "#a64b00", "BW contra-ipsi (dB)"),
        ("RATIO", ratio_fast_arr, ratio_slow_arr, "#9467bd", "#4b2e83", "log-ratio"),
    ]

    for label, fast_arr, slow_arr, fast_color, slow_color, ylabel in measure_configs:
        if fast_arr.ndim == 3:
            fast_arr_list = [fast_arr[i] for i in range(fast_arr.shape[0])]
            slow_arr_list = [slow_arr[i] for i in range(slow_arr.shape[0])]
        elif fast_arr.ndim == 1 and fast_arr.dtype == object:
            fast_arr_list = list(fast_arr)
            slow_arr_list = list(slow_arr)
        else:
            fast_arr_list = [fast_arr]
            slow_arr_list = [slow_arr]

        n_bands = len(fast_arr_list)
        fig, axes = plt.subplots(
            n_bands,
            1,
            figsize=(12, 4.0 * n_bands),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        mode_label = "Respond-Hand" if mode == "hand" else "Cue-Location"
        fig.suptitle(f"Fast vs Slow {rt_type}_rt — {mode_label} reference", fontsize=13)

        for band_idx, axis in enumerate(axes):
            arr_fast = fast_arr_list[band_idx]
            arr_slow = slow_arr_list[band_idx]
            mean_fast, ci_fast = _mean_ci(arr_fast)
            mean_slow, ci_slow = _mean_ci(arr_slow)

            axis.plot(
                t,
                mean_fast,
                color=fast_color,
                linewidth=2.4,
                label=f"Fast RT ({int(round(fast_rt_avg * 1000))} ms)",
            )
            axis.fill_between(t, mean_fast - ci_fast, mean_fast + ci_fast, color=fast_color, alpha=0.18)
            axis.plot(
                t,
                mean_slow,
                color=slow_color,
                linewidth=2.4,
                label=f"Slow RT ({int(round(slow_rt_avg * 1000))} ms)",
            )
            axis.fill_between(t, mean_slow - ci_slow, mean_slow + ci_slow, color=slow_color, alpha=0.18)

            axis.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            axis.axvline(fast_time, color=fast_color, linestyle="--", linewidth=1.5, alpha=0.85)
            axis.axvline(slow_time, color=slow_color, linestyle="--", linewidth=1.5, alpha=0.85)

            sig_fast, details_fast = _cluster_1d(arr_fast)
            sig_slow, details_slow = _cluster_1d(arr_slow)
            sig_diff, details_diff = _cluster_1d(arr_fast - arr_slow)
            if sig_fast.any():
                ymin, ymax = axis.get_ylim()
                y_fast = ymax - 0.05 * (ymax - ymin)
                _draw_sig_bar(axis, sig_fast, t, y_fast, color=fast_color)
            if sig_slow.any():
                ymin, ymax = axis.get_ylim()
                y_slow = ymax - 0.10 * (ymax - ymin)
                _draw_sig_bar(axis, sig_slow, t, y_slow, color=slow_color)
            if sig_diff.any():
                ymin, ymax = axis.get_ylim()
                y_diff = ymax - 0.15 * (ymax - ymin)
                _draw_sig_bar(axis, sig_diff, t, y_diff, color="black")

            # Print cluster p-values to console (only significant ones)
            fast_sig_details = [d for d in details_fast if d.get('significant')]
            slow_sig_details = [d for d in details_slow if d.get('significant')]
            diff_sig_details = [d for d in details_diff if d.get('significant')]
            if fast_sig_details:
                print(f"[{rt_type}] {label} | {band_labels[band_idx]}: fast clusters:")
                _print_cluster_details(fast_sig_details, t=t, only_significant=True)
            if slow_sig_details:
                print(f"[{rt_type}] {label} | {band_labels[band_idx]}: slow clusters:")
                _print_cluster_details(slow_sig_details, t=t, only_significant=True)
            if diff_sig_details:
                print(f"[{rt_type}] {label} | {band_labels[band_idx]}: fast-slow diff clusters:")
                _print_cluster_details(diff_sig_details, t=t, only_significant=True)

            axis.set_title(f"{band_labels[band_idx]} {label} — {mode_label} ref", fontsize=12)
            axis.set_ylabel(ylabel, fontsize=11)
            axis.legend(fontsize=10, loc="upper right")
            axis.grid(alpha=0.12)

        axes[-1].set_xlabel("Time (s)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            out = save_dir / f"fast_slow_{label.lower()}_{rt_type}_{mode}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")

        plt.show()




# ── TF maps ────────────────────────────────────────────────────────────────────

def plot_tf_maps(
    by_subj,
    subj_ids,
    mode="hand",
    rt_type="early",
    measure=None,
    contra_idx=None,
    ipsi_idx=None,
    mid_idx=None,
    trial_filter=None,
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    t_start=0,
    t_stop=None,
    fmax=None,
    vabs=0.3,
    vratio=0.03,
    run_cluster=True,
    n_perm=5000,
    exclude_invalid_rt=True,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Plot subject-level contra, ipsi, midline, and contra-ipsi TF maps."""
    measures = MEASURES if measure is None else (
        [measure] if isinstance(measure, str) else list(measure)
    )

    data = _subject_contra_ipsi(
        by_subj,
        subj_ids,
        mode=mode,
        rt_type=rt_type,
        baseline_correct=baseline_correct,
        bl0=bl0,
        bl1=bl1,
        contra_idx=contra_idx,
        ipsi_idx=ipsi_idx,
        mid_idx=mid_idx,
        trial_filter=trial_filter,
        exclude_invalid_rt=exclude_invalid_rt,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )
    t = data["time"]
    ff = data["ff"]
    if t_stop is None:
        t_stop = float(t[-1])
    tmask = (t >= t_start) & (t <= t_stop)
    fmask = np.ones(len(ff), dtype=bool) if fmax is None else ff <= fmax
    t_sel = t[tmask]
    ff_sel = ff[fmask]

    early_line, late_line = _compute_rt_lines(
        by_subj,
        subj_ids,
        trial_filter=trial_filter,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )

    filter_tag = trial_filter or "all"
    save_dir = Path(FIG_DIR) / ("bl" if baseline_correct else "no-bl") / filter_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = f"{rt_type}_hand" if mode == "hand" else "cue"
    mode_label = f"Respond-Hand ({rt_type})" if mode == "hand" else "Cue-Location"

    results = {}
    for mkey in measures:
        side_arrays = {
            side: data[mkey][side][:, fmask][:, :, tmask]
            for side in ("contra", "ipsi", "mid")
        }
        diff_array = side_arrays["contra"] - side_arrays["ipsi"]
        all_values = np.concatenate(
            [values.ravel() for values in (*side_arrays.values(), diff_array)]
        )
        requested_limit = vratio if mkey == "ratio" else vabs
        value_limit = requested_limit if requested_limit is not None else max(
            float(np.nanpercentile(np.abs(all_values), 98)),
            0.03,
        )
        colorbar_label = "log-ratio" if mkey == "ratio" else "Mean power (dB)"
        measure_label = MEASURE_LABELS.get(mkey, mkey.upper())

        def plot_map(subject_array, ax, title, event_labels=True):
            mean_map = subject_array.mean(axis=0)
            sig, details = (
                _cluster_2d(subject_array, n_perm=n_perm)
                if run_cluster else (np.zeros_like(mean_map, dtype=bool), [])
            )
            image = ax.pcolormesh(
                t_sel,
                ff_sel,
                mean_map,
                cmap="RdBu_r",
                vmin=-value_limit,
                vmax=value_limit,
                shading="auto",
            )
            if sig.any():
                ax.contour(
                    t_sel,
                    ff_sel,
                    sig.astype(float),
                    levels=[0.5],
                    colors="k",
                    linewidths=1.5,
                    zorder=20,
                )
            plt.colorbar(image, ax=ax, pad=0.02).set_label(colorbar_label)
            _decorate_axis(
                ax,
                event_labels=event_labels,
                xlim=(t_start, t_stop),
            )
            _draw_rt_lines(ax, early_line, late_line)
            ax.set_title(title, fontsize=13)
            ax.set_ylabel("Frequency [Hz]", fontsize=14)
            ax.set_ylim(ff_sel[0], ff_sel[-1])
            ax.tick_params(axis="both", labelsize=12)
            if run_cluster:
                print(f"{title} 2D clusters:")
                _print_cluster_details(
                    details,
                    t=t_sel,
                    ff=ff_sel,
                    only_significant=True,
                )
            return details

        side_fig, side_axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        side_details = {}
        for axis, (side, subject_array) in zip(side_axes, side_arrays.items()):
            side_details[side] = plot_map(
                subject_array,
                axis,
                f"{measure_label} ({side.upper()}) -- {mode_label}",
                event_labels=(axis is side_axes[-1]),
            )
        side_axes[-1].set_xlabel("Time (s)", fontsize=13)
        side_fig.tight_layout()
        side_out = save_dir / f"{mkey}_{mode_tag}_tf_contra-ipsi-mid.png"
        side_fig.savefig(side_out, dpi=200, bbox_inches="tight")
        print(f"[saved] {side_out}")
        plt.show()

        diff_fig, diff_ax = plt.subplots(figsize=(12, 4.5))
        diff_details = plot_map(
            diff_array,
            diff_ax,
            f"{measure_label} (CONTRA-IPSI) -- {mode_label}",
        )
        diff_ax.set_xlabel("Time (s)", fontsize=13)
        diff_fig.tight_layout()
        diff_out = save_dir / f"{mkey}_{mode_tag}_tf.png"
        diff_fig.savefig(diff_out, dpi=200, bbox_inches="tight")
        print(f"[saved] {diff_out}")
        plt.show()

        results[mkey] = {
            "diff": (diff_fig, diff_ax, diff_details),
            "sides": (side_fig, side_axes, side_details),
        }

    return results, data


# ── Band time-series ───────────────────────────────────────────────────────────

def plot_band_timeseries(
    by_subj,
    subj_ids,
    mode="hand",
    rt_type="early",
    measure=None,
    contra_idx=None,
    ipsi_idx=None,
    mid_idx=None,
    trial_filter=None,
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    t_start=0,
    t_stop=None,
    freq_bands=FREQ_BANDS,
    ylim=None,
    split_measures=True,
    run_cluster=True,
    n_perm=5000,
    exclude_invalid_rt=True,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Plot contra-ipsi band time courses for each requested measure."""
    measures = MEASURES if measure is None else (
        [measure] if isinstance(measure, str) else list(measure)
    )

    data = _subject_contra_ipsi(
        by_subj,
        subj_ids,
        mode=mode,
        rt_type=rt_type,
        baseline_correct=baseline_correct,
        bl0=bl0,
        bl1=bl1,
        contra_idx=contra_idx,
        ipsi_idx=ipsi_idx,
        mid_idx=mid_idx,
        trial_filter=trial_filter,
        exclude_invalid_rt=exclude_invalid_rt,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )
    t = data["time"]
    ff = data["ff"]
    if t_stop is None:
        t_stop = float(t[-1])
    tmask = (t >= t_start) & (t <= t_stop)
    t_sel = t[tmask]

    early_line, late_line = _compute_rt_lines(
        by_subj,
        subj_ids,
        trial_filter=trial_filter,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )

    if ylim is None:
        ylim = [DEFAULT_YLIMS[mkey] for mkey in measures]
    ylim_by_measure = dict(zip(measures, ylim))

    diff_arrays = {
        mkey: data[mkey]["contra"] - data[mkey]["ipsi"]
        for mkey in measures
    }
    filter_tag = trial_filter or "all"
    save_dir = Path(FIG_DIR) / ("bl" if baseline_correct else "no-bl") / filter_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = f"{rt_type}_hand" if mode == "hand" else "cue"
    mode_label = f"Respond-Hand ({rt_type})" if mode == "hand" else "Cue-Location"

    results = {}

    def make_figure(measure_keys):
        n_rows = len(measure_keys)
        n_cols = len(freq_bands)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 3.3 * n_rows),
            sharex=True,
            sharey="row",
            squeeze=False,
        )
        cluster_details = {mkey: {} for mkey in measure_keys}

        for row, mkey in enumerate(measure_keys):
            color = MEASURE_COLORS[mkey]
            for col, (fmin, fmax, band_label) in enumerate(freq_bands):
                ax = axes[row, col]
                fmask = (ff >= fmin) & (ff <= fmax)
                band_data = diff_arrays[mkey][:, fmask].mean(axis=1)[:, tmask]
                mean, ci = _mean_ci(band_data)

                ax.plot(t_sel, mean, color=color, linewidth=2.0)
                ax.fill_between(
                    t_sel,
                    mean - ci,
                    mean + ci,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8, alpha=0.65)
                _decorate_axis(
                    ax,
                    event_labels=(row == n_rows - 1),
                    label_fontsize=8,
                    label_rotation=45,
                    label_y=-0.20,
                    xlim=(t_start, t_stop),
                )
                _draw_rt_lines(ax, early_line, late_line)

                current_ylim = ylim_by_measure[mkey]
                if current_ylim is not None:
                    ax.set_ylim(*current_ylim)

                details = []
                if run_cluster:
                    sig, details = _cluster_1d(band_data, n_perm=n_perm)
                    if sig.any():
                        ymin, ymax = ax.get_ylim()
                        _draw_sig_bar(
                            ax,
                            sig,
                            t_sel,
                            ymax - 0.05 * (ymax - ymin),
                            color=color,
                        )
                    print(f"{mode_label} | {mkey} | {band_label}:")
                    _print_cluster_details(details, t=t_sel, only_significant=True)
                cluster_details[mkey][band_label] = details

                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(
                    axis="both",
                    labelsize=11,
                    labelbottom=(row == n_rows - 1),
                    labelleft=(col == 0),
                )
                if row == 0:
                    ax.set_title(band_label, fontsize=11, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(MEASURE_YLABELS[mkey], fontsize=11)
                if row == n_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=11)

        measure_tag = "_".join(measure_keys)
        fig.suptitle(
            f"{measure_tag.upper()} contra-ipsi band power -- {mode_label}",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        out = save_dir / f"{measure_tag}_{mode_tag}_band.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.show()
        results[measure_tag] = {
            "figure": fig,
            "axes": axes,
            "clusters": cluster_details,
            "path": out,
        }

    if split_measures:
        for mkey in measures:
            make_figure([mkey])
    else:
        make_figure(measures)

    return results, data


def _align_subjects(data_a, data_b, measure):
    subj_a = list(data_a["subjects"])
    subj_b = list(data_b["subjects"])
    common = [subj for subj in subj_a if subj in subj_b]
    idx_a = [subj_a.index(subj) for subj in common]
    idx_b = [subj_b.index(subj) for subj in common]
    return data_a[measure][idx_a], data_b[measure][idx_b], common


def plot_line_power_bands(
    by_subj,
    subj_ids,
    rt_type="early",
    measure=None,
    trial_filter='all',
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    t_start=0,
    t_stop=7.5,
    freq_bands=FREQ_BANDS,
    line_idx=None,
    run_cluster=True,
    n_perm=5000,
    exclude_invalid_rt=True,
    max_rt=None,
    ylim=None,   # list of (ymin, ymax), one per entry in `measure`, in the same order; None = LINE_POWER_YLIMS per measure
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Plot TW power averaged over electrode lines, one 1x3 figure per measure."""
    if max_rt is None:
        max_rt = MAX_RT
    measures = MEASURES if measure is None else ([measure] if isinstance(measure, str) else list(measure))
    if ylim is None:
        ylim = [LINE_POWER_YLIMS[mk] for mk in measures]
    _ylim_map = dict(zip(measures, ylim))
    data = _subject_line_power(
        by_subj,
        subj_ids,
        rt_type=rt_type,
        trial_filter=trial_filter,
        baseline_correct=baseline_correct,
        bl0=bl0,
        bl1=bl1,
        line_idx=line_idx,
        exclude_invalid_rt=exclude_invalid_rt,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )
    t = data["time"]
    ff = data["ff"]
    tmask = (t >= t_start) & (t <= t_stop)
    t_sel = t[tmask]

    early_line, late_line = _compute_rt_lines(
        by_subj, subj_ids, trial_filter=trial_filter, max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue, exclude_bad_epoch=exclude_bad_epoch,
    )

    fig_dir = Path(FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mkey in measures:
        arr = data[mkey]
        fig, axes = _band_axes(len(freq_bands), row_height=4.2)
        color = MEASURE_COLORS.get(mkey, "black")

        for col, (ax, (f0, f1, label)) in enumerate(zip(axes, freq_bands)):
            fmask = (ff >= f0) & (ff <= f1)
            y = arr[:, fmask].mean(axis=1)[:, tmask]
            mean, ci = _mean_ci(y)
            ax.plot(t_sel, mean, color=color, linewidth=1.8)
            ax.fill_between(t_sel, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0)
            _decorate_axis(
                ax,
                event_labels=True,
                label_fontsize=8,
                label_rotation=32,
                label_y=-0.20,
                xlim=(t_start, t_stop),
            )
            _draw_rt_lines(ax, early_line, late_line)
            ax.tick_params(axis="both", labelsize=11)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=11)
            if col == 0:
                unit = 'log-ratio' if mkey == 'ratio' else 'Mean power (dB)'
                ax.set_ylabel(f"{MEASURE_LABELS.get(mkey, mkey.upper())}\n{unit}", fontsize=11)

            cur_ylim = _ylim_map[mkey]
            if cur_ylim is not None:
                ax.set_ylim(*cur_ylim)
            if run_cluster:
                sig, details = _cluster_1d(y, n_perm=n_perm)
                if sig.any():
                    ymin, ymax = ax.get_ylim()
                    _draw_sig_bar(ax, sig, t_sel, ymax - 0.03 * (ymax - ymin), color=color)
                print(f"{mkey} {label}, line-power, hand-{rt_type}, {trial_filter}, max_rt={max_rt}:")
                _print_cluster_details(details, t=t_sel, only_significant=True)

        fig.suptitle(
            f"11-line Average {MEASURE_LABELS.get(mkey, mkey.upper())} -- "
            f"Respond-Hand ({rt_type}) reference",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.975])

        out = fig_dir / f"line_power_bands_{mkey}_hand-{rt_type}_{trial_filter}.png"
        fig.savefig(out, dpi=200)
        print(f"[saved] {out}")
        plt.show()
        results[mkey] = (fig, axes)

    return results, data


def plot_fixed_random_line_power_bands(
    by_subj,
    subj_ids,
    rt_type="early",
    measure=None,
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
    t_start=0,
    t_stop=7.5,
    freq_bands=FREQ_BANDS,
    line_idx=None,
    run_cluster=True,
    n_perm=5000,
    exclude_invalid_rt=True,
    max_rt=None,
    ylim=None,   # list of (ymin, ymax), one per entry in `measure`, in the same order; None = FIXED_RANDOM_YLIMS per measure
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Plot fixed vs random comparison for TW power averaged over 11 lines."""
    if max_rt is None:
        max_rt = MAX_RT
    measures = MEASURES if measure is None else ([measure] if isinstance(measure, str) else list(measure))
    if ylim is None:
        ylim = [FIXED_RANDOM_YLIMS[mk] for mk in measures]
    _ylim_map = dict(zip(measures, ylim))
    fixed = _subject_line_power(
        by_subj,
        subj_ids,
        rt_type=rt_type,
        trial_filter='fixed',
        baseline_correct=baseline_correct,
        bl0=bl0,
        bl1=bl1,
        line_idx=line_idx,
        exclude_invalid_rt=exclude_invalid_rt,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )
    random = _subject_line_power(
        by_subj,
        subj_ids,
        rt_type=rt_type,
        trial_filter='random',
        baseline_correct=baseline_correct,
        bl0=bl0,
        bl1=bl1,
        line_idx=line_idx,
        exclude_invalid_rt=exclude_invalid_rt,
        max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue,
        exclude_bad_epoch=exclude_bad_epoch,
    )
    t = fixed["time"]
    ff = fixed["ff"]
    tmask = (t >= t_start) & (t <= t_stop)
    t_sel = t[tmask]

    early_line, late_line = _compute_rt_lines(
        by_subj, subj_ids, trial_filter='all', max_rt=max_rt,
        exclude_timing_issue=exclude_timing_issue, exclude_bad_epoch=exclude_bad_epoch,
    )

    fig_dir = Path(FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fixed": "#1f77b4", "random": "#d55e00"}
    results = {}
    for mkey in measures:
        fixed_arr, random_arr, common = _align_subjects(fixed, random, mkey)
        fig, axes = _band_axes(len(freq_bands), row_height=4.2)

        for col, (ax, (f0, f1, label)) in enumerate(zip(axes, freq_bands)):
            fmask = (ff >= f0) & (ff <= f1)
            y_fixed = fixed_arr[:, fmask].mean(axis=1)[:, tmask]
            y_random = random_arr[:, fmask].mean(axis=1)[:, tmask]

            for name, y in [("fixed", y_fixed), ("random", y_random)]:
                mean, ci = _mean_ci(y)
                ax.plot(t_sel, mean, color=colors[name], linewidth=1.8, label=name)
                ax.fill_between(t_sel, mean - ci, mean + ci, color=colors[name], alpha=0.14, linewidth=0)

            _decorate_axis(
                ax,
                event_labels=True,
                label_fontsize=8,
                label_rotation=32,
                label_y=-0.20,
                xlim=(t_start, t_stop),
            )
            _draw_rt_lines(ax, early_line, late_line)
            ax.tick_params(axis="both", labelsize=11)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=11)
            if col == 0:
                unit = 'log-ratio' if mkey == 'ratio' else 'Mean power (dB)'
                ax.set_ylabel(f"{MEASURE_LABELS.get(mkey, mkey.upper())}\n{unit}", fontsize=11)

            cur_ylim = _ylim_map[mkey]
            if cur_ylim is not None:
                ax.set_ylim(*cur_ylim)
            if run_cluster:
                diff = y_fixed - y_random
                sig, details = _cluster_1d(diff, n_perm=n_perm)
                if sig.any():
                    ymin, ymax = ax.get_ylim()
                    _draw_sig_bar(ax, sig, t_sel, ymax - 0.06 * (ymax - ymin), color="#111111")
                print(f"{mkey} {label}, line-power fixed-random, hand-{rt_type}, max_rt={max_rt}:")
                _print_cluster_details(details, t=t_sel, only_significant=True)

        axes[0].legend(fontsize=10, loc="upper right")
        fig.suptitle(
            f"11-line Average {MEASURE_LABELS.get(mkey, mkey.upper())}: fixed vs random",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.975])

        out = fig_dir / f"{mkey}_all.png"
        fig.savefig(out, dpi=200)
        print(f"[saved] {out}")
        plt.show()
        results[mkey] = (fig, axes)

    return results, {"fixed": fixed, "random": random}


def main():
    parser = argparse.ArgumentParser(description="Plot Mingmin sequence contra-ipsi TW results.")
    parser.add_argument("--measure", choices=MEASURES, default="ratio")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--contra-lines", default=None, help="Comma-separated aligned contra lines, e.g. C3,C2,C1.")
    parser.add_argument("--ipsi-lines", default=None, help="Comma-separated aligned ipsi lines, e.g. I1,I2,I3.")
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory with subj*_sess*.pkl result files.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    contra_idx = None if args.contra_lines is None else [x.strip() for x in args.contra_lines.split(",") if x.strip()]
    ipsi_idx = None if args.ipsi_lines is None else [x.strip() for x in args.ipsi_lines.split(",") if x.strip()]

    by_subj, subj_ids = load_results(args.result_dir)
    print(f"Loaded {sum(len(v) for v in by_subj.values())} sessions from {args.result_dir}")
    print(f"Subjects: {subj_ids}")
    print_trial_summary(by_subj, subj_ids)

    filters = ['all', 'fixed', 'random']
    for rt_type in ["early", "late"]:
        for trial_filter in filters:
            plot_tf_maps(
                by_subj,
                subj_ids,
                rt_type=rt_type,
                trial_filter=trial_filter,
                measure=args.measure,
                contra_idx=contra_idx,
                ipsi_idx=ipsi_idx,
                n_perm=args.n_perm,
            )
            plot_band_timeseries(
                by_subj,
                subj_ids,
                mode="hand",
                rt_type=rt_type,
                trial_filter=trial_filter,
                measure=args.measure,
                contra_idx=contra_idx,
                ipsi_idx=ipsi_idx,
                n_perm=args.n_perm,
            )

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
