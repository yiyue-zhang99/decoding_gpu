"""micheal_plot_funcs.py

All plotting and analysis functions for the Michael FFT results.
Import this module in notebooks instead of defining functions inline.
"""

import json
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

# ── Global defaults ────────────────────────────────────────────────────────────
RAW_MAT_ROOT = '/home/dilay/project2/tw/Data/Micheal_Data_exp2'
RAW_TRIAL_TOTALS_CACHE = '/home/dilay/project2/tw/results/micheal_fft/raw_trial_totals.json'
SAVE_ROOT   = '/home/dilay/project2/tw/results/figs/micheal'
FIG_DIR     = Path(SAVE_ROOT)

ALPHA = 0.05
BL0, BL1    = -0.75, -0.25
BL_CORRECT  = False
TMIN_BASE   = -1.25
SFREQ       = 500
WINDOW_SIZE = 250
MAX_RT      = 1.5  # trial-validity RT cutoff 

LINE_NAMES = ['L5', 'L4', 'L3', 'L2', 'L1', 'M', 'R1', 'R2', 'R3', 'R4', 'R5']
FLIPPED_LINE_NAMES = ['C5', 'C4', 'C3', 'C2', 'C1', 'M', 'I1', 'I2', 'I3', 'I4', 'I5']
LINE_ALIASES = {
    **{name.upper(): idx for idx, name in enumerate(LINE_NAMES)},
    **{name.upper(): idx for idx, name in enumerate(FLIPPED_LINE_NAMES)},
    'MID': 5,
    'MIDLINE': 5,
}

FREQ_BANDS = [
    (2,  6,  'Theta (2-6 Hz)'),
    (8,  12, 'Alpha (8-12 Hz)'),
    (14, 30, 'Beta (14-30 Hz)'),
]
MEASURES    = ['fw', 'bw', 'ratio']
MEASURE_COLORS = {
    'fw': '#1f77b4',    # blue
    'bw': '#ff7f0e',    # orange
    'ratio': '#9467bd', # purple
}
MEASURE_LABELS = {
    'fw': 'FW',
    'bw': 'BW',
    'ratio': 'FW/BW ratio',
}
MEASURE_YLABELS = {
    'fw': 'FW contra-ipsi (dB)',
    'bw': 'BW contra-ipsi (dB)',
    'ratio': 'log-ratio',
}


EVENT_TIMES  = [0, 1.2, 1.8, 3.8, 4.3]
EVENT_LABELS = ['target onset', '1st impulse', '1st probe', '2nd impulse', '2nd probe']
PROBE_ONSET  = {'early': 1.8, 'late': 4.3}
BAR_WIDTH_LONG  = 0.3
BAR_WIDTH_SHORT = 0.15
STIM_BARS = [
    (0,   BAR_WIDTH_LONG), (1.2, BAR_WIDTH_SHORT), (1.8, BAR_WIDTH_LONG),
    (3.8, BAR_WIDTH_SHORT), (4.3, BAR_WIDTH_LONG),
]
DEFAULT_YLIMS = {'fw': (-0.1, 0.1), 'bw': (-0.1, 0.1), 'ratio': (-0.02, 0.02)}


# ----------------------------- load data -----------------------------
def load_results(result_dir: Path):
    by_subj = defaultdict(dict)
    for path in sorted(result_dir.glob("subj*_sess*.pkl")):
        match = re.search(r"subj(\d+)_sess(\d+)", path.name)
        if match is None:
            continue
        subj, sess = int(match.group(1)), int(match.group(2))
        with path.open("rb") as fp:
            data = pickle.load(fp)
        data.setdefault("starts", data["time"])
        data.setdefault("cue", data["cue_loc"])
        by_subj[subj][sess] = data
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
            valid = ', '.join(FLIPPED_LINE_NAMES + LINE_NAMES)
            raise ValueError(f"Unknown line index/name {item!r}. Valid names: {valid}")
        resolved.append(LINE_ALIASES[key])
    return resolved


def _line_selection_tag(contra_idx, ipsi_idx, mid_idx=None):
    contra = resolve_line_idx(contra_idx, range(5))
    ipsi = resolve_line_idx(ipsi_idx, range(6, 11))
    mid = resolve_line_idx(mid_idx, [5]) if mid_idx is not None else []
    contra_tag = "-".join(FLIPPED_LINE_NAMES[idx] for idx in contra)
    ipsi_tag = "-".join(FLIPPED_LINE_NAMES[idx] for idx in ipsi)
    if mid:
        mid_tag = "-".join(FLIPPED_LINE_NAMES[idx] for idx in mid)
        return f"{contra_tag}_vs_{ipsi_tag}_mid-{mid_tag}"
    return f"{contra_tag}_vs_{ipsi_tag}"




# ── Trial filters ──────────────────────────────────────────────────────────────
def _respond_left_mask(d, rt_type):
    return (
        ((d[f'{rt_type}_rot'] == 1) & (d[f'{rt_type}_acc'] == 1)) |
        ((d[f'{rt_type}_rot'] != 1) & (d[f'{rt_type}_acc'] == 0))
    )


def _flip_by_response_hand(arr, d, rt_type):
    """Flip left-response trials so every trial uses a right-hemi reference."""
    arr = arr.copy()
    left = _respond_left_mask(d, rt_type)
    arr[:, :, :, left] = arr[::-1, :, :, left]
    return arr


def _trial_filter_mask(d, trial_filter=None):
    """Resolve an all/ipsi/contra filter using the late recall item as cue.

    ``ipsi`` selects trials whose response hand is on the same side as the
    late recall item; ``contra`` selects the opposite side.
    """
    n_trials = len(d["early_rt"])
    if trial_filter is None or trial_filter == "all":
        return np.ones(n_trials, dtype=bool)
    if trial_filter not in {"ipsi", "contra"}:
        raise ValueError(
            "trial_filter should be 'all', 'ipsi', or 'contra'; "
            f"got {trial_filter!r}"
        )

    same_side = _respond_left_mask(d, "late") == (np.asarray(d["late_rot"]) == 1)
    return same_side if trial_filter == "ipsi" else ~same_side


def _rt_valid_mask(d, max_rt=None):
    """Return a boolean mask for trials kept after basic RT/validity exclusions."""
    if max_rt is None:
        max_rt = MAX_RT
    return (~np.isnan(d['early_rt'])) & (d['late_rt'] <= max_rt)


def _resolve_trial_mask(
    d,
    trial_filter=None,
    exclude_invalid_rt=False,
    max_rt=None,
    exclude_timing_issue=True,
    exclude_bad_epoch=False,
):
    """Build the boolean mask used by plotting functions to select trials."""
    mask = np.ones(len(d['early_rt']), dtype=bool)

    if exclude_invalid_rt:
        mask = mask & _rt_valid_mask(d, max_rt=max_rt)

    mask = mask & _trial_filter_mask(d, trial_filter)
    if exclude_timing_issue and "has_timing_issue" in d:
        mask &= ~np.asarray(d["has_timing_issue"], dtype=bool)
    if exclude_bad_epoch and "is_bad_epoch" in d:
        mask &= ~np.asarray(d["is_bad_epoch"], dtype=bool)

    return mask


# ----trial summary printing  ----

def _find_raw_mat_path(subj, sess, root=RAW_MAT_ROOT):
    pattern = f"MemImp3_mem_whole_sess{sess}_{subj}.mat"
    matches = list(Path(root).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one raw file for {pattern} in {root}, found {len(matches)}: {matches}")
    return matches[0]


def _get_raw_trial_totals(by_subj, subj_ids, root=RAW_MAT_ROOT,
                          cache_path=RAW_TRIAL_TOTALS_CACHE, force_reload=False):
    """Return {(subj, sess): (n_total_raw, n_bad)} read straight from the source .mat files.

    n_total_raw is the trial count before any bad-trial removal (data.Results.shape[0]);
    n_bad is the size of the union of bad_trials_mem/imp1/imp2/probe1/probe2 for that file.
    Results are cached to disk since each .mat file is ~1GB and slow to load.
    """
    if not force_reload and cache_path and Path(cache_path).exists():
        with open(cache_path) as fp:
            raw = json.load(fp)
        return {tuple(map(int, k.split('_'))): tuple(v) for k, v in raw.items()}

    from scipy.io import loadmat

    totals = {}
    for subj in subj_ids:
        for sess in sorted(by_subj[subj]):
            mat_path = _find_raw_mat_path(subj, sess, root=root)
            eeg = loadmat(mat_path, struct_as_record=False)
            data = eeg['ft_mem'].item()
            n_total_raw = int(np.asarray(data.Results).shape[0])
            bad_trials = np.unique(
                np.concatenate(
                    [
                        np.asarray(data.bad_trials_mem).reshape(-1),
                        np.asarray(data.bad_trials_imp1).reshape(-1),
                        np.asarray(data.bad_trials_imp2).reshape(-1),
                        np.asarray(data.bad_trials_probe1).reshape(-1),
                        np.asarray(data.bad_trials_probe2).reshape(-1),
                    ]
                )
            )
            totals[(subj, sess)] = (n_total_raw, int(len(bad_trials)))
            print(f"[_get_raw_trial_totals] subj={subj} sess={sess} "
                  f"n_total_raw={n_total_raw} n_bad={len(bad_trials)}")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as fp:
            json.dump({f"{s}_{v}": list(totals[(s, v)]) for (s, v) in totals}, fp, indent=1)

    return totals


def print_trial_summary(by_subj, subj_ids, trial_filter=None, max_rt=None, raw_totals=None):
    """Print per-subject (sessions pooled) and group-level trial count summary.

    Every percentage below is out of the same denominator (total = raw trial
    count), so they add up cleanly:
        kept% + total_removed% = 100%
        total_removed% = bad_removed% + rt_removed%

    total          = original trial count straight from the raw .mat file (data.Results),
                     summed across a subject's sessions, before any exclusion.
    removed_bad    = bad trials only (union of bad_trials_mem/imp1/imp2/probe1/probe2),
                     i.e. the trials dropped upstream in micheal_newbl.py.
    kept           = total - removed_bad; should match sum(len(d['early_rt'])) as a
                     consistency check.
    rt_kept        = kept trials that additionally pass _rt_valid_mask/trial_filter
                     (RT-based validity).
    bad_removed%   = removed_bad / total.
    rt_removed%    = (kept - rt_kept) / total -- RT-only loss, out of the raw total.
    total_removed% = (total - rt_kept) / total.
    kept%          = rt_kept / total -- what's left after bad-trial + RT exclusion.
    """
    if max_rt is None:
        max_rt = MAX_RT
    if raw_totals is None:
        raw_totals = _get_raw_trial_totals(by_subj, subj_ids)

    print("Trial summary (per subject, sessions pooled; all percentages are out of the "
          "raw total, so kept% + total_removed% = 100% and total_removed% = "
          "bad_removed% + rt_removed%):")
    print("subj\ttotal\tkept%\tbad_removed%\trt_removed%\ttotal_removed%")

    total_trials      = 0
    total_removed_bad = 0
    total_rt_kept     = 0
    subj_kept_pct          = []
    subj_bad_removed_pct   = []
    subj_rt_removed_pct    = []
    subj_total_removed_pct = []

    for subj in subj_ids:
        subj_total   = 0
        subj_removed = 0
        subj_kept_n  = 0
        subj_rt_kept = 0

        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            n_total, n_removed = raw_totals[(subj, sess)]
            n_kept = n_total - n_removed
            if n_kept != len(d['early_rt']):
                print(f"  [warn] subj={subj} sess={sess}: kept={n_kept} from raw bad_trials "
                      f"!= len(early_rt)={len(d['early_rt'])} in cached pkl")
            rt_keep = _rt_valid_mask(d, max_rt=max_rt) & _trial_filter_mask(d, trial_filter)
            n_rt_kept = int(np.sum(rt_keep))

            subj_total   += n_total
            subj_removed += n_removed
            subj_kept_n  += n_kept
            subj_rt_kept += n_rt_kept

        total_trials      += subj_total
        total_removed_bad += subj_removed
        total_rt_kept     += subj_rt_kept

        bad_removed_pct   = 100.0 * subj_removed / subj_total if subj_total > 0 else 0.0
        rt_removed_pct    = 100.0 * (subj_kept_n - subj_rt_kept) / subj_total if subj_total > 0 else 0.0
        total_removed_pct = 100.0 * (subj_total - subj_rt_kept) / subj_total if subj_total > 0 else 0.0
        kept_pct          = 100.0 * subj_rt_kept / subj_total if subj_total > 0 else 0.0
        subj_kept_pct.append(kept_pct)
        subj_bad_removed_pct.append(bad_removed_pct)
        subj_rt_removed_pct.append(rt_removed_pct)
        subj_total_removed_pct.append(total_removed_pct)

        print(f"{subj}\t{subj_total}\t{kept_pct:.2f}%\t{bad_removed_pct:.2f}%\t"
              f"{rt_removed_pct:.2f}%\t{total_removed_pct:.2f}%")

    if total_trials > 0:
        overall_bad_removed_pct   = 100.0 * total_removed_bad / total_trials
        overall_rt_removed_pct    = 100.0 * (total_trials - total_removed_bad - total_rt_kept) / total_trials
        overall_total_removed_pct = 100.0 * (total_trials - total_rt_kept) / total_trials
        overall_kept_pct          = 100.0 * total_rt_kept / total_trials
    else:
        overall_bad_removed_pct = overall_rt_removed_pct = overall_total_removed_pct = overall_kept_pct = 0.0

    print("\nOverall (pooled across subjects):")
    print(f"  total trials     = {total_trials}")
    print(f"  kept%            = {overall_kept_pct:.2f}%")
    print(f"  bad_removed%     = {overall_bad_removed_pct:.2f}%")
    print(f"  rt_removed%      = {overall_rt_removed_pct:.2f}%")
    print(f"  total_removed%   = {overall_total_removed_pct:.2f}%")

    if subj_kept_pct:
        print("\nGroup average (mean ± SD across subjects):")
        print(f"  kept%          = {np.mean(subj_kept_pct):.2f} ± {np.std(subj_kept_pct, ddof=1):.2f}")
        print(f"  bad_removed%   = {np.mean(subj_bad_removed_pct):.2f} ± {np.std(subj_bad_removed_pct, ddof=1):.2f}")
        print(f"  rt_removed%    = {np.mean(subj_rt_removed_pct):.2f} ± {np.std(subj_rt_removed_pct, ddof=1):.2f}")
        print(f"  total_removed% = {np.mean(subj_total_removed_pct):.2f} ± {np.std(subj_total_removed_pct, ddof=1):.2f}")

# ── Statistics helpers ─────────────────────────────────────────────────────────

def _cluster_1d(data, alpha=ALPHA, n_perm=5000):
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


def _draw_sig_bar(ax, mask, t, y, color='#b00020', lw=5):
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    for start, stop in zip(starts, stops):
        ax.plot(
            [t[start], t[min(stop - 1, len(t) - 1)]],
            [y, y],
            color=color,
            linewidth=lw,
            solid_capstyle='butt',
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
    """Red dashed = median first (early/probe1) RT; green dashed = median second (late/probe2) RT."""
    ax.axvline(early_line, color='red', linestyle='--', linewidth=1.4, alpha=0.85, zorder=15)
    ax.axvline(late_line, color='green', linestyle='--', linewidth=1.4, alpha=0.85, zorder=15)


def _mean_ci(y):
    mean = y.mean(axis=0)
    sem = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    ci = stats.t.ppf(0.975, df=y.shape[0] - 1) * sem
    return mean, ci


def _time_axis(d, tmin_base=TMIN_BASE, sfreq=SFREQ, window_size=WINDOW_SIZE):
    starts = np.asarray(d['starts'])
    return tmin_base + (starts + window_size // 2) / sfreq


def _measure_arrays(
    d,
    baseline_correct=BL_CORRECT,
    bl0=BL0,
    bl1=BL1,
):
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
    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, row_height), sharey=True)
    axes = np.asarray(axes).ravel()
    return fig, axes
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

# ── All-lines average ──────────────────────────────────────────────────────────

def plot_all_lines_measures(
    by_subj, subj_ids,
    mode,
    rt_type='early',
    bl0=-0.75, bl1=-0.25,
    baseline_correct=BL_CORRECT,
    t_start=0, t_stop=6,
    tmin_base=TMIN_BASE, sfreq=SFREQ, window_size=WINDOW_SIZE,
    freq_bands=FREQ_BANDS,
    ylim=None,   # list of (ymin, ymax), one per entry in `measure`, in the same order; None = DEFAULT_YLIMS per measure
    trial_filter=None,
    exclude_invalid_rt=True,
    max_rt=None,
    measure=['ratio'],
    line_idx=None,   # which lines to average over, e.g. 'M' for midline only; None = all 11 lines
):
    """选定 line 的平均（不做 flip）— FW / BW / Ratio 各频段时间序列。
    每个 measure 单独画一张横排图（一行，列 = 频段）。
    """
    _idx = resolve_line_idx(line_idx, range(11))
    line_tag = '-'.join(LINE_NAMES[i] for i in _idx)

    _d0    = by_subj[subj_ids[0]][1]
    _t     = _time_axis(_d0, tmin_base=tmin_base, sfreq=sfreq, window_size=window_size)
    ff     = _d0['ff']
    t_mask = (_t >= t_start) & (_t <= t_stop)
    t_sel  = _t[t_mask]

    early_line, late_line = _compute_rt_lines(by_subj, subj_ids, trial_filter=trial_filter, max_rt=max_rt)

    fw_list, bw_list, ratio_list = [], [], []
    for subj in subj_ids:
        sess_fw, sess_bw, sess_ratio = [], [], []
        for sess in [1, 2]:
            d        = by_subj[subj][sess]
            arrays = _measure_arrays(
                d, baseline_correct=baseline_correct, bl0=bl0, bl1=bl1,
            )
            fw_db, bw_db, ratio_db = arrays['fw'], arrays['bw'], arrays['ratio']
            _tmask = _resolve_trial_mask(d, trial_filter, exclude_invalid_rt, max_rt)
            sess_fw.append(fw_db   [_idx][:, :, :, _tmask].mean(axis=(0, -1)))
            sess_bw.append(bw_db   [_idx][:, :, :, _tmask].mean(axis=(0, -1)))
            sess_ratio.append(ratio_db[_idx][:, :, :, _tmask].mean(axis=(0, -1)))
        fw_list.append(np.mean(sess_fw,    axis=0))
        bw_list.append(np.mean(sess_bw,    axis=0))
        ratio_list.append(np.mean(sess_ratio, axis=0))

    fw_arr    = np.array(fw_list)
    bw_arr    = np.array(bw_list)
    ratio_arr = np.array(ratio_list)

    _mkeys = [m.lower() for m in ([measure] if isinstance(measure, str) else measure)]
    _arrs   = {'fw': fw_arr, 'bw': bw_arr, 'ratio': ratio_arr}
    _labels = {
        'fw':    f'FW {line_tag} avg (dB)',
        'bw':    f'BW {line_tag} avg (dB)',
        'ratio': f'log-ratio {line_tag} avg',
    }
    if ylim is None:
        ylim = [DEFAULT_YLIMS[mk] for mk in _mkeys]
    _ylim_map = dict(zip(_mkeys, ylim))
    measure_configs = [(mk, _arrs[mk], MEASURE_COLORS[mk], _labels[mk]) for mk in _mkeys]

    # 每个 measure 单独画一张横排图：一行 = 该 measure，列 = 频段。
    n_cols     = len(freq_bands)
    mode_label = f'Respond-Hand ({rt_type})' if mode == 'hand' else 'Cue-Location'

    filter_tag  = trial_filter or 'all'
    save_dir    = Path(SAVE_ROOT) / ('bl' if baseline_correct else 'no-bl') / filter_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    for mname, arr, color, ylabel in measure_configs:
        cur_ylim = _ylim_map[mname]

        fig, axes = _band_axes(n_cols, row_height=3.2)

        for col, (fmin, fmax, fname) in enumerate(freq_bands):
            ax     = axes[col]
            f_mask = (ff >= fmin) & (ff <= fmax)
            band   = arr[:, f_mask, :][:, :, t_mask].mean(axis=1)
            m, ci  = _mean_ci(band)
            ax.plot(t_sel, m, color=color, linewidth=2.5)
            ax.fill_between(t_sel, m - ci, m + ci, color=color, alpha=0.15, linewidth=0)

            if cur_ylim is not None:
                ax.set_ylim(*cur_ylim)
            ymin_ax, ymax_ax = ax.get_ylim()
            bar_y = ymax_ax - 0.05 * (ymax_ax - ymin_ax)

            try:
                sig_mask, details = _cluster_1d(band, n_perm=5000)
                print(f'{mode_label} | {mname} | {fname}:')
                _print_cluster_details(details, t=t_sel, only_significant=True)
                if sig_mask.any():
                    _draw_sig_bar(ax, sig_mask, t_sel, bar_y, color, lw=4)
            except Exception:
                pass

            ax.axhline(0, color='0.55', lw=0.8)
            _decorate_axis(
                ax,
                event_labels=(col == 0),
                label_fontsize=9,
                label_rotation=45,
                xlim=(t_start, t_stop),
            )
            _draw_rt_lines(ax, early_line, late_line)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xticks(np.arange(t_start, t_stop + 0.01, 0.5))
            ax.tick_params(labelsize=11, labelleft=(col == 0))
            ax.set_title(fname, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

        plt.tight_layout()
        fig.savefig(save_dir / f'{mname}_{line_tag}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
