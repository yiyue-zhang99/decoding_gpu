"""guven_plot_funcs.py

Plotting/analysis functions for the Guven FFT traveling-wave results
(fft/guven/guven_tw.py output). Import this module in notebooks instead of
defining functions inline.

Scope, per instruction: cue-aligned only and grand-average traveling-wave
measures only -- no RT-based fast/slow splitting, no response-hand
contrasts. See micheal_plot_funcs.py for that style of analysis on the
Micheal dataset.

guven_tw.py runs every trial through the same absolute electrode-line order
(L5..L1, M, R1..R5) -- it does NOT reorder lines by cue side. Cue-relative
contra/ipsi flipping happens inline at each computation site instead (`arr[...,
flip] = arr[::-1, ..., flip]`, `flip = cue_loc == CUE_LEFT`), matching
micheal_plot_funcs.py's own hand/cue-mode flip style rather than a separate
helper. The line order is a palindrome around M (index 5), so reversing
L1..L5,M,R5..R1 lines up with the C5..C1, mid, I1..I5 template that cue-right
trials already match with no flip. This flip trigger (reverse when the cue is
on the LEFT) and the resulting C-then-I position order are chosen to match
micheal_plot_funcs.py's own cue-mode flip exactly (C = contralateral to cue,
I = ipsilateral to cue).
"""

import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mne.stats import permutation_cluster_1samp_test

# ── Global defaults ────────────────────────────────────────────────────────────
CACHE_DIR = '/home/dilay/project2/tw/results/guven_fft'
SAVE_ROOT = '/home/dilay/project2/tw/results/figs/guven'
alpha_level = 0.05
WINDOW_SIZE = 250  # must match guven_tw.WINDOW_SIZE

CONTRA_IDX = list(range(0, 5))    # C5..C1 (opposite side from cue, far->near midline)
MID_IDX    = [5]                   # mid
IPSI_IDX   = list(range(6, 11))    # I1..I5 (cue side, near->far midline)

# Post-flip relative line order (index 0..10), for resolve_line_idx name lookup.
RELATIVE_NAMES = ['C5', 'C4', 'C3', 'C2', 'C1', 'M',
                  'I1', 'I2', 'I3', 'I4', 'I5']

CUE_LEFT, CUE_RIGHT = 1, 2  # ResultMatrix cue_cond values; must match guven_tw.py

# Trigger timeline relative to target onset (t=0), from fft/guven/preprocess.py.
EVENT_TIMES  = [0, 0.75, 1.85, 2.45, 3.55, 4.15]
EVENT_LABELS = ['target', 'cue', 'impulse 1', 'rotation', 'impulse 2', 'probe']

# Event-marker bar widths (seconds) for the broken_barh strip drawn under each
# dashed event line -- BAR_WIDTH_LONG for sustained displays (target/cue/
# rotation/probe), BAR_WIDTH_SHORT for brief impulse flashes. Kept identical
# to micheal_plot_funcs.py/mingmin_plot_funcs.py so the marker strips read the
# same width regardless of dataset.
BAR_WIDTH_LONG  = 0.3
BAR_WIDTH_SHORT = 0.15
EVENT_STIM_BARS = [
    (0,    BAR_WIDTH_LONG), (0.75, BAR_WIDTH_LONG), (1.85, BAR_WIDTH_SHORT),
    (2.45, BAR_WIDTH_LONG), (3.55, BAR_WIDTH_SHORT), (4.15, BAR_WIDTH_LONG),
]

BASE_COLOR_FW    = '#1f77b4'  # blue
BASE_COLOR_BW    = '#ff7f0e'  # orange
BASE_COLOR_RATIO = '#9467bd'  # purple

# ylim used by default (when the caller doesn't pass one) for measures not
# listed here is None (auto-scaled). Only 'ratio' gets a fixed default, since
# switching it from 10*log10 (dB, O(0.1-ish) range) to plain log10(fw/bw)
# shrinks its natural range to roughly +-0.02.
DEFAULT_YLIMS = {'fw': (-0.1, 0.1), 'bw': (-0.1, 0.1), 'ratio': (-0.02, 0.02)}


# ── Loading ─────────────────────────────────────────────────────────────────────

def load_results(result_dir: Path):
    """Load subXX_sessionYY.pkl files and return ``(by_subj, subj_ids)``."""
    by_subj = defaultdict(dict)
    for path in sorted(result_dir.glob('sub*_session*.pkl')):
        match = re.search(r'sub(\d+)_session(\d+)', path.name)
        if match is None:
            continue
        subj, sess = int(match.group(1)), int(match.group(2))
        with path.open('rb') as fp:
            by_subj[subj][sess] = pickle.load(fp)
    return by_subj, sorted(by_subj)


def resolve_line_idx(line_idx, default):
    """Resolve line indices given as ints or RELATIVE_NAMES strings (e.g. 'I1',
    'C3' -- must be uppercase). `line_idx=None` keeps `default` (an IPSI_IDX/
    CONTRA_IDX/MID_IDX-style list).
    """
    if line_idx is None:
        return list(default)
    if isinstance(line_idx, (str, int, np.integer)):
        line_idx = [line_idx]
    name2idx = {name: i for i, name in enumerate(RELATIVE_NAMES)}
    name2idx.update({'MID': 5, 'MIDLINE': 5})  # extra aliases, matching micheal_plot_funcs.py
    resolved = []
    for item in line_idx:
        if isinstance(item, (int, np.integer)):
            resolved.append(int(item))
            continue
        if item not in name2idx:
            raise ValueError(f"Unknown line index/name {item!r}. Valid names: {RELATIVE_NAMES}")
        resolved.append(name2idx[item])
    return resolved


# ── Stats helpers ──────────────────────────────────────────────────────────────

def _print_clusters_1d(cls, ps, t_sel, label, alpha=alpha_level):
    """Print clusters with p < alpha. `alpha` can be looser than alpha_level
    (e.g. 0.1) to surface marginal clusters in the printout without changing
    which ones get drawn as significant on the plot -- clusters at or above
    alpha_level are marked '~' (marginal) vs '*' (significant) so the two
    don't get confused.
    """
    shown = False
    for i, (cl, p) in enumerate(zip(cls, ps), start=1):
        if p < alpha:
            idx = cl[0]
            marker = '*' if p < alpha_level else '~'
            print(f"[{label}] cluster {i}: p={p:.4f}{marker}, time=[{t_sel[idx].min():.2f}, {t_sel[idx].max():.2f}] s")
            shown = True
    if not shown:
        print(f"[{label}] no clusters with p<{alpha}")


def _print_clusters_2d(cls, ps, ff, t_sel, label):
    shown = False
    for i, (cl, p) in enumerate(zip(cls, ps), start=1):
        if p < alpha_level:
            f_idx, t_idx = cl
            print(f"[{label}] cluster {i}: p={p:.4f}, "
                  f"freq=[{ff[f_idx].min():.1f}, {ff[f_idx].max():.1f}] Hz, "
                  f"time=[{t_sel[t_idx].min():.2f}, {t_sel[t_idx].max():.2f}] s")
            shown = True
    if not shown:
        print(f"[{label}] no significant clusters (p<{alpha_level})")


def cluster_1d(data, threshold=None, n_perm=5000, return_details=False):
    if threshold is None:
        threshold = stats.t.ppf(1 - 0.05 / 2, df=data.shape[0] - 1)
    obs, cls, ps, _ = permutation_cluster_1samp_test(
        data, threshold=threshold, n_permutations=n_perm,
        tail=0, n_jobs=1, verbose=False, seed=42,
    )
    sig = np.zeros(data.shape[1], dtype=bool)
    details = []
    for i, (cl, p) in enumerate(zip(cls, ps), start=1):
        idx = cl[0]
        if p < alpha_level:
            sig[idx] = True
        details.append({'cluster_id': i, 'p_value': float(p),
                         'start_idx': int(idx[0]), 'end_idx': int(idx[-1]),
                         'is_significant': bool(p < alpha_level)})
    if return_details:
        return sig, details
    return sig


def draw_bar(ax, mask, y, color, t_vec, lw=4):
    segs = np.where(np.diff(np.concatenate([[0], mask.astype(int), [0]])))[0].reshape(-1, 2)
    for s, e in segs:
        ax.plot([t_vec[s], t_vec[min(e, len(t_vec) - 1)]],
                [y, y], color=color, linewidth=lw, solid_capstyle='butt')


def _decorate_events(ax, show_labels=True, label_fontsize=9, label_rotation=45, label_y=-0.1):
    for x in EVENT_TIMES:
        ax.axvline(x, color='0.25', linestyle='--', linewidth=0.9, dashes=(5, 5), alpha=0.5)
    ax.broken_barh(EVENT_STIM_BARS, (-0.05, 0.05), facecolors='gray',
                   clip_on=False, transform=ax.get_xaxis_transform())
    if show_labels:
        for x, lab in zip(EVENT_TIMES, EVENT_LABELS):
            ax.text(x - 0.05, label_y, lab, fontsize=label_fontsize, fontweight='bold', color='k',
                    rotation=label_rotation, va='top', ha='right',
                    transform=ax.get_xaxis_transform(), clip_on=False)


def _trial_mask(d, trial_filter, exclude_bad=True):
    mask = ~d['is_bad_epoch'] if exclude_bad else np.ones(len(d['is_bad_epoch']), dtype=bool)
    if trial_filter is not None:
        mask = mask & trial_filter(d)
    return mask


def print_trial_summary(by_subj, subj_ids, trial_filter=None):
    """Print per-subject (sessions pooled) and group-level trial count summary,
    in the same table style as micheal_plot_funcs.print_trial_summary.

    Unlike Micheal, guven_tw.py runs *every* trial through the wave
    computation and carries is_bad_epoch through as a flag rather than
    dropping bad trials upstream -- so there is no separate "raw total before
    removal" to re-read from the source .mat files here; the cached .pkl
    already holds the raw total (len(is_bad_epoch)) directly.

    total       = all epochs in the cache (optionally narrowed by trial_filter),
                  before is_bad_epoch removal.
    removed_bad = is_bad_epoch trials within `total`.
    kept        = total - removed_bad.
    kept%       = 100 * kept / total.

    guven has no RT-validity concept analogous to Micheal's MAX_RT, so unlike
    Micheal's version there is no rt_kept/rt_removed% column here.
    """
    if trial_filter is None:
        trial_filter = lambda d: np.ones(len(d['is_bad_epoch']), dtype=bool)

    print("Trial summary (per subject, sessions pooled; total = all epochs, "
          "removed_bad = is_bad_epoch, kept = total - removed_bad):")
    print("subj\ttotal\tkept\tremoved_bad\tkept%")

    total_all, total_kept = 0, 0
    subj_kept_pct = []

    for subj in subj_ids:
        subj_total, subj_kept = 0, 0
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            base = trial_filter(d)
            n_total = int(base.sum())
            n_kept = int((base & ~d['is_bad_epoch']).sum())
            subj_total += n_total
            subj_kept += n_kept

        kept_pct = 100.0 * subj_kept / subj_total if subj_total > 0 else 0.0
        subj_kept_pct.append(kept_pct)
        print(f"{subj}\t{subj_total}\t{subj_kept}\t{subj_total - subj_kept}\t{kept_pct:.1f}%")

        total_all += subj_total
        total_kept += subj_kept

    overall_kept_pct = 100.0 * total_kept / total_all if total_all > 0 else 0.0
    print("\nOverall (pooled across subjects):")
    print(f"  total trials  = {total_all}")
    print(f"  kept trials   = {total_kept}  (removed_bad = {total_all - total_kept}, kept ratio = {overall_kept_pct:.1f}%)")

    if subj_kept_pct:
        print("\nGroup average (mean ± SD across subjects):")
        print(f"  kept% = {np.mean(subj_kept_pct):.1f} ± {np.std(subj_kept_pct, ddof=1):.1f}")


# ── TF maps: contra / ipsi / mid ─────────────────────────────────────────────────

def plot_tf_maps_cue(
    by_subj, subj_ids,
    baseline_correct=False,
    bl0=-0.75, bl1=-0.25,
    t_start=0, t_stop=5.25,
    fmax=None,
    vabs=0.3, vratio=0.3,
    measure=('fw', 'bw', 'ratio'),
    trial_filter=None,
    exclude_bad=True,
    contra_idx=None,
    ipsi_idx=None,
    mid_idx=None,
    save_root=None,
):
    """Grand-average FW/BW/ratio TF maps split by contra-side / ipsi-side / mid
    line, plus the contra-ipsi difference map, with cluster-based significance
    across subjects.

    contra_idx/ipsi_idx/mid_idx pick which of the 11 (post cue-flip) lines go
    into each group -- pass ints (0..10) or RELATIVE_NAMES strings like 'C1',
    'I3'; None keeps the full default 5/1/5 split (CONTRA_IDX/MID_IDX/IPSI_IDX).
    """
    contra_idx = resolve_line_idx(contra_idx, CONTRA_IDX)
    ipsi_idx   = resolve_line_idx(ipsi_idx, IPSI_IDX)
    mid_idx    = resolve_line_idx(mid_idx, MID_IDX)

    d0 = next(iter(by_subj[subj_ids[0]].values()))
    time_vec = d0['epoch_tmin'] + (d0['starts'] + WINDOW_SIZE // 2) / d0['sfreq']
    win_mask = (time_vec >= t_start) & (time_vec <= t_stop)
    t_sel = time_vec[win_mask]
    ff = d0['ff']
    freq_mask = np.ones(len(ff), dtype=bool) if fmax is None else ff <= fmax
    ff_sel = ff[freq_mask]

    contra_list, ipsi_list, mid_list = (
        {'fw': [], 'bw': [], 'ratio': []},
        {'fw': [], 'bw': [], 'ratio': []},
        {'fw': [], 'bw': [], 'ratio': []},
    )

    for subj in subj_ids:
        sess_contra = {'fw': [], 'bw': [], 'ratio': []}
        sess_ipsi   = {'fw': [], 'bw': [], 'ratio': []}
        sess_mid    = {'fw': [], 'bw': [], 'ratio': []}

        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            fw_db    = 10 * np.log10(d['fwMax'] / d['fwMaxSS'])
            bw_db    = 10 * np.log10(d['bwMax'] / d['bwMaxSS'])
            ratio_db = np.log10(d['fwMax'] / d['bwMax'])
            t_s  = d['epoch_tmin'] + (d['starts'] + WINDOW_SIZE // 2) / d['sfreq']
            bl_s = (t_s >= bl0) & (t_s <= bl1)

            if baseline_correct:
                fw_db    -= fw_db   [:, :, bl_s, :].mean(axis=2, keepdims=True)
                bw_db    -= bw_db   [:, :, bl_s, :].mean(axis=2, keepdims=True)
                ratio_db -= ratio_db[:, :, bl_s, :].mean(axis=2, keepdims=True)

            mask = _trial_mask(d, trial_filter, exclude_bad=exclude_bad)
            cue_masked = d['cue_loc'][mask]
            flip = cue_masked == CUE_LEFT
            fw_aligned    = fw_db   [..., mask]  # (11, n_freq, n_win, n_trials)
            bw_aligned    = bw_db   [..., mask]
            ratio_aligned = ratio_db[..., mask]
            fw_aligned   [..., flip] = fw_aligned   [::-1, ..., flip]
            bw_aligned   [..., flip] = bw_aligned   [::-1, ..., flip]
            ratio_aligned[..., flip] = ratio_aligned[::-1, ..., flip]
            #session level
            for key, arr in (('fw', fw_aligned), ('bw', bw_aligned), ('ratio', ratio_aligned)):
                sess_contra[key].append(arr[contra_idx].mean(axis=(0, -1)))
                sess_ipsi[key].append(arr[ipsi_idx].mean(axis=(0, -1)))
                sess_mid[key].append(arr[mid_idx].mean(axis=(0, -1)))
        #subject level
        for key in ('fw', 'bw', 'ratio'):
            contra_list[key].append(np.mean(sess_contra[key], axis=0))
            ipsi_list[key].append(np.mean(sess_ipsi[key], axis=0))
            mid_list[key].append(np.mean(sess_mid[key], axis=0))

    df = len(subj_ids) - 1
    threshold = stats.t.ppf(1 - alpha_level / 2, df=df)

    def run_cluster(arr_test, label):
        _, cls, ps, _ = permutation_cluster_1samp_test(
            arr_test, threshold=threshold, n_permutations=5000, tail=0, n_jobs=1, seed=42, verbose=False)
        mask = np.zeros(arr_test.shape[1:], dtype=bool)
        for cl, p in zip(cls, ps):
            if p < alpha_level:
                mask[cl] = True
        _print_clusters_2d(cls, ps, ff_sel, t_sel, label)
        return mask

    def decorate(ax, show_labels=True):
        ax.set_yticks(np.arange(0, ff_sel[-1] + 5, 5))
        ax.set_ylabel('Frequency [Hz]', fontsize=14)
        ax.set_ylim(ff_sel[0], ff_sel[-1])
        ax.tick_params(labelsize=12)
        _decorate_events(ax, show_labels=show_labels)

    save_root = SAVE_ROOT if save_root is None else save_root
    save_dir = os.path.join(save_root, 'bl' if baseline_correct else 'no-bl')
    os.makedirs(save_dir, exist_ok=True)

    measure_keys = [measure] if isinstance(measure, str) else list(measure)
    labels = {'fw': 'FW', 'bw': 'BW', 'ratio': 'FW/BW ratio'}
    cbar_labels = {'fw': 'Mean power (dB)', 'bw': 'Mean power (dB)',
                   'ratio': 'log10(FW/BW) (dB)'}
    vlimits = {'fw': vabs, 'bw': vabs, 'ratio': vratio}

    for key in measure_keys:
        contra_arr = np.array(contra_list[key])[:, freq_mask][:, :, win_mask]
        ipsi_arr   = np.array(ipsi_list[key])  [:, freq_mask][:, :, win_mask]
        mid_arr    = np.array(mid_list[key])   [:, freq_mask][:, :, win_mask]
        diff_arr   = contra_arr - ipsi_arr

        vlim = vlimits[key]
        if vlim is None:
            vlim = np.percentile(np.abs(np.concatenate(
                [contra_arr.ravel(), ipsi_arr.ravel(), mid_arr.ravel(), diff_arr.ravel()])), 98)

        def plot_map(arr_test, ax, title, show_labels=True):
            arr_mean = arr_test.mean(axis=0)
            sig_mask = run_cluster(arr_test, title)
            im = ax.pcolormesh(t_sel, ff_sel, arr_mean, cmap='RdBu_r',
                               vmin=-vlim, vmax=vlim, shading='auto')
            if sig_mask.any():
                ax.contour(t_sel, ff_sel, sig_mask.astype(float),
                           levels=[0.5], colors='k', linewidths=1.5, zorder=20)
            plt.colorbar(im, ax=ax, pad=0.02).set_label(cbar_labels[key])
            ax.set_title(title, fontsize=13)
            decorate(ax, show_labels=show_labels)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for arr, ax, title in [
            (contra_arr, axes[0], f'{labels[key]} (CONTRA)'),
            (ipsi_arr,   axes[1], f'{labels[key]} (IPSI)'),
            (mid_arr,    axes[2], f'{labels[key]} (MID)'),
        ]:
            plot_map(arr, ax, title, show_labels=(ax is axes[-1]))
        axes[-1].set_xlabel('Time (s)', fontsize=13)
        fig.tight_layout()
        out = os.path.join(save_dir, f'{key}_cue_tf_contra-ipsi-mid.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"[saved] {out}")
        plt.show()

        fig_d, ax_d = plt.subplots(1, 1, figsize=(12, 4.5))
        plot_map(diff_arr, ax_d, f'{labels[key]} (CONTRA-IPSI)')
        ax_d.set_xlabel('Time (s)', fontsize=13)
        fig_d.tight_layout()
        out = os.path.join(save_dir, f'{key}_cue_tf.png')
        fig_d.savefig(out, dpi=150, bbox_inches='tight')
        print(f"[saved] {out}")
        plt.show()


# ── All-lines grand average ─────────────────────────────────────────────────────
def plot_all_lines_measures_cue(
    by_subj, subj_ids,
    bl0=-0.75, bl1=-0.25,
    baseline_correct=False,
    t_start=0, t_stop=5.25,
    freq_bands=None,
    ylim=None,
    trial_filter=None,
    exclude_bad=True,
    line_idx=None,   # which lines to average over, e.g. 'M' for midline only; None = all 11 lines
    measure=('fw', 'bw', 'ratio'),
    save_root=None,
):
    """Line average (no contra/ipsi split, just pooled) FW/BW/ratio time series
    per freq band. Each measure (fw/bw/ratio) is drawn as its own figure (one
    row per freq band), rather than one combined multi-column figure.

    line_idx picks which of the 11 (post cue-flip) lines go into the pooled
    average -- pass ints (0..10) or RELATIVE_NAMES strings like 'C1', 'I3',
    'M'; None keeps the full 11-line default. Averaging over the *full*
    11-line set is invariant to line order, so no cue flip would be needed
    there -- but any narrower subset (e.g. only midline) is not, since
    "contra"/"ipsi" are cue-relative. Cue flip is applied unconditionally so
    the default (full set) is unaffected while subsets are handled correctly.

    ratio is plotted as plain log10(fw/bw) ("log-ratio"), not the 10*log10
    dB form used for fw/bw. ylim=None uses DEFAULT_YLIMS per measure (fixed
    +-0.02 for ratio, auto-scaled for fw/bw); pass an explicit ylim to apply
    the same limits to every requested measure instead.
    """
    line_idx = resolve_line_idx(line_idx, range(11))
    line_tag = '-'.join(RELATIVE_NAMES[i] for i in line_idx)

    if freq_bands is None:
        freq_bands = [
            (2, 6, 'Theta (2-6 Hz)'),
            (8, 12, 'Alpha (8-12 Hz)'),
            (14, 30, 'Beta (14-30 Hz)'),
        ]

    d0 = next(iter(by_subj[subj_ids[0]].values()))
    t = d0['epoch_tmin'] + (d0['starts'] + WINDOW_SIZE // 2) / d0['sfreq']
    ff = d0['ff']
    t_mask = (t >= t_start) & (t <= t_stop)
    t_sel = t[t_mask]

    fw_list, bw_list, ratio_list = [], [], []
    for subj in subj_ids:
        sess_fw, sess_bw, sess_ratio = [], [], []
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            fw_db     = 10 * np.log10(d['fwMax'] / d['fwMaxSS'])
            bw_db     = 10 * np.log10(d['bwMax'] / d['bwMaxSS'])
            ratio_log = np.log10(d['fwMax'] / d['bwMax'])
            t_s  = d['epoch_tmin'] + (d['starts'] + WINDOW_SIZE // 2) / d['sfreq']
            bl_s = (t_s >= bl0) & (t_s <= bl1)
            if baseline_correct:
                fw_db     -= fw_db    [:, :, bl_s, :].mean(axis=2, keepdims=True)
                bw_db     -= bw_db    [:, :, bl_s, :].mean(axis=2, keepdims=True)
                ratio_log -= ratio_log[:, :, bl_s, :].mean(axis=2, keepdims=True)
            mask = _trial_mask(d, trial_filter, exclude_bad=exclude_bad)
            cue_masked = d['cue_loc'][mask]
            flip = cue_masked == CUE_LEFT
            fw_aligned    = fw_db    [..., mask]
            bw_aligned    = bw_db    [..., mask]
            ratio_aligned = ratio_log[..., mask]
            fw_aligned   [..., flip] = fw_aligned   [::-1, ..., flip]
            bw_aligned   [..., flip] = bw_aligned   [::-1, ..., flip]
            ratio_aligned[..., flip] = ratio_aligned[::-1, ..., flip]
            sess_fw.append(fw_aligned   [line_idx].mean(axis=(0, -1)))
            sess_bw.append(bw_aligned   [line_idx].mean(axis=(0, -1)))
            sess_ratio.append(ratio_aligned[line_idx].mean(axis=(0, -1)))
        fw_list.append(np.mean(sess_fw, axis=0))
        bw_list.append(np.mean(sess_bw, axis=0))
        ratio_list.append(np.mean(sess_ratio, axis=0))

    fw_arr, bw_arr, ratio_arr = np.array(fw_list), np.array(bw_list), np.array(ratio_list)

    df = len(subj_ids) - 1
    threshold = stats.t.ppf(1 - alpha_level / 2, df=df)

    measure_keys = [measure] if isinstance(measure, str) else list(measure)
    all_configs = [
        ('fw', fw_arr, BASE_COLOR_FW, f'FW {len(line_idx)}-line avg (dB)', f'FW {len(line_idx)}-line avg (dB)'),
        ('bw', bw_arr, BASE_COLOR_BW, f'BW {len(line_idx)}-line avg (dB)', f'BW {len(line_idx)}-line avg (dB)'),
        ('ratio', ratio_arr, BASE_COLOR_RATIO, 'log10(FW/BW) - log-ratio', 'log-ratio'),
    ]
    measure_configs = [c for c in all_configs if c[0] in measure_keys]

    save_root = SAVE_ROOT if save_root is None else save_root
    save_dir = os.path.join(save_root, 'bl' if baseline_correct else 'no-bl')
    os.makedirs(save_dir, exist_ok=True)
    n_cols = len(freq_bands)

    for mname, arr, color, fig_title, ylabel in measure_configs:
        measure_ylim = ylim if ylim is not None else DEFAULT_YLIMS.get(mname)

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2), sharex=True, sharey=True, squeeze=False)
        axes = axes[0, :]

        for col, (fmin, fmax, fname) in enumerate(freq_bands):
            ax = axes[col]
            f_mask = (ff >= fmin) & (ff <= fmax)
            band = arr[:, f_mask, :][:, :, t_mask].mean(axis=1)
            m = band.mean(axis=0)
            ci = stats.t.ppf(0.975, df=df) * band.std(axis=0, ddof=1) / np.sqrt(len(subj_ids))
            ax.plot(t_sel, m, color=color, linewidth=2.5)
            ax.fill_between(t_sel, m - ci, m + ci, color=color, alpha=0.15, linewidth=0)

            if measure_ylim is not None:
                ax.set_ylim(*measure_ylim)
            ymin_ax, ymax_ax = ax.get_ylim()
            bar_y = ymax_ax - 0.05 * (ymax_ax - ymin_ax)

            try:
                _, cls, ps, _ = permutation_cluster_1samp_test(
                    band, threshold=threshold, n_permutations=5000,
                    tail=0, n_jobs=1, seed=42, verbose=False)
                _print_clusters_1d(cls, ps, t_sel, f'{mname} {fname}')
                for cl, p in zip(cls, ps):
                    if p < alpha_level:
                        sig_mask = np.zeros(len(t_sel), dtype=bool)
                        sig_mask[cl[0]] = True
                        draw_bar(ax, sig_mask, bar_y, color, t_sel)
            except Exception:
                pass

            ax.axhline(0, color='0.55', lw=0.8)
            _decorate_events(ax, show_labels=True, label_fontsize=8, label_rotation=32, label_y=-0.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=11, labelleft=(col == 0))
            ax.set_title(fname, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        out = os.path.join(save_dir, f'{mname}_{line_tag}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"[saved] {out}")
        plt.show()


# ── Per-band CONTRA-IPSI time series ─────────────────────────────────────────────

def plot_band_timeseries_cue(
    by_subj, subj_ids,
    baseline_correct=False,
    bl0=-0.75, bl1=-0.25,
    t_start=0, t_stop=5.25,
    freq_bands=None,
    ylim=None,
    measure=('fw', 'bw', 'ratio'),
    trial_filter=None,
    exclude_bad=True,
    contra_idx=None,
    ipsi_idx=None,
    title=None,
    split_measures=True,
    save_root=None,
):
    """Per-frequency-band CONTRA-IPSI power time series (cue-aligned), with
    cluster-based significance across subjects. Analogous to
    micheal_plot_funcs.plot_band_timeseries(mode='cue'), minus the RT-median
    markers and bar chart (out of scope here).

    contra_idx/ipsi_idx pick which of the 11 (post cue-flip) lines go into
    each group -- pass ints (0..10) or RELATIVE_NAMES strings like 'C1', 'I3';
    None keeps the full default 5-line CONTRA_IDX/IPSI_IDX split.
    """
    contra_idx = resolve_line_idx(contra_idx, CONTRA_IDX)
    ipsi_idx   = resolve_line_idx(ipsi_idx, IPSI_IDX)

    if freq_bands is None:
        freq_bands = [
            (2, 6, 'Theta (2-6 Hz)'),
            (8, 12, 'Alpha (8-12 Hz)'),
            (14, 30, 'Beta (14-30 Hz)'),
        ]

    d0 = next(iter(by_subj[subj_ids[0]].values()))
    t = d0['epoch_tmin'] + (d0['starts'] + WINDOW_SIZE // 2) / d0['sfreq']
    ff = d0['ff']
    t_mask = (t >= t_start) & (t <= t_stop)
    t_sel = t[t_mask]

    diff_lists = {'fw': [], 'bw': [], 'ratio': []}
    for subj in subj_ids:
        sess_diff = {'fw': [], 'bw': [], 'ratio': []}
        for sess in sorted(by_subj[subj]):
            d = by_subj[subj][sess]
            fw_db     = 10 * np.log10(d['fwMax'] / d['fwMaxSS'])
            bw_db     = 10 * np.log10(d['bwMax'] / d['bwMaxSS'])
            ratio_log = np.log10(d['fwMax'] / d['bwMax'])
            t_s  = d['epoch_tmin'] + (d['starts'] + WINDOW_SIZE // 2) / d['sfreq']
            bl_s = (t_s >= bl0) & (t_s <= bl1)
            if baseline_correct:
                fw_db     -= fw_db    [:, :, bl_s, :].mean(axis=2, keepdims=True)
                bw_db     -= bw_db    [:, :, bl_s, :].mean(axis=2, keepdims=True)
                ratio_log -= ratio_log[:, :, bl_s, :].mean(axis=2, keepdims=True)
            mask = _trial_mask(d, trial_filter, exclude_bad=exclude_bad)
            cue_masked = d['cue_loc'][mask]
            flip = cue_masked == CUE_LEFT
            for key, arr in (('fw', fw_db), ('bw', bw_db), ('ratio', ratio_log)):
                aligned = arr[..., mask]
                aligned[..., flip] = aligned[::-1, ..., flip]
                contra_mean = aligned[contra_idx].mean(axis=(0, -1))
                ipsi_mean   = aligned[ipsi_idx].mean(axis=(0, -1))
                sess_diff[key].append(contra_mean - ipsi_mean)
        for key in ('fw', 'bw', 'ratio'):
            diff_lists[key].append(np.mean(sess_diff[key], axis=0))

    diff_arrs = {key: np.array(vals) for key, vals in diff_lists.items()}

    df = len(subj_ids) - 1
    threshold = stats.t.ppf(1 - alpha_level / 2, df=df)

    measure_keys = [measure] if isinstance(measure, str) else list(measure)
    colors = {'fw': BASE_COLOR_FW, 'bw': BASE_COLOR_BW, 'ratio': BASE_COLOR_RATIO}
    labels = {'fw': 'FW contra-ipsi (dB)', 'bw': 'BW contra-ipsi (dB)', 'ratio': 'log ratio'}

    def draw_band(ax, band_data, color, label, key, show_labels=False):
        m = band_data.mean(axis=0)
        ci = stats.t.ppf(0.975, df=df) * band_data.std(axis=0, ddof=1) / np.sqrt(len(subj_ids))
        ax.plot(t_sel, m, color=color, linewidth=2.5)
        ax.fill_between(t_sel, m - ci, m + ci, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color='0.55', lw=0.8, linestyle='--', alpha=0.65)
        ax.set_ylim(*DEFAULT_YLIMS[key])

        ymin_ax, ymax_ax = ax.get_ylim()
        bar_y = ymax_ax - 0.05 * (ymax_ax - ymin_ax)
        try:
            _, cls, ps, _ = permutation_cluster_1samp_test(
                band_data, threshold=threshold, n_permutations=5000,
                tail=0, n_jobs=1, seed=42, verbose=False)
            _print_clusters_1d(cls, ps, t_sel, label, alpha=0.1)
            for cl, p in zip(cls, ps):
                if p < alpha_level:
                    sig_mask = np.zeros(len(t_sel), dtype=bool)
                    sig_mask[cl[0]] = True
                    draw_bar(ax, sig_mask, bar_y, color, t_sel)
        except Exception:
            pass

        _decorate_events(ax, show_labels=show_labels, label_fontsize=8, label_rotation=32, label_y=-0.2)
        ax.spines[['top', 'right']].set_visible(False)

    save_root = SAVE_ROOT if save_root is None else save_root
    save_dir = os.path.join(save_root, 'bl' if baseline_correct else 'no-bl')
    os.makedirs(save_dir, exist_ok=True)
    n_cols = len(freq_bands)

    def make_figure(keys, fname_suffix):
        n_rows = len(keys)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows),
                                 sharex=True, sharey='row', squeeze=False)
        for row, key in enumerate(keys):
            arr = diff_arrs[key]
            for col, (fmin, fmax, fname) in enumerate(freq_bands):
                ax = axes[row, col]
                f_mask = (ff >= fmin) & (ff <= fmax)
                band_data = arr[:, f_mask, :][:, :, t_mask].mean(axis=1)
                draw_band(ax, band_data, colors[key], f'{key} {fname}', key,
                          show_labels=(row == n_rows - 1))
                ax.tick_params(labelsize=11, labelbottom=(row == n_rows - 1))
                if row == 0:
                    ax.set_title(fname, fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(labels[key], fontsize=11)
                if row == n_rows - 1:
                    ax.set_xlabel('Time (s)', fontsize=11)
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        out = os.path.join(save_dir, f'{fname_suffix}_cue_band.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"[saved] {out}")
        plt.show()

    if split_measures:
        for key in measure_keys:
            make_figure([key], key)
    else:
        make_figure(measure_keys, '_'.join(measure_keys))
