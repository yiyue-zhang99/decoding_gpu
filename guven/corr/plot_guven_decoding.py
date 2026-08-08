#!/usr/bin/env python3
"""Plot cue/uncue item decoding time courses from PyTorch .npz results.

Adapted from micheal/plot_michael_decoding.py: 'cue'/'uncue' item-orientation
decoding instead of 'early'/'late', x-axis in seconds (guven's epoch spans
several seconds, not a few hundred ms), and guven's trigger event markers
(target/cue/impulse 1/rotation/impulse 2/probe) instead of a single t=0
event. Saves PNG only (no PDF).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

PROJECT_DIR = Path("/home/dilay/project2/tw")

# Trigger timeline relative to target onset (t=0), from
# fft/guven/guven_plot_funcs.py / fft/guven/preprocess.py.
EVENT_TIMES = [0, 0.75, 1.85, 2.45, 3.55, 4.15]
EVENT_LABELS = ["target", "cue", "impulse 1", "rotation", "impulse 2", "probe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "guven_alpha_decoding_discrete_fir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <input-dir>/decoding_accuracy.png",
    )
    parser.add_argument(
        "--pattern",
        default="subject_*_alpha_torch.npz",
        help="Glob pattern for subject result files inside input-dir",
    )
    parser.add_argument("--title", default="Guven cue/uncue item decoding")
    parser.add_argument("--xmin", type=float, default=-1.15)
    parser.add_argument("--xmax", type=float, default=5.45)
    parser.add_argument("--smooth-points", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--cluster-p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.stack(
        [np.convolve(row, kernel, mode="same") for row in values], axis=0
    )


def contiguous_clusters(mask: np.ndarray) -> list[np.ndarray]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def cluster_sign_flip_test(
    data: np.ndarray,
    permutations: int,
    cluster_alpha: float,
    seed: int,
    two_sided: bool = False,
) -> list[dict]:
    """Cluster-mass sign-flip test of the group mean against zero."""
    n_subjects = data.shape[0]
    if n_subjects < 2:
        return []
    threshold = student_t.ppf(
        1 - cluster_alpha / 2 if two_sided else 1 - cluster_alpha,
        n_subjects - 1,
    )
    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    observed_t = np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)
    if two_sided:
        observed_clusters = (
            contiguous_clusters(observed_t > threshold)
            + contiguous_clusters(observed_t < -threshold)
        )
    else:
        observed_clusters = contiguous_clusters(observed_t > threshold)
    if not observed_clusters:
        return []

    rng = np.random.default_rng(seed)
    sum_squares = np.square(data).sum(axis=0)
    max_masses = np.zeros(permutations)
    chunk_size = 2000
    for first in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - first)
        signs = rng.choice((-1.0, 1.0), size=(count, n_subjects))
        signed_sum = signs @ data
        perm_mean = signed_sum / n_subjects
        variance = (
            sum_squares[None, :] - np.square(signed_sum) / n_subjects
        ) / (n_subjects - 1)
        perm_sem = np.sqrt(np.maximum(variance, 0) / n_subjects)
        perm_t = np.divide(
            perm_mean,
            perm_sem,
            out=np.zeros_like(perm_mean),
            where=perm_sem > 0,
        )
        for row in range(count):
            if two_sided:
                clusters = (
                    contiguous_clusters(perm_t[row] > threshold)
                    + contiguous_clusters(perm_t[row] < -threshold)
                )
            else:
                clusters = contiguous_clusters(perm_t[row] > threshold)
            if clusters:
                max_masses[first + row] = max(
                    np.abs(perm_t[row, cluster]).sum() for cluster in clusters
                )

    results = []
    for cluster in observed_clusters:
        mass = np.abs(observed_t[cluster]).sum()
        p_value = (1 + np.count_nonzero(max_masses >= mass)) / (
            permutations + 1
        )
        results.append(
            {
                "indices": cluster,
                "mass": mass,
                "p": p_value,
                "sign": int(np.sign(observed_t[cluster].mean())),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    output = args.output or (args.input_dir / "decoding_accuracy.png")
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No subject result files found in {args.input_dir}")

    cue = []
    uncue = []
    reference_time = None
    for path in files:
        with np.load(path, allow_pickle=True) as result:
            time = np.asarray(result["time_dec"], dtype=float)
            if reference_time is None:
                reference_time = time
            elif not np.allclose(time, reference_time):
                raise ValueError(f"Time vector in {path.name} does not match")
            cue.append(np.asarray(result["dec_cue"], dtype=float))
            uncue.append(np.asarray(result["dec_uncue"], dtype=float))

    time_s = reference_time
    cue = moving_average(np.stack(cue), args.smooth_points)
    uncue = moving_average(np.stack(uncue), args.smooth_points)
    analysis_mask = (time_s >= args.xmin) & (time_s <= args.xmax)
    if not np.any(analysis_mask):
        raise ValueError("No decoding samples fall inside xmin/xmax")
    # Crop before permutation testing so the displayed and statistically
    # tested time intervals are exactly the same.
    time_s = time_s[analysis_mask]
    cue = cue[:, analysis_mask]
    uncue = uncue[:, analysis_mask]
    cue_mean = cue.mean(axis=0)
    uncue_mean = uncue.mean(axis=0)
    difference = cue - uncue
    difference_mean = difference.mean(axis=0)
    cue_clusters = cluster_sign_flip_test(
        cue, args.permutations, args.cluster_alpha, args.seed
    )
    uncue_clusters = cluster_sign_flip_test(
        uncue, args.permutations, args.cluster_alpha, args.seed + 1
    )
    difference_clusters = cluster_sign_flip_test(
        difference,
        args.permutations,
        args.cluster_alpha,
        args.seed + 2,
        two_sided=True,
    )

    blue = "#1428e8"
    red = "#f01d24"
    purple = "#7b2cbf"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    if len(files) > 1:
        cue_sem = cue.std(axis=0, ddof=1) / np.sqrt(len(files))
        uncue_sem = uncue.std(axis=0, ddof=1) / np.sqrt(len(files))
        ax.fill_between(
            time_s, cue_mean - cue_sem, cue_mean + cue_sem,
            color=blue, alpha=0.25, linewidth=0,
        )
        ax.fill_between(
            time_s, uncue_mean - uncue_sem, uncue_mean + uncue_sem,
            color=red, alpha=0.25, linewidth=0,
        )

    ax.plot(time_s, cue_mean, color=blue, linewidth=2.5, label="Tested cue item")
    ax.plot(time_s, uncue_mean, color=red, linewidth=2.5, label="Tested uncue item")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    for event_time, event_label in zip(EVENT_TIMES, EVENT_LABELS):
        ax.axvline(event_time, color="0.6", linestyle=":", linewidth=1)
        # Anchored near the bottom axis spine (not the top, which is already
        # crowded by the title/legend/cluster-significance bars).
        ax.text(
            event_time, 0.02, event_label, transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="0.4", rotation=90,
        )
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_xlabel("Time relative to target onset (s)")
    ax.set_ylabel("Decoding accuracy\ncosine-weighted Mahalanobis distance")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"{args.title} (N = {len(files)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

    plotted_values = np.r_[cue_mean, uncue_mean]
    y_min = plotted_values.min()
    y_max = plotted_values.max()
    y_range = max(y_max - y_min, np.finfo(float).eps)
    cue_bar_y = y_max + 0.16 * y_range
    uncue_bar_y = y_max + 0.10 * y_range
    difference_bar_y = y_max + 0.04 * y_range
    half_step = np.median(np.diff(time_s)) / 2
    for result in cue_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_s[indices[0]] - half_step, time_s[indices[-1]] + half_step],
                [cue_bar_y, cue_bar_y],
                color=blue, linewidth=5, solid_capstyle="butt",
            )
    for result in uncue_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_s[indices[0]] - half_step, time_s[indices[-1]] + half_step],
                [uncue_bar_y, uncue_bar_y],
                color=red, linewidth=5, solid_capstyle="butt",
            )
    for result in difference_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_s[indices[0]] - half_step, time_s[indices[-1]] + half_step],
                [difference_bar_y, difference_bar_y],
                color=purple, linewidth=5, solid_capstyle="butt",
            )
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.30 * y_range)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Loaded {len(files)} subject(s)")
    for label, clusters in (("cue", cue_clusters), ("uncue", uncue_clusters)):
        significant = [item for item in clusters if item["p"] < args.cluster_p]
        if not significant:
            print(f"{label}: no significant positive clusters")
        for item in significant:
            indices = item["indices"]
            print(
                f"{label}: {time_s[indices[0]]:.3f} to "
                f"{time_s[indices[-1]]:.3f} s, cluster p={item['p']:.6f}"
            )
    significant_difference = [
        item for item in difference_clusters if item["p"] < args.cluster_p
    ]
    if not significant_difference:
        print("cue - uncue: no significant two-sided clusters")
    for item in significant_difference:
        indices = item["indices"]
        direction = "cue > uncue" if item["sign"] > 0 else "uncue > cue"
        print(
            f"cue - uncue ({direction}): "
            f"{time_s[indices[0]]:.3f} to {time_s[indices[-1]]:.3f} s, "
            f"cluster p={item['p']:.6f}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
