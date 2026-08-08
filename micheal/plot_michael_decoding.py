#!/usr/bin/env python3
"""Plot early/late decoding time courses from PyTorch .npz results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


PROJECT_DIR = Path("/home/dilay/project2/tw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "michael_decoding_gpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results" / "michael_decoding_gpu"
        / "decoding_accuracy.png",
    )
    parser.add_argument(
        "--pattern",
        default="subject_*_voltage_torch.npz",
        help="Glob pattern for subject result files inside input-dir",
    )
    parser.add_argument("--title", default="Michael decoding")
    parser.add_argument("--xmin", type=float, default=-50)
    parser.add_argument("--xmax", type=float, default=500)
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
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No subject result files found in {args.input_dir}")

    early = []
    late = []
    reference_time = None
    for path in files:
        with np.load(path, allow_pickle=True) as result:
            time = np.asarray(result["time_dec"], dtype=float)
            if reference_time is None:
                reference_time = time
            elif not np.allclose(time, reference_time):
                raise ValueError(f"Time vector in {path.name} does not match")
            early.append(np.asarray(result["dec_early"], dtype=float))
            late.append(np.asarray(result["dec_late"], dtype=float))

    time_ms = reference_time * 1000
    early = moving_average(np.stack(early), args.smooth_points)
    late = moving_average(np.stack(late), args.smooth_points)
    analysis_mask = (time_ms >= args.xmin) & (time_ms <= args.xmax)
    if not np.any(analysis_mask):
        raise ValueError("No decoding samples fall inside xmin/xmax")
    # Crop before permutation testing so displayed and tested intervals match.
    time_ms = time_ms[analysis_mask]
    early = early[:, analysis_mask]
    late = late[:, analysis_mask]
    early_mean = early.mean(axis=0)
    late_mean = late.mean(axis=0)
    early_clusters = cluster_sign_flip_test(
        early, args.permutations, args.cluster_alpha, args.seed
    )
    late_clusters = cluster_sign_flip_test(
        late, args.permutations, args.cluster_alpha, args.seed + 1
    )
    difference_clusters = cluster_sign_flip_test(
        early - late,
        args.permutations,
        args.cluster_alpha,
        args.seed + 2,
        two_sided=True,
    )

    blue = "#1428e8"
    red = "#f01d24"
    purple = "#7b2cbf"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if len(files) > 1:
        early_sem = early.std(axis=0, ddof=1) / np.sqrt(len(files))
        late_sem = late.std(axis=0, ddof=1) / np.sqrt(len(files))
        ax.fill_between(
            time_ms,
            early_mean - early_sem,
            early_mean + early_sem,
            color=blue,
            alpha=0.25,
            linewidth=0,
        )
        ax.fill_between(
            time_ms,
            late_mean - late_sem,
            late_mean + late_sem,
            color=red,
            alpha=0.25,
            linewidth=0,
        )

    ax.plot(time_ms, early_mean, color=blue, linewidth=2.5, label="Tested early")
    ax.plot(time_ms, late_mean, color=red, linewidth=2.5, label="Tested late")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="0.45", linestyle=":", linewidth=1)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_xlabel("Time relative to event (ms)")
    ax.set_ylabel("Decoding accuracy\ncosine-weighted Mahalanobis distance")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"{args.title} (N = {len(files)})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

    plotted_values = np.r_[early_mean, late_mean]
    y_min = plotted_values.min()
    y_max = plotted_values.max()
    y_range = max(y_max - y_min, np.finfo(float).eps)
    early_bar_y = y_max + 0.16 * y_range
    late_bar_y = y_max + 0.10 * y_range
    difference_bar_y = y_max + 0.04 * y_range
    half_step = np.median(np.diff(time_ms)) / 2
    for result in early_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_ms[indices[0]] - half_step, time_ms[indices[-1]] + half_step],
                [early_bar_y, early_bar_y],
                color=blue,
                linewidth=5,
                solid_capstyle="butt",
            )
    for result in late_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_ms[indices[0]] - half_step, time_ms[indices[-1]] + half_step],
                [late_bar_y, late_bar_y],
                color=red,
                linewidth=5,
                solid_capstyle="butt",
            )
    difference_label_added = False
    for result in difference_clusters:
        if result["p"] < args.cluster_p:
            indices = result["indices"]
            ax.plot(
                [time_ms[indices[0]] - half_step, time_ms[indices[-1]] + half_step],
                [difference_bar_y, difference_bar_y],
                color=purple,
                linewidth=5,
                solid_capstyle="butt",
                label=(
                    "Early − late ($p<.05$)"
                    if not difference_label_added
                    else None
                ),
            )
            difference_label_added = True
    if difference_label_added:
        ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.25 * y_range)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Loaded {len(files)} subject(s)")
    for label, clusters in (("early", early_clusters), ("late", late_clusters)):
        significant = [item for item in clusters if item["p"] < args.cluster_p]
        if not significant:
            print(f"{label}: no significant positive clusters")
        for item in significant:
            indices = item["indices"]
            print(
                f"{label}: {time_ms[indices[0]]:.0f} to "
                f"{time_ms[indices[-1]]:.0f} ms, cluster p={item['p']:.6f}"
            )
    significant_difference = [
        item for item in difference_clusters if item["p"] < args.cluster_p
    ]
    if not significant_difference:
        print("early - late: no significant two-sided clusters")
    for item in significant_difference:
        indices = item["indices"]
        direction = "early > late" if item["sign"] > 0 else "late > early"
        print(
            f"early - late ({direction}): "
            f"{time_ms[indices[0]]:.0f} to {time_ms[indices[-1]]:.0f} ms, "
            f"cluster p={item['p']:.6f}"
        )
    print(f"Saved {args.output}")
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
