"""Matched-trial TW x time-cross decoding analysis in three dimensions.

The first level correlates trials within each session at every
TW-time x train-time x test-time cell. Session correlations are Fisher-z
transformed and averaged so each subject contributes exactly one 3-D map.
The group test uses one sign per subject and 6-neighbour 3-D cluster-mass
correction.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.stats import t as student_t

PLOT_STYLE_VERSION = "shared-origin-v2"


EVENT_TIMES = (0.0, 1.2, 1.8, 3.8, 4.3)


def _subject_number(path: Path) -> int:
    match = re.search(r"subject_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject number from {path.name}")
    return int(match.group(1))


def _tw_time(data: dict) -> np.ndarray:
    starts = np.asarray(data.get("starts", data["time"]))
    return -1.25 + (starts + 125) / 500


def _tw_measure(data: dict, measure: str) -> np.ndarray:
    measure = str(measure).strip().lower()
    if measure == "fw":
        return 10 * np.log10(data["fwmax"] / data["fwssmax"])
    if measure == "bw":
        return 10 * np.log10(data["bwmax"] / data["bwssmax"])
    raise ValueError(
        f"measure must be 'fw' or 'bw', received {measure!r}"
    )


def _component_array(values: np.ndarray, cue: np.ndarray, component: str) -> np.ndarray:
    # Results column 3 is the test-order condition:
    # 1 = right item tested late, left tested early, 2 = left item tested late, right tested early.
    # contra and ipsi are based on the tested late item
    cue_left = (cue == 2)[None, None, :]
    physical_left = values[0:5].mean(axis=0)
    physical_right = values[6:11].mean(axis=0)
    contra = np.where(cue_left, physical_right, physical_left)
    ipsi = np.where(cue_left, physical_left, physical_right)
    components = {
        "difference": contra - ipsi,
        "contra": contra,
        "ipsi": ipsi,
        "midline": values[5],
        "all_lines": values.mean(axis=0),
    }
    if component not in components:
        raise ValueError(f"Unknown component {component!r}; choose {tuple(components)}")
    return components[component]


def _select_time(time: np.ndarray, limits, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("time stride must be >= 1")
    mask = np.ones(time.size, dtype=bool)
    if limits is not None:
        # Include nominal boundary samples despite floating-point
        # representations such as 1.7999999999999998 for 1.8 s.
        step = np.median(np.diff(time)) if time.size > 1 else 1.0
        tolerance = max(abs(step) * 1e-6, np.finfo(float).eps * 100)
        mask &= (time >= limits[0] - tolerance) & (
            time <= limits[1] + tolerance
        )
    indices = np.flatnonzero(mask)[::stride]
    if indices.size == 0:
        raise ValueError(f"No samples in requested limits {limits}")
    return indices


def _trial_correlation_3d(tw: np.ndarray, decoding: np.ndarray) -> np.ndarray:
    """Pearson r across trials for 2-D diagonal or 3-D time-cross output."""
    if (
        tw.ndim != 2
        or decoding.ndim not in (2, 3)
        or tw.shape[0] != decoding.shape[0]
    ):
        raise ValueError(
            "Expected matched trial x TW-time and either trial x decoding-time "
            "or trial x train-time x test-time"
        )
    n_trials = tw.shape[0]
    if n_trials < 4:
        raise ValueError("At least four matched trials are required")
    tw64 = np.asarray(tw, dtype=np.float64)
    decoding_flat = np.asarray(decoding, dtype=np.float64).reshape(n_trials, -1)
    tw64 -= tw64.mean(axis=0, keepdims=True)
    decoding_flat -= decoding_flat.mean(axis=0, keepdims=True)
    tw_norm = np.sqrt(np.square(tw64).sum(axis=0))
    decoding_norm = np.sqrt(np.square(decoding_flat).sum(axis=0))
    if np.any(tw_norm == 0) or np.any(decoding_norm == 0):
        raise ValueError("A TW or decoding cell has zero variance across trials")
    correlation = (tw64.T @ decoding_flat) / (
        tw_norm[:, None] * decoding_norm[None, :]
    )
    return correlation.reshape(tw.shape[1], *decoding.shape[1:]).astype(np.float32)


def build_subject_cross_maps(
    tw_dir: Path,
    decoding_dir: Path,
    measure="fw",
    component="difference",
    condition="early",
    fmin=8.0,
    fmax=12.0,
    tw_limits=(0.0, 5.7),
    train_limits=(0.0, 6.0),
    test_limits=(0.0, 6.0),
    tw_stride=1,
    decoding_stride=1,
    diagonal_only=False,
    symmetrize_time_cross=False,
):
    """Create one Fisher-z 3-D map per subject from matched held-out trials.

    Sessions are never pooled. Pearson r is computed within each session,
    transformed with arctanh, and the two session z maps are averaged. When
    ``symmetrize_time_cross`` is true, each trial matrix is replaced by
    ``(matrix + matrix.T) / 2`` before any condition contrast or correlation;
    no temporal or Gaussian smoothing is applied.
    """
    tw_dir, decoding_dir = Path(tw_dir), Path(decoding_dir)
    paths = sorted(decoding_dir.glob("subject_*_alpha_time_cross_torch.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No subject_*_alpha_time_cross_torch.npz files in {decoding_dir}"
        )
    if condition not in {"early", "late", "early_minus_late"}:
        raise ValueError("condition must be early, late, or early_minus_late")

    subjects, subject_maps, matched_counts = [], [], {}
    selected_tw_time = selected_train_time = selected_test_time = None
    epsilon = np.finfo(np.float32).eps

    for decoding_path in paths:
        subject = _subject_number(decoding_path)
        session_maps = []
        matched_counts[subject] = {}
        with np.load(decoding_path, allow_pickle=True) as decoded_file:
            decoding_time = np.asarray(decoded_file["time_dec"], dtype=float)
            train_indices = _select_time(decoding_time, train_limits, decoding_stride)
            test_indices = _select_time(decoding_time, test_limits, decoding_stride)
            this_train_time = decoding_time[train_indices]
            this_test_time = decoding_time[test_indices]
            if selected_train_time is None:
                selected_train_time, selected_test_time = this_train_time, this_test_time
            elif not (
                np.allclose(selected_train_time, this_train_time)
                and np.allclose(selected_test_time, this_test_time)
            ):
                raise ValueError(f"Decoding time mismatch in {decoding_path.name}")

            for session in (1, 2):
                trial_ids = np.asarray(
                    decoded_file[f"sess{session}_trial_ids"], dtype=int
                ).reshape(-1)
                early = np.asarray(
                    decoded_file[f"sess{session}_trial_dec_early"], dtype=np.float32
                )[:, train_indices][:, :, test_indices]
                late = np.asarray(
                    decoded_file[f"sess{session}_trial_dec_late"], dtype=np.float32
                )[:, train_indices][:, :, test_indices]
                if symmetrize_time_cross:
                    if early.shape[1] != early.shape[2] or not np.allclose(
                        this_train_time, this_test_time
                    ):
                        raise ValueError(
                            "symmetrize_time_cross requires identical training "
                            "and test time axes"
                        )
                    early = (early + early.transpose(0, 2, 1)) * 0.5
                    late = (late + late.transpose(0, 2, 1)) * 0.5
                evidence = (
                    early if condition == "early" else late
                    if condition == "late" else early - late
                )
                if diagonal_only:
                    if evidence.shape[1] != evidence.shape[2] or not np.allclose(
                        this_train_time, this_test_time
                    ):
                        raise ValueError(
                            "diagonal_only requires identical training and test times"
                        )
                    # trial x train x test -> trial x matched train=test time
                    evidence = np.diagonal(
                        evidence, axis1=1, axis2=2
                    ).copy()
                if evidence.shape[0] != trial_ids.size:
                    raise ValueError(
                        f"Trial ID/evidence mismatch: subject {subject}, session {session}"
                    )

                tw_path = tw_dir / f"subj{subject:02d}_sess{session}.pkl"
                with tw_path.open("rb") as stream:
                    tw_data = pickle.load(stream)
                n_tw_trials = np.asarray(tw_data["cue_loc"]).size
                if np.any(trial_ids < 1) or np.any(trial_ids > n_tw_trials):
                    raise ValueError(f"Trial IDs exceed TW trial axis in {tw_path.name}")
                tw_rows = trial_ids - 1
                common = np.ones(trial_ids.size, dtype=bool)
                if "is_bad_epoch" in tw_data:
                    common &= ~np.asarray(tw_data["is_bad_epoch"], dtype=bool)[tw_rows]
                if "has_timing_issue" in tw_data:
                    common &= ~np.asarray(
                        tw_data["has_timing_issue"], dtype=bool
                    )[tw_rows]
                tw_rows = tw_rows[common]
                evidence = evidence[common]
                matched_counts[subject][session] = int(common.sum())

                tw_time = _tw_time(tw_data)
                tw_indices = _select_time(tw_time, tw_limits, tw_stride)
                this_tw_time = tw_time[tw_indices]
                if selected_tw_time is None:
                    selected_tw_time = this_tw_time
                elif not np.allclose(selected_tw_time, this_tw_time):
                    raise ValueError(f"TW time mismatch in {tw_path.name}")
                frequencies = np.asarray(tw_data["ff"])
                frequency_mask = (frequencies >= fmin) & (frequencies <= fmax)
                if not frequency_mask.any():
                    raise ValueError(f"No TW frequencies in {fmin:g}-{fmax:g} Hz")
                values = _tw_measure(tw_data, measure)
                component_values = _component_array(
                    values, np.asarray(tw_data["cue_loc"]), component
                )
                # frequency x TW-time x original-trial -> matched trial x TW-time
                trial_tw = component_values[frequency_mask][:, tw_indices, :][
                    :, :, tw_rows
                ].mean(axis=0).T
                correlation = _trial_correlation_3d(trial_tw, evidence)
                session_maps.append(
                    np.arctanh(np.clip(correlation, -1 + epsilon, 1 - epsilon))
                )
                del tw_data, values, component_values, trial_tw, evidence, early, late

        if len(session_maps) != 2:
            raise ValueError(f"Expected two sessions for subject {subject}")
        subject_maps.append(np.mean(session_maps, axis=0, dtype=np.float32))
        subjects.append(subject)
        print(
            f"subject {subject:02d}: matched s1={matched_counts[subject][1]}, "
            f"s2={matched_counts[subject][2]}; map={subject_maps[-1].shape}",
            flush=True,
        )

    return (
        np.asarray(subjects),
        selected_tw_time,
        selected_train_time,
        selected_test_time,
        np.stack(subject_maps),
        matched_counts,
    )


def cluster_signflip_2d(
    subject_z: np.ndarray,
    permutations=1000,
    cluster_alpha=0.05,
    cluster_p=0.05,
    seed=42,
    return_all=False,
):
    """Two-sided 2-D cluster-mass sign-flip test with 4-neighbour adjacency."""
    if subject_z.ndim != 3:
        raise ValueError("subject_z must be subject x TW-time x decoding-time")
    mean_z, observed_t, clusters, null_max = cluster_signflip_3d(
        subject_z[..., None],
        permutations=permutations,
        cluster_alpha=cluster_alpha,
        cluster_p=cluster_p,
        seed=seed,
        return_all=return_all,
    )
    for cluster in clusters:
        cluster["mask"] = cluster["mask"][..., 0]
    return mean_z[..., 0], observed_t[..., 0], clusters, null_max


def print_significant_cluster_ranges_2d(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    decoding_time: np.ndarray,
) -> None:
    """Print corrected TW x diagonal-decoding cluster extents and peaks."""
    significant = [cluster for cluster in clusters if cluster["significant"]]
    if not significant:
        print("  significant 2-D cluster ranges: none", flush=True)
        return
    print("  significant 2-D cluster ranges:", flush=True)
    for index, cluster in enumerate(significant, start=1):
        peak = np.unravel_index(
            np.argmax(np.where(cluster["mask"], np.abs(observed_t), -np.inf)),
            observed_t.shape,
        )
        direction = "positive" if cluster["sign"] > 0 else "negative"
        print(
            f"    cluster {index}: {direction}; corrected p={cluster['p']:.4f}; "
            f"mass={cluster['mass']:.1f}; cells={cluster['n_cells']}",
            flush=True,
        )
        print(
            f"      TW {tw_time[cluster['tw_min']]:.3f}–"
            f"{tw_time[cluster['tw_max']]:.3f} s; diagonal decoding "
            f"{decoding_time[cluster['train_min']]:.3f}–"
            f"{decoding_time[cluster['train_max']]:.3f} s",
            flush=True,
        )
        print(
            f"      peak t={observed_t[peak]:.3f} at TW={tw_time[peak[0]]:.3f} s, "
            f"diagonal decoding={decoding_time[peak[1]]:.3f} s",
            flush=True,
        )


def plot_diagonal_tw_correlation_2d(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    decoding_time: np.ndarray,
    output: Path,
    title: str,
):
    """Plot TW-time x diagonal-decoding-time group t map and contours."""
    significant_mask = np.zeros(observed_t.shape, dtype=bool)
    for cluster in clusters:
        if cluster["significant"]:
            significant_mask |= cluster["mask"]
    limit = float(np.percentile(np.abs(observed_t[np.isfinite(observed_t)]), 98))
    limit = max(limit, np.finfo(float).eps)
    fig, axis = plt.subplots(figsize=(7.4, 6.3), facecolor="white")
    image = axis.imshow(
        observed_t,
        origin="lower",
        aspect="auto",
        extent=[decoding_time[0], decoding_time[-1], tw_time[0], tw_time[-1]],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    if significant_mask.any():
        axis.contour(
            decoding_time,
            tw_time,
            significant_mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.8,
        )
    axis.set_xlabel("Diagonal decoding time: train = test (s)", fontweight="bold")
    axis.set_ylabel("TW time (s)", fontweight="bold")
    axis.set_title(title, fontsize=13, pad=12)
    axis.tick_params(direction="out")
    colourbar = fig.colorbar(image, ax=axis, pad=0.025)
    colourbar.set_label("Group one-sample t")
    fig.text(
        0.5,
        0.015,
        "Black contour: 2-D cluster-mass corrected p < .05",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    return fig


def _t_map_from_signs(
    subject_z: np.ndarray,
    signs: np.ndarray,
    sum_squares: np.ndarray | None = None,
) -> np.ndarray:
    """One-sample t map; sum of squares is invariant under sign flips."""
    n_subjects = subject_z.shape[0]
    signed_sum = np.tensordot(signs, subject_z, axes=(0, 0))
    mean = signed_sum / n_subjects
    if sum_squares is None:
        sum_squares = np.square(subject_z, dtype=np.float64).sum(axis=0)
    variance = (sum_squares - n_subjects * np.square(mean)) / (n_subjects - 1)
    sem = np.sqrt(np.maximum(variance, 0) / n_subjects)
    return np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)


def _signed_cluster_labels(t_map: np.ndarray, threshold: float, connectivity):
    """Label positive and negative clusters separately for a two-sided test."""
    positive, n_positive = label(t_map > threshold, connectivity)
    negative, n_negative = label(t_map < -threshold, connectivity)
    negative[negative > 0] += n_positive
    return positive + negative, n_positive + n_negative


def cluster_signflip_3d(
    subject_z: np.ndarray,
    permutations=5000,
    cluster_alpha=0.05,
    cluster_p=0.05,
    seed=42,
    return_all=False,
):
    """3-D two-sided cluster-mass test with one sign per subject/map."""
    subject_z = np.asarray(subject_z, dtype=np.float32)
    if subject_z.ndim != 4 or subject_z.shape[0] < 2:
        raise ValueError("subject_z must be subject x TW x train x test")
    n_subjects = subject_z.shape[0]
    threshold = student_t.ppf(1 - cluster_alpha / 2, n_subjects - 1)
    connectivity = generate_binary_structure(3, 1)  # six face neighbours
    sum_squares = np.square(subject_z, dtype=np.float64).sum(axis=0)
    observed_t = _t_map_from_signs(
        subject_z, np.ones(n_subjects), sum_squares
    )
    observed_labels, n_observed = _signed_cluster_labels(
        observed_t, threshold, connectivity
    )

    rng = np.random.default_rng(seed)
    null_max = np.zeros(permutations, dtype=np.float64)
    for permutation in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=n_subjects)
        permuted_t = _t_map_from_signs(subject_z, signs, sum_squares)
        permuted_labels, n_clusters = _signed_cluster_labels(
            permuted_t, threshold, connectivity
        )
        if n_clusters:
            masses = np.bincount(
                permuted_labels.ravel(),
                weights=np.abs(permuted_t).ravel(),
            )[1:]
            null_max[permutation] = masses.max(initial=0)
        if (permutation + 1) % max(1, permutations // 20) == 0:
            print(f"  permutations {permutation + 1}/{permutations}", flush=True)

    clusters = []
    for index in range(1, n_observed + 1):
        mask = observed_labels == index
        mass = float(np.abs(observed_t)[mask].sum())
        p_value = float((1 + np.count_nonzero(null_max >= mass)) / (permutations + 1))
        coordinates = np.where(mask)
        cluster = {
            "mask": mask,
            "p": p_value,
            "mass": mass,
            "sign": int(np.sign(observed_t[mask].mean())),
            "significant": p_value < cluster_p,
            "tw_min": int(coordinates[0].min()),
            "tw_max": int(coordinates[0].max()),
            "train_min": int(coordinates[1].min()),
            "train_max": int(coordinates[1].max()),
            "test_min": int(coordinates[2].min()),
            "test_max": int(coordinates[2].max()),
            "n_cells": int(mask.sum()),
        }
        if return_all or cluster["significant"]:
            clusters.append(cluster)
    return subject_z.mean(axis=0), observed_t, clusters, null_max


def print_significant_cluster_ranges(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
) -> None:
    """Print corrected cluster extents and peaks on all three time axes."""
    significant = [cluster for cluster in clusters if cluster["significant"]]
    if not significant:
        print("  significant 3-D cluster ranges: none", flush=True)
        return
    print("  significant 3-D cluster ranges:", flush=True)
    for index, cluster in enumerate(significant, start=1):
        masked_absolute = np.where(
            cluster["mask"], np.abs(observed_t), -np.inf
        )
        peak = np.unravel_index(np.argmax(masked_absolute), observed_t.shape)
        peak_t = float(observed_t[peak])
        direction = "positive" if cluster["sign"] > 0 else "negative"
        print(
            f"    cluster {index}: {direction}; corrected p={cluster['p']:.4f}; "
            f"mass={cluster['mass']:.1f}; voxels={cluster['n_cells']}",
            flush=True,
        )
        print(
            f"      TW       {tw_time[cluster['tw_min']]:.3f}–"
            f"{tw_time[cluster['tw_max']]:.3f} s",
            flush=True,
        )
        print(
            f"      Training {train_time[cluster['train_min']]:.3f}–"
            f"{train_time[cluster['train_max']]:.3f} s",
            flush=True,
        )
        print(
            f"      Test     {test_time[cluster['test_min']]:.3f}–"
            f"{test_time[cluster['test_max']]:.3f} s",
            flush=True,
        )
        print(
            f"      peak t={peak_t:.3f} at "
            f"TW={tw_time[peak[0]]:.3f}, "
            f"Training={train_time[peak[1]]:.3f}, "
            f"Test={test_time[peak[2]]:.3f} s",
            flush=True,
        )


def _max_abs_projection(values: np.ndarray, axis: int) -> np.ndarray:
    index = np.abs(values).argmax(axis=axis, keepdims=True)
    return np.take_along_axis(values, index, axis=axis).squeeze(axis)


def plot_significant_3d_projections(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    output: Path,
    title: str,
):
    """Plot max-|t| projections and contours of corrected 3-D clusters."""
    significant = [cluster for cluster in clusters if cluster["significant"]]
    combined = np.zeros(observed_t.shape, dtype=bool)
    for cluster in significant:
        combined |= cluster["mask"]
    projections = (
        (_max_abs_projection(observed_t, 2), combined.any(axis=2), train_time, tw_time,
         "Training time (s)", "TW time (s)", "max over test time"),
        (_max_abs_projection(observed_t, 1), combined.any(axis=1), test_time, tw_time,
         "Test time (s)", "TW time (s)", "max over training time"),
        (_max_abs_projection(observed_t, 0), combined.any(axis=0), test_time, train_time,
         "Test time (s)", "Training time (s)", "max over TW time"),
    )
    limit = float(np.nanmax(np.abs(observed_t)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    image = None
    for axis, (values, mask, x, y, xlabel, ylabel, subtitle) in zip(axes, projections):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        if mask.any():
            axis.contour(x, y, mask.astype(float), levels=[0.5], colors="black", linewidths=2)
        for event in EVENT_TIMES:
            if x[0] <= event <= x[-1]:
                axis.axvline(event, color=".45", ls=":", lw=.7)
            if y[0] <= event <= y[-1]:
                axis.axhline(event, color=".45", ls=":", lw=.7)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=subtitle)
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
    fig.suptitle(title)
    fig.subplots_adjust(left=.06, right=.91, bottom=.13, top=.84, wspace=.24)
    colorbar_axis = fig.add_axes([.93, .18, .012, .60])
    fig.colorbar(image, cax=colorbar_axis, label="Group one-sample t")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_3d_cube(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    threshold: float,
    output: Path,
    title: str,
):
    """Create a clean interactive cube using voxel isosurfaces."""
    import plotly.graph_objects as go
    from scipy.ndimage import zoom

    observed_t = np.asarray(observed_t)
    # Statistical maps and clusters stay at full resolution. Only the WebGL
    # display grid is reduced so the complete 3-D volume opens responsively.
    display_shape = tuple(min(size, 36) for size in observed_t.shape)
    zoom_factors = tuple(
        display / original
        for display, original in zip(display_shape, observed_t.shape)
    )
    display_t = zoom(
        observed_t, zoom_factors, order=1, mode="nearest", prefilter=False
    )
    display_tw_time = np.linspace(tw_time[0], tw_time[-1], display_t.shape[0])
    display_train_time = np.linspace(
        train_time[0], train_time[-1], display_t.shape[1]
    )
    display_test_time = np.linspace(
        test_time[0], test_time[-1], display_t.shape[2]
    )
    tw_grid, train_grid, test_grid = np.meshgrid(
        display_tw_time, display_train_time, display_test_time, indexing="ij"
    )
    x = test_grid.ravel()
    y = train_grid.ravel()
    z = tw_grid.ravel()
    # A few extreme voxels should not wash out the colour of the other 99%.
    # Clip only the display scale; the statistics and cluster test are unchanged.
    finite_absolute = np.abs(display_t[np.isfinite(display_t)])
    maximum = float(np.percentile(finite_absolute, 97.5))
    maximum = max(maximum, np.finfo(float).eps)
    figure = go.Figure()

    # Continuous volume rendering includes every voxel. Values near zero are
    # almost transparent so internal positive/negative structure remains
    # visible instead of being hidden by an opaque outer cube.
    figure.add_trace(
        go.Volume(
            x=x,
            y=y,
            z=z,
            value=display_t.ravel(),
            isomin=-maximum,
            isomax=maximum,
            surface_count=20,
            colorscale="RdBu_r",
            opacity=0.25,
            opacityscale=[
                [0.00, 0.62],
                [0.25, 0.30],
                [0.42, 0.08],
                [0.50, 0.006],
                [0.58, 0.08],
                [0.75, 0.30],
                [1.00, 0.62],
            ],
            caps={"x_show": False, "y_show": False, "z_show": False},
            colorbar={"title": "Group t", "len": 0.72},
            name="continuous group t volume",
            hovertemplate=(
                "test=%{x:.2f}s<br>train=%{y:.2f}s<br>"
                "TW=%{z:.2f}s<br>t=%{value:.3f}<extra></extra>"
            ),
        )
    )

    significant = [cluster for cluster in clusters if cluster["significant"]]

    # Render each corrected mask only inside its tight local crop. This keeps
    # the real full-resolution shape without sending a full-cube mask to WebGL.
    for index, cluster in enumerate(significant, start=1):
        locations = np.where(cluster["mask"])
        slices = []
        for axis, indices in enumerate(locations):
            start = max(0, int(indices.min()) - 1)
            stop = min(observed_t.shape[axis], int(indices.max()) + 2)
            slices.append(slice(start, stop))
        tw_slice, train_slice, test_slice = slices
        local_mask = cluster["mask"][tw_slice, train_slice, test_slice]
        local_tw, local_train, local_test = np.meshgrid(
            tw_time[tw_slice],
            train_time[train_slice],
            test_time[test_slice],
            indexing="ij",
        )
        colour = "#d73027" if cluster["sign"] > 0 else "#4575b4"
        figure.add_trace(
            go.Isosurface(
                x=local_test.ravel(),
                y=local_train.ravel(),
                z=local_tw.ravel(),
                value=local_mask.astype(float).ravel(),
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                colorscale=[[0.0, colour], [1.0, colour]],
                opacity=0.58,
                caps={"x_show": False, "y_show": False, "z_show": False},
                showscale=False,
                lighting={
                    "ambient": 0.72,
                    "diffuse": 0.72,
                    "roughness": 0.9,
                    "specular": 0.08,
                },
                name=(
                    f"cluster {index}, corrected p={cluster['p']:.3f}, "
                    f"cells={cluster['n_cells']}"
                ),
                hoverinfo="name",
            )
        )

    # Explicit shared test/train origin. Plotly otherwise places automatic
    # axes on different cube edges depending on the camera angle.
    origin_z = float(tw_time[0])
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, float(test_time[-1]), None, 0.0, 0.0],
            y=[0.0, 0.0, None, 0.0, float(train_time[-1])],
            z=[origin_z, origin_z, None, origin_z, origin_z],
            mode="lines",
            line={"color": "#334155", "width": 5},
            name="shared Test/Training origin",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[origin_z],
            mode="markers+text",
            marker={"size": 5, "color": "#111827"},
            text=["0"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    test_ticks = np.arange(1.0, float(test_time[-1]) + 0.01, 1.0)
    train_ticks = np.arange(1.0, float(train_time[-1]) + 0.01, 1.0)
    figure.add_trace(
        go.Scatter3d(
            x=test_ticks,
            y=np.zeros(test_ticks.size),
            z=np.full(test_ticks.size, origin_z),
            mode="text",
            text=[f"{value:g}" for value in test_ticks],
            textposition="top center",
            textfont={"size": 11, "color": "#334155"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=np.zeros(train_ticks.size),
            y=train_ticks,
            z=np.full(train_ticks.size, origin_z),
            mode="text",
            text=[f"{value:g}" for value in train_ticks],
            textposition="middle left",
            textfont={"size": 11, "color": "#334155"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[float(test_time[-1]), 0.0],
            y=[0.0, float(train_time[-1])],
            z=[origin_z, origin_z],
            mode="text",
            text=["Test time (s)", "Training time (s)"],
            textposition="bottom center",
            textfont={"size": 13, "color": "#1e293b"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # One labelled peak per corrected cluster replaces thousands of outlined
    # markers and gives exact inferential information on hover.
    for index, cluster in enumerate(significant, start=1):
        masked_t = np.where(cluster["mask"], np.abs(observed_t), -np.inf)
        peak = np.unravel_index(np.argmax(masked_t), observed_t.shape)
        peak_t = float(observed_t[peak])
        hover = (
            f"cluster {index}<br>corrected p={cluster['p']:.4f}"
            f"<br>mass={cluster['mass']:.1f}<br>cells={cluster['n_cells']}"
            f"<br>test={test_time[peak[2]]:.2f}s"
            f"<br>train={train_time[peak[1]]:.2f}s"
            f"<br>TW={tw_time[peak[0]]:.2f}s<br>peak t={peak_t:.3f}"
        )
        figure.add_trace(
            go.Scatter3d(
                x=[test_time[peak[2]]],
                y=[train_time[peak[1]]],
                z=[tw_time[peak[0]]],
                mode="markers",
                marker={
                    "size": 3.5,
                    "color": "#161616",
                    "line": {"color": "#FFFFFF", "width": 1},
                },
                name=f"cluster {index} peak (p={cluster['p']:.3f})",
                text=[hover],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if not significant:
        figure.add_annotation(
            text=(
                "No cluster-corrected p < .05 surface; "
                "the transparent colour field shows uncorrected group t."
            ),
            x=0.5,
            y=0.96,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#555"},
        )
    figure.update_layout(
        title=title,
        scene={
            "xaxis": {
                "title": "",
                "range": [0.0, float(test_time[-1])],
                "backgroundcolor": "rgba(248,247,245,0.30)",
                "gridcolor": "#E8E5E1",
                "zeroline": False,
                "showline": True,
                "linecolor": "#AAA5A0",
                "tick0": 0.0,
                "dtick": 0.5,
                "showticklabels": False,
            },
            "yaxis": {
                "title": "",
                "range": [0.0, float(train_time[-1])],
                "backgroundcolor": "rgba(248,247,245,0.30)",
                "gridcolor": "#E8E5E1",
                "zeroline": False,
                "showline": True,
                "linecolor": "#AAA5A0",
                "tick0": 0.0,
                "dtick": 0.5,
                "showticklabels": False,
            },
            "zaxis": {
                "title": "TW time (s)",
                "range": [tw_time[0], tw_time[-1]],
                "backgroundcolor": "rgba(248,247,245,0.30)",
                "gridcolor": "#E8E5E1",
                "zeroline": False,
                "showline": True,
                "linecolor": "#AAA5A0",
                "tick0": float(tw_time[0]),
                "dtick": 0.5,
            },
            "aspectmode": "cube",
            # Looking from the two negative horizontal directions puts the
            # shared test/train 1.8-s origin at the front corner.
            "camera": {
                "eye": {"x": -1.55, "y": -1.55, "z": 1.15},
                "up": {"x": 0, "y": 0, "z": 1},
            },
        },
        legend={"x": 0.01, "y": 0.99},
        template="plotly_white",
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        height=760,
        annotations=[
            {
                "text": (
                    f"Fast 3-D display: {display_t.shape}; "
                    f"statistics: full resolution {observed_t.shape}"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 0.01,
                "xanchor": "right",
                "showarrow": False,
                "font": {"size": 11, "color": "#64748b"},
            }
        ],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Store one shared plotly.min.js beside the HTML files instead of embedding
    # several megabytes of library code into every result.
    figure.write_html(output, include_plotlyjs="directory", full_html=True)
    return figure


def plot_3d_statistic_cluster_panels(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    threshold: float,
    output: Path,
    title: str,
):
    """Publication-style 3-D t-map and corrected-cluster voxel panels."""
    print(f"  plot style: {PLOT_STYLE_VERSION}", flush=True)
    observed_t = np.asarray(observed_t, dtype=float)
    significant = [cluster for cluster in clusters if cluster["significant"]]
    supra = np.isfinite(observed_t) & (np.abs(observed_t) >= threshold)
    coordinates = np.where(supra)

    # Keep the strongest display voxels only when a map is exceptionally
    # dense. This affects rendering only, never the statistics or clusters.
    max_stat_voxels = 50000
    if coordinates[0].size > max_stat_voxels:
        absolute = np.abs(observed_t[coordinates])
        selected = np.argpartition(absolute, -max_stat_voxels)[-max_stat_voxels:]
        coordinates = tuple(axis[selected] for axis in coordinates)
    stat_values = observed_t[coordinates]
    if stat_values.size:
        colour_limit = float(np.percentile(np.abs(stat_values), 98.0))
    else:
        colour_limit = float(threshold)
    colour_limit = max(colour_limit, float(threshold))

    fig = plt.figure(figsize=(15.5, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, left=0.045, right=0.865, bottom=0.10, top=0.86, wspace=0.12
    )
    axes = [
        fig.add_subplot(grid[0, 0], projection="3d"),
        fig.add_subplot(grid[0, 1], projection="3d"),
    ]

    image = axes[0].scatter(
        test_time[coordinates[2]],
        train_time[coordinates[1]],
        tw_time[coordinates[0]],
        c=stat_values,
        cmap="coolwarm",
        vmin=-colour_limit,
        vmax=colour_limit,
        marker="s",
        s=9,
        alpha=0.24,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )

    cluster_colours = {1: "#e53935", -1: "#2864dc"}
    for index, cluster in enumerate(significant, start=1):
        location = np.where(cluster["mask"])
        axes[1].scatter(
            test_time[location[2]],
            train_time[location[1]],
            tw_time[location[0]],
            color=cluster_colours[cluster["sign"]],
            marker="s",
            s=11,
            alpha=0.40,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            label=f"cluster {index}, p={cluster['p']:.3f}",
        )

    panel_titles = (
        rf"A   3D statistic map ($|t| \geq {threshold:.2f}$)",
        "B   Significant 3D cluster",
    )
    for axis, panel_title in zip(axes, panel_titles):
        axis.set_title(panel_title, fontsize=16, fontweight="bold", pad=18)
        axis.set_xlabel("Test time (s)", fontsize=12, fontweight="bold", labelpad=10)
        axis.set_ylabel("Training time (s)", fontsize=12, fontweight="bold", labelpad=10)
        axis.set_zlabel("TW time (s)", fontsize=12, fontweight="bold", labelpad=9)
        # With this camera Matplotlib places the Training axis at the x=max
        # edge. Reversing the displayed Test direction makes Test=0 and
        # Training=0 meet at the same front corner.
        axis.set_xlim(float(test_time[-1]), 0.0)
        axis.set_ylim(0.0, float(train_time[-1]))
        axis.set_zlim(float(tw_time[0]), float(tw_time[-1]))
        axis.view_init(elev=23, azim=-52)
        axis.set_box_aspect((1.0, 1.0, 0.92))
        axis.grid(True, alpha=0.24)
        for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_edgecolor((0.70, 0.70, 0.70, 0.65))
        for axis_info in (axis.xaxis, axis.yaxis, axis.zaxis):
            axis_info._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.34)
            axis_info._axinfo["grid"]["linewidth"] = 0.55
        axis.tick_params(labelsize=10, pad=1)

    if significant:
        axes[1].legend(
            loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False, fontsize=10
        )
    else:
        axes[1].text2D(
            0.5,
            0.52,
            "No cluster-corrected p < .05",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#555555",
        )

    # A dedicated far-right colourbar cannot overlap either panel's axis label.
    colour_axis = fig.add_axes([0.905, 0.285, 0.015, 0.36])
    colourbar = fig.colorbar(image, cax=colour_axis)
    colourbar.set_label("t-value", fontsize=12)
    fig.suptitle(title, fontsize=15, y=0.965)
    fig.text(
        0.5,
        0.025,
        "Panel A: cluster-forming voxels; Panel B: 3-D cluster-mass corrected p < .05",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    return fig


def plot_interactive_3d_slices(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    threshold: float,
    output: Path,
    title: str,
):
    """Interactive train x test heatmap with a slider over TW time.

    This displays the complete 3-D result without expensive WebGL volume
    rendering. Corrected clusters are outlined independently on every slice.
    """
    import plotly.graph_objects as go

    observed_t = np.asarray(observed_t, dtype=float)
    tw_time = np.asarray(tw_time, dtype=float)
    train_time = np.asarray(train_time, dtype=float)
    test_time = np.asarray(test_time, dtype=float)
    expected = (tw_time.size, train_time.size, test_time.size)
    if observed_t.shape != expected:
        raise ValueError(
            f"observed_t shape {observed_t.shape} does not match {expected}"
        )

    significant_mask = np.zeros(observed_t.shape, dtype=bool)
    significant = [cluster for cluster in clusters if cluster["significant"]]
    for cluster in significant:
        significant_mask |= cluster["mask"]

    finite_absolute = np.abs(observed_t[np.isfinite(observed_t)])
    color_limit = float(np.percentile(finite_absolute, 97.5))
    color_limit = max(color_limit, np.finfo(float).eps)

    def heatmap(index: int):
        return go.Heatmap(
            x=test_time,
            y=train_time,
            z=observed_t[index],
            zmin=-color_limit,
            zmax=color_limit,
            colorscale="RdBu_r",
            colorbar={"title": "Group t", "len": 0.82},
            hovertemplate=(
                "test=%{x:.2f}s<br>train=%{y:.2f}s<br>"
                "t=%{z:.3f}<extra></extra>"
            ),
        )

    def contour(index: int):
        return go.Contour(
            x=test_time,
            y=train_time,
            z=significant_mask[index].astype(float),
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1.0,
                "coloring": "none",
                "showlabels": False,
            },
            line={"color": "black", "width": 2.2},
            showscale=False,
            hoverinfo="skip",
            name="cluster-corrected p < .05",
        )

    frames = [
        go.Frame(
            data=[heatmap(index), contour(index)],
            name=str(index),
            layout={
                "title": {
                    "text": f"{title}<br><sup>TW time = {tw_time[index]:.2f} s</sup>"
                }
            },
        )
        for index in range(tw_time.size)
    ]
    figure = go.Figure(data=[heatmap(0), contour(0)], frames=frames)
    slider_steps = [
        {
            "method": "animate",
            "label": f"{value:.2f}",
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index, value in enumerate(tw_time)
    ]
    figure.update_layout(
        title=f"{title}<br><sup>TW time = {tw_time[0]:.2f} s</sup>",
        xaxis={
            "title": "Test time (s)",
            "range": [float(test_time[0]), float(test_time[-1])],
            "constrain": "domain",
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        yaxis={
            "title": "Training time (s)",
            "range": [float(train_time[0]), float(train_time[-1])],
            "scaleanchor": "x",
            "scaleratio": 1,
            "gridcolor": "#e5e7eb",
            "zeroline": False,
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "TW time: ", "suffix": " s"},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.16,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 180, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                            },
                        ],
                    },
                ],
            }
        ],
        annotations=[
            {
                "text": (
                    "Black outline: 3-D cluster-corrected p < .05"
                    if significant
                    else "No 3-D cluster-corrected p < .05"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 1.0,
                "y": -0.16,
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 12, "color": "#374151"},
            }
        ],
        template="plotly_white",
        width=850,
        height=850,
        margin={"l": 75, "r": 90, "b": 150, "t": 90},
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    return figure


def plot_interactive_3d_orthoslices(
    observed_t: np.ndarray,
    clusters: list[dict],
    tw_time: np.ndarray,
    train_time: np.ndarray,
    test_time: np.ndarray,
    threshold: float,
    output: Path,
    title: str,
):
    """Lightweight rotatable 3-D view with three synchronized slice planes."""
    import plotly.graph_objects as go

    observed_t = np.asarray(observed_t, dtype=float)
    tw_time = np.asarray(tw_time, dtype=float)
    train_time = np.asarray(train_time, dtype=float)
    test_time = np.asarray(test_time, dtype=float)
    finite_absolute = np.abs(observed_t[np.isfinite(observed_t)])
    color_limit = max(
        float(np.percentile(finite_absolute, 97.5)), np.finfo(float).eps
    )
    colorscale = "RdBu_r"

    test_xy, train_xy = np.meshgrid(test_time, train_time, indexing="xy")
    test_xz, tw_xz = np.meshgrid(test_time, tw_time, indexing="xy")
    train_yz, tw_yz = np.meshgrid(train_time, tw_time, indexing="xy")

    def plane_traces(tw_index: int):
        value = tw_time[tw_index]
        train_index = int(np.argmin(np.abs(train_time - value)))
        test_index = int(np.argmin(np.abs(test_time - value)))
        common = {
            "cmin": -color_limit,
            "cmax": color_limit,
            "colorscale": colorscale,
            "showscale": False,
            "opacity": 0.92,
            "hoverinfo": "skip",
            "lighting": {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0},
        }
        horizontal = go.Surface(
            x=test_xy,
            y=train_xy,
            z=np.full_like(test_xy, tw_time[tw_index]),
            surfacecolor=observed_t[tw_index],
            name=f"TW = {tw_time[tw_index]:.2f} s",
            colorbar={"title": "Group t", "len": 0.72},
            showscale=True,
            **{key: val for key, val in common.items() if key != "showscale"},
        )
        training = go.Surface(
            x=test_xz,
            y=np.full_like(test_xz, train_time[train_index]),
            z=tw_xz,
            surfacecolor=observed_t[:, train_index, :],
            name=f"Training = {train_time[train_index]:.2f} s",
            **common,
        )
        testing = go.Surface(
            x=np.full_like(train_yz, test_time[test_index]),
            y=train_yz,
            z=tw_yz,
            surfacecolor=observed_t[:, :, test_index],
            name=f"Test = {test_time[test_index]:.2f} s",
            **common,
        )
        return horizontal, training, testing

    initial = plane_traces(0)
    figure = go.Figure(data=list(initial))

    # Corrected clusters remain true 3-D surfaces, independent of the slices.
    tw_grid, train_grid, test_grid = np.meshgrid(
        tw_time, train_time, test_time, indexing="ij"
    )
    significant = [cluster for cluster in clusters if cluster["significant"]]
    for sign, colour, name in (
        (1, "#d73027", "corrected positive p < .05"),
        (-1, "#4575b4", "corrected negative p < .05"),
    ):
        mask = np.zeros(observed_t.shape, dtype=float)
        selected = [cluster for cluster in significant if cluster["sign"] == sign]
        for cluster in selected:
            mask[cluster["mask"]] = 1.0
        if selected:
            figure.add_trace(
                go.Isosurface(
                    x=test_grid.ravel(),
                    y=train_grid.ravel(),
                    z=tw_grid.ravel(),
                    value=mask.ravel(),
                    isomin=0.5,
                    isomax=1.0,
                    surface_count=1,
                    colorscale=[[0, colour], [1, colour]],
                    opacity=0.48,
                    caps={"x_show": False, "y_show": False, "z_show": False},
                    showscale=False,
                    name=name,
                    hoverinfo="skip",
                )
            )

    frames = []
    for index, value in enumerate(tw_time):
        frames.append(
            go.Frame(
                data=list(plane_traces(index)),
                traces=[0, 1, 2],
                name=str(index),
                layout={
                    "title": {
                        "text": f"{title}<br><sup>cross-section time = {value:.2f} s</sup>"
                    }
                },
            )
        )
    figure.frames = frames
    steps = [
        {
            "method": "animate",
            "label": f"{value:.2f}",
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index, value in enumerate(tw_time)
    ]
    axis_style = {
        "backgroundcolor": "rgba(248,250,252,0.25)",
        "gridcolor": "#e5e7eb",
        "zeroline": False,
        "showline": True,
        "linecolor": "#94a3b8",
    }
    figure.update_layout(
        title=f"{title}<br><sup>cross-section time = {tw_time[0]:.2f} s</sup>",
        scene={
            "xaxis": {
                **axis_style,
                "title": "Test time (s)",
                "range": [float(test_time[0]), float(test_time[-1])],
            },
            "yaxis": {
                **axis_style,
                "title": "Training time (s)",
                "range": [float(train_time[0]), float(train_time[-1])],
            },
            "zaxis": {
                **axis_style,
                "title": "TW time (s)",
                "range": [float(tw_time[0]), float(tw_time[-1])],
            },
            "aspectmode": "cube",
            "camera": {"eye": {"x": -1.5, "y": -1.5, "z": 1.2}},
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Cross-section time: ", "suffix": " s"},
                "pad": {"t": 35},
                "steps": steps,
            }
        ],
        legend={"x": 0.01, "y": 0.99},
        template="plotly_white",
        height=820,
        margin={"l": 10, "r": 20, "b": 100, "t": 80},
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    return figure


def save_analysis(
    output_prefix: Path,
    subjects,
    tw_time,
    train_time,
    test_time,
    subject_z,
    mean_z,
    observed_t,
    clusters,
    null_max,
    matched_counts,
):
    """Save numeric results, cluster masks/metadata, and a JSON report."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    significant = [cluster for cluster in clusters if cluster["significant"]]
    significant_mask = np.zeros(observed_t.shape, dtype=bool)
    for cluster in significant:
        significant_mask |= cluster["mask"]
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        subjects=subjects,
        tw_time=tw_time,
        train_time=train_time,
        test_time=test_time,
        subject_fisher_z=subject_z,
        mean_fisher_z=mean_z,
        observed_t=observed_t,
        null_max_cluster_mass=null_max,
        significant_mask=significant_mask,
        **{f"cluster_{i + 1}_mask": cluster["mask"] for i, cluster in enumerate(significant)},
    )
    report = []
    for index, cluster in enumerate(significant, start=1):
        report.append(
            {
                "cluster": index,
                "p": cluster["p"],
                "mass": cluster["mass"],
                "sign": cluster["sign"],
                "n_cells": cluster["n_cells"],
                "tw_seconds": [float(tw_time[cluster["tw_min"]]), float(tw_time[cluster["tw_max"]])],
                "train_seconds": [float(train_time[cluster["train_min"]]), float(train_time[cluster["train_max"]])],
                "test_seconds": [float(test_time[cluster["test_min"]]), float(test_time[cluster["test_max"]])],
            }
        )
    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"matched_trials": matched_counts, "significant_clusters": report},
            stream,
            indent=2,
        )
    return report
