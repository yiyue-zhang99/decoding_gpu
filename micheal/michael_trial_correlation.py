"""Matched single-trial TW × alpha-decoding correlations."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from scipy.stats import t as student_t


EVENT_TIMES = (0.0, 1.2, 1.8, 3.8, 4.3)
DECODING_KEYS = ("early", "late", "early_minus_late")


def _subject_number(path: Path) -> int:
    match = re.search(r"subject_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject number from {path.name}")
    return int(match.group(1))


def _tw_time(data: dict) -> np.ndarray:
    starts = np.asarray(data.get("starts", data["time"]))
    return -1.25 + (starts + 125) / 500


def _tw_measure(data: dict, measure: str) -> np.ndarray:
    if measure == "fw":
        return 10 * np.log10(data["fwmax"] / data["fwssmax"])
    if measure == "bw":
        return 10 * np.log10(data["bwmax"] / data["bwssmax"])
    raise ValueError("Trial correlation supports measure='fw' or 'bw'")


def _component_arrays(values: np.ndarray, cue: np.ndarray) -> dict:
    cue_two = (cue == 2)[None, None, :]
    physical_left = values[0:5].mean(axis=0)
    physical_right = values[6:11].mean(axis=0)
    # physical_left = values[[4]].mean(axis=0)
    # physical_right = values[[7]].mean(axis=0)
    
    contra = np.where(cue_two, physical_right, physical_left)
    ipsi = np.where(cue_two, physical_left, physical_right)
    return {
        "difference": contra - ipsi,
        "contra": contra,
        "ipsi": ipsi,
        "all_lines": values.mean(axis=0),
        "midline": values[5],
    }


def _trial_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return x-time × y-time Pearson r across exactly matched rows/trials."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("TW and decoding must contain the same matched trials")
    x_sd = x.std(axis=0, ddof=1)
    y_sd = y.std(axis=0, ddof=1)
    if np.any(x_sd == 0) or np.any(y_sd == 0):
        raise ValueError("A time point has zero variance across trials")
    xz = (x - x.mean(axis=0)) / x_sd
    yz = (y - y.mean(axis=0)) / y_sd
    return xz.T @ yz / (x.shape[0] - 1)


def _combine_session_correlations(
    sessions: list[np.ndarray], fisher_transform: bool
) -> np.ndarray:
    """Average two session correlations, optionally in Fisher-z space."""
    if fisher_transform:
        epsilon = np.finfo(float).eps
        sessions = [
            np.arctanh(np.clip(values, -1.0 + epsilon, 1.0 - epsilon))
            for values in sessions
        ]
    return np.mean(sessions, axis=0)


def build_subject_trial_maps(
    tw_dir: Path,
    decoding_dir: Path,
    measures=("fw", "bw"),
    components=("difference", "contra", "ipsi", "midline"),
    fmin=8,
    fmax=12,
    tw_tmin=0.0,
    decoding_tmin=0.0,
    fisher_transform=False,
    pool_raw_trials_across_sessions=False,
):
    """Build subject effect maps after session-wise matched-trial correlations.

    When ``fisher_transform`` is true, each session Pearson-r map is
    transformed with arctanh before the two sessions are averaged.  The
    returned maps are therefore Fisher-z values in that mode and raw Pearson-r
    values otherwise.

    When ``pool_raw_trials_across_sessions`` is true, the filtered raw trial
    rows from both sessions are concatenated first and a single subject-level
    Pearson-r map is computed. No within-session centering or standardization
    is applied before concatenation.
    """
    decoding_dir = Path(decoding_dir)
    alpha_paths = sorted(decoding_dir.glob("subject_*_alpha_torch.npz"))
    voltage_paths = sorted(decoding_dir.glob("subject_*_voltage_torch.npz"))
    if alpha_paths and voltage_paths:
        raise ValueError(
            f"{decoding_dir} contains both alpha and voltage decoding files. "
            "Place the two result types in separate directories."
        )
    paths = alpha_paths or voltage_paths
    if not paths:
        raise FileNotFoundError(
            f"No single-trial decoding files found in {decoding_dir}. "
            "Expected subject_*_alpha_torch.npz or "
            "subject_*_voltage_torch.npz."
        )
    output = {
        measure: {
            component: {key: [] for key in DECODING_KEYS}
            for component in components
        }
        for measure in measures
    }
    subjects = []
    matched_counts = {}
    tw_time_selected = None
    decoding_time_selected = None

    for alpha_path in paths:
        subject = _subject_number(alpha_path)
        with np.load(alpha_path, allow_pickle=True) as alpha:
            required = {
                f"sess{session}_trial_dec_{key}"
                for session in (1, 2)
                for key in ("early", "late")
            } | {f"sess{session}_trial_ids" for session in (1, 2)}
            missing = sorted(required - set(alpha.files))
            if missing:
                raise ValueError(
                    f"{alpha_path.name} is an old subject-average result; "
                    f"missing single-trial fields: {missing}"
                )
            decoding_time = np.asarray(alpha["time_dec"], dtype=float)
            decoding_mask = decoding_time >= decoding_tmin
            if decoding_time_selected is None:
                decoding_time_selected = decoding_time[decoding_mask]
            elif not np.allclose(
                decoding_time_selected, decoding_time[decoding_mask]
            ):
                raise ValueError(f"Decoding time mismatch in {alpha_path}")

            subject_sessions = {
                measure: {
                    component: {key: [] for key in DECODING_KEYS}
                    for component in components
                }
                for measure in measures
            }
            pooled_tw = {
                measure: {component: [] for component in components}
                for measure in measures
            }
            pooled_decoding = {key: [] for key in DECODING_KEYS}
            matched_counts[subject] = {}

            for session in (1, 2):
                trial_ids = np.asarray(
                    alpha[f"sess{session}_trial_ids"], dtype=int
                ).reshape(-1)
                decoding = {
                    key: np.asarray(
                        alpha[f"sess{session}_trial_dec_{key}"], dtype=float
                    )[:, decoding_mask]
                    for key in ("early", "late")
                }
                decoding["early_minus_late"] = (
                    decoding["early"] - decoding["late"]
                )
                if any(values.shape[0] != trial_ids.size for values in decoding.values()):
                    raise ValueError(
                        f"Trial ID/decoding row mismatch: subject {subject}, "
                        f"session {session}"
                    )

                tw_path = Path(tw_dir) / f"subj{subject:02d}_sess{session}.pkl"
                with tw_path.open("rb") as stream:
                    tw_data = pickle.load(stream)
                n_tw_trials = np.asarray(tw_data["cue_loc"]).size
                if np.any(trial_ids < 1) or np.any(trial_ids > n_tw_trials):
                    raise ValueError(
                        f"Saved trial IDs exceed TW trial axis in {tw_path.name}"
                    )

                tw_rows = trial_ids - 1
                common = np.ones(trial_ids.size, dtype=bool)
                if "is_bad_epoch" in tw_data:
                    common &= ~np.asarray(
                        tw_data["is_bad_epoch"], dtype=bool
                    )[tw_rows]
                if "has_timing_issue" in tw_data:
                    common &= ~np.asarray(
                        tw_data["has_timing_issue"], dtype=bool
                    )[tw_rows]
                trial_ids = trial_ids[common]
                tw_rows = tw_rows[common]
                decoding = {key: values[common] for key, values in decoding.items()}
                if pool_raw_trials_across_sessions:
                    for key in DECODING_KEYS:
                        pooled_decoding[key].append(decoding[key])
                matched_counts[subject][session] = int(trial_ids.size)
                if trial_ids.size < 4:
                    raise ValueError(
                        f"Too few matched trials: subject {subject}, session {session}"
                    )

                tw_time = _tw_time(tw_data)
                tw_mask = tw_time >= tw_tmin
                if tw_time_selected is None:
                    tw_time_selected = tw_time[tw_mask]
                elif not np.allclose(tw_time_selected, tw_time[tw_mask]):
                    raise ValueError(f"TW time mismatch in {tw_path.name}")
                frequencies = np.asarray(tw_data["ff"])
                frequency_mask = (frequencies >= fmin) & (frequencies <= fmax)
                cue = np.asarray(tw_data["cue_loc"])[tw_rows]

                for measure in measures:
                    values = _tw_measure(tw_data, measure)
                    component_data = _component_arrays(values, np.asarray(tw_data["cue_loc"]))
                    for component in components:
                        # component is frequency × TW-time × original-trial.
                        trial_tw = component_data[component][frequency_mask][
                            :, tw_mask, :
                        ][:, :, tw_rows].mean(axis=0).T
                        if not np.array_equal(
                            trial_ids, np.asarray(alpha[f"sess{session}_trial_ids"])[common]
                        ):
                            raise AssertionError("Trial identity/order changed")
                        if pool_raw_trials_across_sessions:
                            pooled_tw[measure][component].append(trial_tw)
                        else:
                            for key in DECODING_KEYS:
                                correlation = _trial_correlation(
                                    trial_tw, decoding[key]
                                )
                                subject_sessions[measure][component][key].append(
                                    correlation
                                )
                    del values
                del tw_data

            for measure in measures:
                for component in components:
                    for key in DECODING_KEYS:
                        if pool_raw_trials_across_sessions:
                            correlation = _trial_correlation(
                                np.concatenate(
                                    pooled_tw[measure][component], axis=0
                                ),
                                np.concatenate(pooled_decoding[key], axis=0),
                            )
                            output[measure][component][key].append(
                                _combine_session_correlations(
                                    [correlation], fisher_transform
                                )
                            )
                            continue
                        sessions = subject_sessions[measure][component][key]
                        if len(sessions) != 2:
                            raise ValueError(
                                f"Expected two sessions for subject {subject}"
                            )
                        output[measure][component][key].append(
                            _combine_session_correlations(
                                sessions, fisher_transform
                            )
                        )
        subjects.append(subject)
        print(
            f"subject {subject:02d}: matched trials "
            f"s1={matched_counts[subject][1]}, s2={matched_counts[subject][2]}",
            flush=True,
        )

    output = {
        measure: {
            component: {
                key: np.stack(subject_maps)
                for key, subject_maps in key_data.items()
            }
            for component, key_data in component_data.items()
        }
        for measure, component_data in output.items()
    }
    return subjects, tw_time_selected, decoding_time_selected, output, matched_counts


def cluster_signflip_trial_maps(
    subject_r: np.ndarray,
    permutations=5000,
    cluster_alpha=0.05,
    cluster_p=0.05,
    seed=2026,
    return_all=False,
):
    """Group test of within-subject Pearson-r maps against zero."""
    n_subjects = subject_r.shape[0]
    mean_r = subject_r.mean(axis=0)
    sem = subject_r.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    observed_t = np.divide(
        mean_r, sem, out=np.zeros_like(mean_r), where=sem > 0
    )
    threshold = student_t.ppf(1 - cluster_alpha / 2, n_subjects - 1)
    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    observed_labels, count = label(
        np.abs(observed_t) > threshold, connectivity
    )

    rng = np.random.default_rng(seed)
    null_max = np.zeros(permutations)
    for permutation in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(n_subjects, 1, 1))
        permuted = subject_r * signs
        perm_mean = permuted.mean(axis=0)
        perm_sem = permuted.std(axis=0, ddof=1) / np.sqrt(n_subjects)
        perm_t = np.divide(
            perm_mean,
            perm_sem,
            out=np.zeros_like(perm_mean),
            where=perm_sem > 0,
        )
        labels, n_clusters = label(
            np.abs(perm_t) > threshold, connectivity
        )
        if n_clusters:
            null_max[permutation] = max(
                np.abs(perm_t)[labels == index].sum()
                for index in range(1, n_clusters + 1)
            )

    clusters = []
    for index in range(1, count + 1):
        mask = observed_labels == index
        mass = np.abs(observed_t)[mask].sum()
        p_value = (1 + np.count_nonzero(null_max >= mass)) / (
            permutations + 1
        )
        rows, columns = np.where(mask)
        cluster = {
            "mask": mask,
            "p": p_value,
            "mass": mass,
            "significant": p_value < cluster_p,
            "row_min": int(rows.min()),
            "row_max": int(rows.max()),
            "col_min": int(columns.min()),
            "col_max": int(columns.max()),
        }
        if return_all or cluster["significant"]:
            clusters.append(cluster)
    return mean_r, clusters


def _add_cluster_contours(axis, clusters, decoding_ms, tw_time):
    for cluster in clusters:
        axis.contour(
            decoding_ms,
            tw_time,
            cluster["mask"].astype(float),
            levels=[0.5],
            colors="black",
            linewidths=2,
        )


def plot_trial_maps(
    maps,
    clusters,
    decoding_time,
    tw_time,
    output: Path,
    title,
    value_label="Mean within-subject trial correlation (r)",
):
    decoding_ms = decoding_time * 1000
    limit = max(np.abs(values).max() for values in maps.values())
    extent = [decoding_ms[0], decoding_ms[-1], tw_time[0], tw_time[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True, sharey=True)
    for axis, key in zip(axes, ("early", "late")):
        image = axis.imshow(
            maps[key], origin="lower", aspect="auto", extent=extent,
            cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest",
        )
        _add_cluster_contours(axis, clusters[key], decoding_ms, tw_time)
        for event in EVENT_TIMES:
            axis.axhline(event, color=".3", ls=":", lw=.7, alpha=.5)
            axis.axvline(event * 1000, color=".3", ls=":", lw=.7, alpha=.5)
        # Event markers outside a cropped ROI must not expand the displayed
        # axes; these limits also make the plotted area match the tested map.
        axis.set_xlim(decoding_ms[0], decoding_ms[-1])
        axis.set_ylim(tw_time[0], tw_time[-1])
        axis.set_title(f"Tested {key}")
        axis.set_xlabel("Single-trial decoding time (ms)")
    axes[0].set_ylabel("Single-trial travelling-wave time (s)")
    fig.suptitle(title)
    fig.subplots_adjust(
        left=.08, right=.84, bottom=.12, top=.86, wspace=.08
    )
    colorbar_axis = fig.add_axes([.87, .16, .018, .66])
    fig.colorbar(
        image, cax=colorbar_axis,
        label=value_label,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
