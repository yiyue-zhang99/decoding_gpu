"""Guven matched-trial TW x time-cross decoding analysis in 3 dimensions.

Trial correlations are calculated within each of the four sessions, Fisher-z
transformed, and then averaged so every subject contributes one TW x training
time x test time map. Group inference and plotting reuse the established
Micheal 3-D cluster implementation.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np

MICHEAL_CROSS_DIR = (
    Path(__file__).resolve().parents[3] / "micheal" / "corr" / "cross_corr"
)
if str(MICHEAL_CROSS_DIR) not in sys.path:
    sys.path.insert(0, str(MICHEAL_CROSS_DIR))

from michael_tw_cross_correlation import (  # noqa: E402,F401
    cluster_signflip_3d,
    plot_3d_statistic_cluster_panels,
    plot_significant_3d_projections,
    print_significant_cluster_ranges,
)

SESSIONS = (1, 2, 3, 4)
TW_WINDOW_SIZE = 250


def _subject_number(path: Path) -> int:
    match = re.search(r"subject_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject number from {path.name}")
    return int(match.group(1))


def _tw_time(data: dict) -> np.ndarray:
    """Return centres of Guven's TW windows on the epoch time axis."""
    starts = np.asarray(data["starts"], dtype=float)
    return float(data["epoch_tmin"]) + (
        starts + TW_WINDOW_SIZE / 2
    ) / float(data["sfreq"])


def _tw_measure(data: dict, measure: str) -> np.ndarray:
    if measure == "fw":
        return 10 * np.log10(data["fwMax"] / data["fwMaxSS"])
    if measure == "bw":
        return 10 * np.log10(data["bwMax"] / data["bwMaxSS"])
    raise ValueError("measure must be 'fw' or 'bw'")


def _component_array(
    values: np.ndarray, cue: np.ndarray, component: str
) -> np.ndarray:
    """Select Guven TW component; difference is contra minus ipsi."""
    physical_left = values[0:5].mean(axis=0)
    physical_right = values[6:11].mean(axis=0)
    cue_left = (np.asarray(cue) == 1)[None, None, :]
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
        raise ValueError(
            f"Unknown component {component!r}; choose {tuple(components)}"
        )
    return components[component]


def _select_time(time: np.ndarray, limits, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("time stride must be >= 1")
    mask = np.ones(time.size, dtype=bool)
    if limits is not None:
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
    """Pearson correlation across matched trials at every 3-D cell."""
    if tw.ndim != 2 or decoding.ndim != 3 or tw.shape[0] != decoding.shape[0]:
        raise ValueError(
            "Expected trial x TW-time and trial x train-time x test-time"
        )
    if tw.shape[0] < 4:
        raise ValueError("At least four matched trials are required")
    n_trials = tw.shape[0]
    tw64 = np.asarray(tw, dtype=np.float64)
    decoding64 = np.asarray(decoding, dtype=np.float64).reshape(n_trials, -1)
    tw64 -= tw64.mean(axis=0, keepdims=True)
    decoding64 -= decoding64.mean(axis=0, keepdims=True)
    tw_norm = np.sqrt(np.square(tw64).sum(axis=0))
    decoding_norm = np.sqrt(np.square(decoding64).sum(axis=0))
    if np.any(tw_norm == 0) or np.any(decoding_norm == 0):
        raise ValueError("A TW or decoding cell has zero variance across trials")
    correlation = (tw64.T @ decoding64) / (
        tw_norm[:, None] * decoding_norm[None, :]
    )
    return correlation.reshape(tw.shape[1], *decoding.shape[1:]).astype(
        np.float32
    )


def build_subject_cross_maps(
    tw_dir: Path,
    decoding_dir: Path,
    measure="fw",
    component="difference",
    condition="cue",
    fmin=8.0,
    fmax=12.0,
    tw_limits=(0.75, 1.85),
    train_limits=(0.75, 1.85),
    test_limits=(0.75, 1.85),
    tw_stride=1,
    decoding_stride=1,
    pool_raw_trials_across_sessions=False,
):
    """Build one Fisher-z TW x train x test map per Guven subject.

    By default correlations are computed independently within each session,
    Fisher-z transformed, and equally averaged. This controls session-level
    offsets even though the decoder itself was fitted using pooled sessions.
    """
    tw_dir, decoding_dir = Path(tw_dir), Path(decoding_dir)
    paths = sorted(decoding_dir.glob("subject_*_alpha_time_cross_torch.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No subject_*_alpha_time_cross_torch.npz files in {decoding_dir}"
        )
    if condition not in {"cue", "uncue", "cue_minus_uncue"}:
        raise ValueError("condition must be cue, uncue, or cue_minus_uncue")

    subjects, subject_maps, matched_counts = [], [], {}
    selected_tw = selected_train = selected_test = None
    epsilon = np.finfo(np.float32).eps

    for decoding_path in paths:
        subject = _subject_number(decoding_path)
        session_maps, pooled_tw, pooled_evidence = [], [], []
        matched_counts[subject] = {}
        with np.load(decoding_path, allow_pickle=True) as decoded:
            time = np.asarray(decoded["time_dec"], dtype=float)
            train_idx = _select_time(time, train_limits, decoding_stride)
            test_idx = _select_time(time, test_limits, decoding_stride)
            this_train, this_test = time[train_idx], time[test_idx]
            if selected_train is None:
                selected_train, selected_test = this_train, this_test
            elif not (
                np.allclose(selected_train, this_train)
                and np.allclose(selected_test, this_test)
            ):
                raise ValueError(f"Decoding time mismatch in {decoding_path.name}")

            for session in SESSIONS:
                id_key = f"sess{session}_trial_ids"
                cue_key = f"sess{session}_trial_dec_cue"
                uncue_key = f"sess{session}_trial_dec_uncue"
                missing = [k for k in (id_key, cue_key, uncue_key) if k not in decoded]
                if missing:
                    raise ValueError(
                        f"{decoding_path.name} missing fields {missing}"
                    )
                trial_ids = np.asarray(decoded[id_key], dtype=int).reshape(-1)
                cue = np.asarray(decoded[cue_key], dtype=np.float32)[
                    :, train_idx
                ][:, :, test_idx]
                uncue = np.asarray(decoded[uncue_key], dtype=np.float32)[
                    :, train_idx
                ][:, :, test_idx]
                evidence = (
                    cue if condition == "cue" else uncue
                    if condition == "uncue" else cue - uncue
                )
                if evidence.shape[0] != trial_ids.size:
                    raise ValueError(
                        f"Trial/evidence mismatch: subject {subject}, "
                        f"session {session}"
                    )

                tw_path = tw_dir / f"sub{subject:02d}_session{session}.pkl"
                with tw_path.open("rb") as stream:
                    tw_data = pickle.load(stream)
                n_tw_trials = np.asarray(tw_data["cue_loc"]).size
                if np.any(trial_ids < 1) or np.any(trial_ids > n_tw_trials):
                    raise ValueError(f"Trial IDs exceed TW axis in {tw_path.name}")
                tw_rows = trial_ids - 1
                keep = np.ones(trial_ids.size, dtype=bool)
                if "is_bad_epoch" in tw_data:
                    keep &= ~np.asarray(tw_data["is_bad_epoch"], dtype=bool)[tw_rows]
                tw_rows, evidence = tw_rows[keep], evidence[keep]
                matched_counts[subject][session] = int(keep.sum())

                tw_time = _tw_time(tw_data)
                tw_idx = _select_time(tw_time, tw_limits, tw_stride)
                this_tw = tw_time[tw_idx]
                if selected_tw is None:
                    selected_tw = this_tw
                elif not np.allclose(selected_tw, this_tw):
                    raise ValueError(f"TW time mismatch in {tw_path.name}")
                frequency = np.asarray(tw_data["ff"], dtype=float)
                frequency_mask = (frequency >= fmin) & (frequency <= fmax)
                if not frequency_mask.any():
                    raise ValueError(f"No TW frequencies in {fmin:g}-{fmax:g} Hz")
                values = _tw_measure(tw_data, measure)
                component_values = _component_array(
                    values, np.asarray(tw_data["cue_loc"]), component
                )
                trial_tw = component_values[frequency_mask][:, tw_idx, :][
                    :, :, tw_rows
                ].mean(axis=0).T
                if pool_raw_trials_across_sessions:
                    pooled_tw.append(trial_tw)
                    pooled_evidence.append(evidence)
                else:
                    correlation = _trial_correlation_3d(trial_tw, evidence)
                    session_maps.append(
                        np.arctanh(
                            np.clip(correlation, -1 + epsilon, 1 - epsilon)
                        )
                    )
                del tw_data, values, component_values, trial_tw, evidence, cue, uncue

        if pool_raw_trials_across_sessions:
            correlation = _trial_correlation_3d(
                np.concatenate(pooled_tw, axis=0),
                np.concatenate(pooled_evidence, axis=0),
            )
            subject_map = np.arctanh(
                np.clip(correlation, -1 + epsilon, 1 - epsilon)
            )
        else:
            if len(session_maps) != len(SESSIONS):
                raise ValueError(
                    f"Expected {len(SESSIONS)} sessions for subject {subject}"
                )
            subject_map = np.mean(session_maps, axis=0, dtype=np.float32)
        subject_maps.append(subject_map)
        subjects.append(subject)
        print(
            f"subject {subject:02d}: "
            + ", ".join(
                f"s{s}={matched_counts[subject][s]}" for s in SESSIONS
            )
            + f"; map={subject_map.shape}",
            flush=True,
        )

    return (
        np.asarray(subjects), selected_tw, selected_train, selected_test,
        np.stack(subject_maps), matched_counts,
    )
