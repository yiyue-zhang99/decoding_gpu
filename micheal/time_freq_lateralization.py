#!/usr/bin/env python3
"""Single-trial time-frequency lateralization for the Michael dataset.

This is a Python counterpart of ``time_freq_lateralization.m``. Power is
estimated with a Hanning-tapered sliding Fourier transform. Frequency windows
can be adaptive (cycles / frequency, matching FieldTrip's t_ftimwin) or fixed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import label
from scipy.signal import fftconvolve
from scipy.stats import t as student_t


PROJECT_DIR = Path("/home/dilay/project2/tw")
DEFAULT_DATA_DIR = PROJECT_DIR / "Data" / "Micheal_Data_exp2"
LEFT_CHANNELS = ("P7", "P5", "P3", "P1", "PO7", "PO3", "O1")
RIGHT_CHANNELS = ("P8", "P6", "P4", "P2", "PO8", "PO4", "O2")
BAD_TRIAL_FIELDS = (
    "bad_trials_mem",
    "bad_trials_imp1",
    "bad_trials_imp2",
    "bad_trials_probe1",
    "bad_trials_probe2",
)


@dataclass(frozen=True)
class AnalysisConfig:
    frequency_min: float = 6.0
    frequency_max: float = 16.0
    frequency_step: float = 0.5
    toi_min: float = -0.1
    toi_max: float = 4.8
    toi_step_samples: int = 5
    window_mode: str = "adaptive"
    cycles: float = 5.0
    fixed_window_ms: float = 500.0
    batch_trials: int = 32

    def frequencies(self) -> np.ndarray:
        values = np.arange(
            self.frequency_min,
            self.frequency_max + self.frequency_step / 2,
            self.frequency_step,
            dtype=np.float64,
        )
        return values[values <= self.frequency_max + 1e-12]

    def validate(self) -> None:
        if not 0 < self.frequency_min <= self.frequency_max:
            raise ValueError("frequency range must be positive and ordered")
        if self.frequency_step <= 0:
            raise ValueError("frequency_step must be positive")
        if self.toi_min >= self.toi_max:
            raise ValueError("toi_min must be smaller than toi_max")
        if self.toi_step_samples < 1:
            raise ValueError("toi_step_samples must be at least 1")
        if self.window_mode not in {"adaptive", "fixed"}:
            raise ValueError("window_mode must be 'adaptive' or 'fixed'")
        if self.cycles <= 0 or self.fixed_window_ms <= 0:
            raise ValueError("cycles and fixed_window_ms must be positive")
        if self.batch_trials < 1:
            raise ValueError("batch_trials must be at least 1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 20)))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frequency-min", type=float, default=6.0)
    parser.add_argument("--frequency-max", type=float, default=16.0)
    parser.add_argument("--frequency-step", type=float, default=0.5)
    parser.add_argument("--toi", type=float, nargs=2, default=(-0.1, 4.8))
    parser.add_argument("--toi-step-samples", type=int, default=5)
    parser.add_argument(
        "--window-mode", choices=["adaptive", "fixed"], default="adaptive"
    )
    parser.add_argument("--cycles", type=float, default=5.0)
    parser.add_argument("--fixed-window-ms", type=float, default=500.0)
    parser.add_argument("--batch-trials", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_dir(config: AnalysisConfig) -> Path:
    if config.window_mode == "adaptive":
        tag = f"adaptive_{config.cycles:g}cycles"
    else:
        tag = f"fixed_{config.fixed_window_ms:g}ms"
    return PROJECT_DIR / "results" / "freq_python" / tag


def load_session(path: Path) -> dict:
    return loadmat(
        path,
        variable_names=["ft_mem"],
        simplify_cells=True,
    )["ft_mem"]


def _channel_indices(labels: np.ndarray, requested: tuple[str, ...]) -> np.ndarray:
    lookup = {
        str(label).strip().upper(): index
        for index, label in enumerate(np.asarray(labels).reshape(-1))
    }
    missing = [name for name in requested if name.upper() not in lookup]
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    return np.asarray([lookup[name.upper()] for name in requested], dtype=int)


def _good_trial_indices(ft_mem: dict) -> np.ndarray:
    n_trials = np.asarray(ft_mem["trial"]).shape[0]
    bad_parts = []
    for field in BAD_TRIAL_FIELDS:
        values = np.asarray(ft_mem.get(field, []), dtype=int).reshape(-1)
        if values.size:
            bad_parts.append(values - 1)
    bad = np.unique(np.concatenate(bad_parts)) if bad_parts else np.empty(0, int)
    return np.setdiff1d(np.arange(n_trials), bad)


def _toi_indices(time: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    eligible = np.flatnonzero(
        (time >= config.toi_min) & (time <= config.toi_max)
    )
    if eligible.size == 0:
        raise ValueError("No samples lie inside the requested toi")
    return eligible[:: config.toi_step_samples]


def _window_samples(frequency: float, hz: float, config: AnalysisConfig) -> int:
    if config.window_mode == "adaptive":
        seconds = config.cycles / frequency
    else:
        seconds = config.fixed_window_ms / 1000
    return max(2, round(seconds * hz))


def _complete_toi_indices(
    time: np.ndarray,
    hz: float,
    frequencies: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Keep requested centres with a complete window at every frequency."""
    requested = _toi_indices(time, config)
    window_lengths = np.asarray(
        [_window_samples(frequency, hz, config) for frequency in frequencies]
    )
    max_left = max((window_lengths - 1) // 2)
    max_right = max(window_lengths - 1 - (window_lengths - 1) // 2)
    complete = requested[
        (requested >= max_left)
        & (requested + max_right < time.size)
    ]
    if complete.size == 0:
        raise ValueError(
            "No requested toi has a complete Fourier window at every frequency"
        )
    if complete.size != requested.size:
        print(
            f"  trimmed incomplete edge toi values: requested "
            f"{1000 * time[requested[0]]:.0f}.."
            f"{1000 * time[requested[-1]]:.0f} ms, using "
            f"{1000 * time[complete[0]]:.0f}.."
            f"{1000 * time[complete[-1]]:.0f} ms",
            flush=True,
        )
    return complete


def _session_log_power_difference(
    raw: np.ndarray,
    time: np.ndarray,
    hz: float,
    frequencies: np.ndarray,
    toi_indices: np.ndarray,
    n_left: int,
    config: AnalysisConfig,
) -> np.ndarray:
    """Return right-minus-left mean log10 power per trial/frequency/time."""
    n_trials = raw.shape[0]
    output = np.empty(
        (n_trials, frequencies.size, toi_indices.size),
        dtype=np.float32,
    )
    tiny = np.finfo(np.float32).tiny

    for frequency_index, frequency in enumerate(frequencies):
        n_window = _window_samples(frequency, hz, config)
        half_left = (n_window - 1) // 2
        half_right = n_window - 1 - half_left
        if (
            toi_indices[0] - half_left < 0
            or toi_indices[-1] + half_right >= time.size
        ):
            raise ValueError(
                f"{frequency:g} Hz window ({n_window} samples) exceeds the "
                "available epoch at one or more requested toi values"
            )

        taper = np.hanning(n_window)
        taper /= np.sqrt(np.square(taper).sum())
        relative_time = (
            np.arange(n_window) - (n_window - 1) / 2
        ) / hz
        kernel = (
            taper * np.exp(2j * np.pi * frequency * relative_time)
        )[None, None, :]

        for first in range(0, n_trials, config.batch_trials):
            stop = min(first + config.batch_trials, n_trials)
            fourier = fftconvolve(
                raw[first:stop],
                kernel,
                mode="same",
                axes=-1,
            )[..., toi_indices]
            log_power = np.log10(
                np.maximum(np.square(np.abs(fourier)), tiny)
            )
            left = log_power[:, :n_left].mean(axis=1)
            right = log_power[:, n_left:].mean(axis=1)
            output[first:stop, frequency_index] = (
                right - left
            ).astype(np.float32, copy=False)

        print(
            f"    {frequency:g} Hz: {n_window} samples "
            f"({1000 * n_window / hz:.1f} ms)",
            flush=True,
        )
    return output


def _config_json(config: AnalysisConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True)


def subject_cache_is_current(path: Path, config: AnalysisConfig) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "lat_trials",
                "sess_label",
                "incl1",
                "incl2",
                "cue_loc",
                "frequencies",
                "plot_time",
                "config_json",
            }
            return required.issubset(saved.files) and (
                str(saved["config_json"].item()) == _config_json(config)
            )
    except (OSError, ValueError, EOFError):
        return False


def process_subject(
    subject: int,
    data_dir: Path,
    output_file: Path,
    config: AnalysisConfig,
) -> np.ndarray:
    """Compute and cache one subject; return frequency × time mean."""
    session_differences = []
    good_trials = []
    cue_locations = []
    reference_time = None
    reference_toi_indices = None
    frequencies = config.frequencies()

    for session in (1, 2):
        path = data_dir / f"MemImp3_mem_whole_sess{session}_{subject}.mat"
        ft_mem = load_session(path)
        time = np.asarray(ft_mem["time"], dtype=np.float64).reshape(-1)
        hz = round(1 / np.median(np.diff(time)))
        if frequencies[-1] >= hz / 2:
            raise ValueError("Requested frequency reaches or exceeds Nyquist")
        toi_indices = _complete_toi_indices(
            time,
            hz,
            frequencies,
            config,
        )
        if reference_time is None:
            reference_time = time
            reference_toi_indices = toi_indices
        elif not np.allclose(time, reference_time):
            raise ValueError(f"Session time mismatch for subject {subject}")

        labels = np.asarray(ft_mem["label"]).reshape(-1)
        left_indices = _channel_indices(labels, LEFT_CHANNELS)
        right_indices = _channel_indices(labels, RIGHT_CHANNELS)
        selected = np.r_[left_indices, right_indices]
        good = _good_trial_indices(ft_mem)
        cue = np.unique(np.asarray(ft_mem["Results"])[good, 2])
        if cue.size != 1:
            raise ValueError(
                f"Subject {subject}, session {session}: cue location is not constant"
            )
        cue_location = int(cue[0])

        raw = np.asarray(
            ft_mem["trial"][
                np.ix_(good, selected, np.arange(time.size))
            ],
            dtype=np.float32,
        )
        del ft_mem
        print(
            f"  session {session}: {good.size} trials, cue={cue_location}",
            flush=True,
        )
        right_minus_left = _session_log_power_difference(
            raw,
            time,
            hz,
            frequencies,
            toi_indices,
            len(left_indices),
            config,
        )
        del raw

        # Contra-minus-ipsi relative to the tested item.
        if cue_location == 1:
            lateralization = right_minus_left
        elif cue_location == 2:
            lateralization = -right_minus_left
        else:
            raise ValueError(f"Unexpected cue location: {cue_location}")
        session_differences.append(lateralization)
        good_trials.append(good + 1)
        cue_locations.append(cue_location)

    lat_trials = np.concatenate(session_differences, axis=0)
    sess_label = np.concatenate(
        [
            np.full(values.shape[0], session, dtype=np.int8)
            for session, values in enumerate(session_differences, start=1)
        ]
    )
    plot_time = reference_time[reference_toi_indices] * 1000
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        lat_trials=lat_trials,
        sess_label=sess_label,
        incl1=good_trials[0],
        incl2=good_trials[1],
        cue_loc=np.asarray(cue_locations, dtype=np.int8),
        frequencies=frequencies,
        plot_time=plot_time,
        left_channels=np.asarray(LEFT_CHANNELS),
        right_channels=np.asarray(RIGHT_CHANNELS),
        config_json=_config_json(config),
    )
    temporary.replace(output_file)
    print(f"  saved {output_file}", flush=True)
    return lat_trials.mean(axis=0)


def run_analysis(
    subjects: list[int] | tuple[int, ...],
    data_dir: Path,
    output_dir: Path,
    config: AnalysisConfig,
    overwrite: bool = False,
) -> Path:
    """Run/cached-load subjects and save a group summary."""
    config.validate()
    subject_means = []
    for subject in subjects:
        output_file = output_dir / f"sub{subject:02d}.npz"
        if not overwrite and subject_cache_is_current(output_file, config):
            print(f"Subject {subject:02d}: cached", flush=True)
            with np.load(output_file, allow_pickle=False) as saved:
                subject_means.append(saved["lat_trials"].mean(axis=0))
        else:
            print(f"Subject {subject:02d}", flush=True)
            subject_means.append(
                process_subject(subject, data_dir, output_file, config)
            )

    freq_lat_sub = np.stack(subject_means).astype(np.float32, copy=False)
    with np.load(output_dir / f"sub{subjects[0]:02d}.npz") as first:
        frequencies = first["frequencies"]
        plot_time = first["plot_time"]
    group_file = output_dir / "group_freq_lateralization.npz"
    temporary = group_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        freq_lat_sub=freq_lat_sub,
        subjects=np.asarray(subjects, dtype=int),
        frequencies=frequencies,
        plot_time=plot_time,
        config_json=_config_json(config),
    )
    temporary.replace(group_file)
    print(f"Saved {group_file}", flush=True)
    return group_file


def cluster_sign_flip_2d(
    subject_data: np.ndarray,
    permutations: int = 50_000,
    cluster_alpha: float = 0.05,
    cluster_p: float = 0.05,
    seed: int = 2026,
) -> tuple[np.ndarray, list[dict]]:
    """Two-sided cluster-mass sign-flip test across subjects."""
    n_subjects = subject_data.shape[0]
    mean = subject_data.mean(axis=0)
    sem = subject_data.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    observed_t = np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)
    threshold = student_t.ppf(1 - cluster_alpha / 2, n_subjects - 1)
    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    observed_labels, n_observed = label(
        np.abs(observed_t) > threshold,
        connectivity,
    )

    rng = np.random.default_rng(seed)
    null_max = np.zeros(permutations)
    for permutation in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(n_subjects, 1, 1))
        permuted = subject_data * signs
        perm_mean = permuted.mean(axis=0)
        perm_sem = permuted.std(axis=0, ddof=1) / np.sqrt(n_subjects)
        perm_t = np.divide(
            perm_mean,
            perm_sem,
            out=np.zeros_like(perm_mean),
            where=perm_sem > 0,
        )
        perm_labels, n_clusters = label(
            np.abs(perm_t) > threshold,
            connectivity,
        )
        if n_clusters:
            null_max[permutation] = max(
                np.abs(perm_t)[perm_labels == index].sum()
                for index in range(1, n_clusters + 1)
            )

    clusters = []
    for index in range(1, n_observed + 1):
        mask = observed_labels == index
        mass = float(np.abs(observed_t)[mask].sum())
        p_value = float(
            (1 + np.count_nonzero(null_max >= mass)) / (permutations + 1)
        )
        if p_value < cluster_p:
            clusters.append({"mask": mask, "mass": mass, "p": p_value})
    return mean, clusters


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        frequency_min=args.frequency_min,
        frequency_max=args.frequency_max,
        frequency_step=args.frequency_step,
        toi_min=args.toi[0],
        toi_max=args.toi[1],
        toi_step_samples=args.toi_step_samples,
        window_mode=args.window_mode,
        cycles=args.cycles,
        fixed_window_ms=args.fixed_window_ms,
        batch_trials=args.batch_trials,
    )
    output_dir = args.output_dir or default_output_dir(config)
    run_analysis(
        args.subjects,
        args.data_dir,
        output_dir,
        config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
