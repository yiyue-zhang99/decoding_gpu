#!/usr/bin/env python3
"""Pooled-session trial-wise alpha Mahalanobis decoding, Guven cue/uncue items.

Cue and uncue item orientation are each one of exactly 6 categories (see
guven_behavior.ITEM_CATEGORY_DEG), so this uses mahal_discrete_kfold_torch's
hard k-fold decoder (6 folds, one per category) instead of the continuous
kernel or trial-relative LOOCV decoders used for Michael's continuous
orientations.

Alpha-power extraction (FIR/Hilbert, IIR/Hilbert, or Hanning-tapered sliding
Fourier power) is copied unchanged from micheal_alpha_decoding_gpu.py's
alpha_power/mtmconvol_power -- those are generic, not michael-specific.

The four sessions are feature-standardised separately, then pooled. Cross-
validation folds are stratified within every session x category cell, while
templates and covariance are estimated from all pooled training trials.

Bad epochs (epochs.metadata["is_bad_epoch"], from fft/guven/preprocess.py)
are dropped before decoding, matching micheal_alpha_decoding_gpu.py's
bad-trial handling. Trial IDs are 1-based row numbers into the full (bad
epochs included) per-session trial axis, so they line up with the TW pickle
files written by fft/guven/guven_tw.py (see guven_tw.py's docstring): a
later correlation step can index tw_rows = trial_ids - 1 exactly as
guven_trial_correlation.py does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import (
    butter,
    fftconvolve,
    filtfilt,
    firwin,
    hilbert,
    sosfiltfilt,
)

PROJECT_DIR = Path("/home/dilay/project2/tw")
sys.path.insert(0, str(PROJECT_DIR / "functions" / "decoding" / "decoding_gpu"))
sys.path.insert(0, str(PROJECT_DIR / "travelling_waves" / "tw" / "fft" / "guven"))

from helper import DecodeConfig, make_sliding_features  # noqa: E402
from mahal_discrete_kfold_torch import decode_discrete_repetitions  # noqa: E402
from mahal_discrete_loocv_torch import decode_discrete_loocv  # noqa: E402
from guven_behavior import item_theta_rad, load_behavior_table  # noqa: E402

EPOCH_DIR = PROJECT_DIR / "Data" / "guven_data" / "epoch_autoreject"

CHANNELS = [
    "PZ", "POZ", "OZ", "P1", "PO3", "O1", "P7", "PO7", "P5", "P3", "P2",
    "PO4", "O2", "P8", "PO8", "P4", "P6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit result directory. If omitted, use the legacy "
            "automatically generated directory name."
        ),
    )
    parser.add_argument(
        "--cv-method", choices=["kfold", "loocv"], default="kfold"
    )
    parser.add_argument(
        "--pool-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pool sessions with session-stratified folds (default), or use "
            "--no-pool-sessions to decode every session independently"
        ),
    )
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--span-ms", type=float, default=10.0)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--smooth-ms", type=float, default=0.0)
    parser.add_argument("--toi", type=float, nargs=2, default=(-1.15, 5.45))
    parser.add_argument(
        "--decoding-feature-mode",
        choices=["sliding-window", "timepoint"],
        default="sliding-window",
    )
    parser.add_argument("--alpha-low", type=float, default=8.0)
    parser.add_argument("--alpha-high", type=float, default=12.0)
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir", "mtmconvol"],
        default="fir",
        help=(
            "Alpha-power method: FIR/Hilbert, IIR/Hilbert, or "
            "Hanning-tapered sliding Fourier power"
        ),
    )
    parser.add_argument(
        "--mtm-window-mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help=(
            "For mtmconvol: frequency-adaptive cycles/frequency windows "
            "or one fixed window for every frequency"
        ),
    )
    parser.add_argument("--mtm-cycles", type=float, default=5.0)
    parser.add_argument("--mtm-fixed-window-ms", type=float, default=500.0)
    parser.add_argument("--mtm-frequency-step", type=float, default=1.0)
    parser.add_argument("--filter-batch-trials", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute subjects whose complete output file already exists",
    )
    return parser.parse_args()


def channel_indices(ch_names: list[str]) -> np.ndarray:
    available = {name.strip().upper(): index for index, name in enumerate(ch_names)}
    missing = [name for name in CHANNELS if name not in available]
    if missing:
        raise ValueError(f"Missing requested channels: {missing}")
    return np.asarray([available[name] for name in CHANNELS], dtype=int)


def load_session(subject: int, session: int) -> dict:
    fif_path = EPOCH_DIR / f"sub{subject:02d}_session{session}.fif"
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    ch_names = epochs.copy().pick("eeg").ch_names
    indices = channel_indices(ch_names)
    data = epochs.get_data(picks=ch_names)[:, indices, :]  # trial x channel x time
    time = epochs.times
    is_bad_epoch = epochs.metadata["is_bad_epoch"].to_numpy()

    behavior = load_behavior_table(subject, session)
    if len(behavior) != data.shape[0]:
        raise ValueError(
            f"sub{subject:02d}_session{session}: {data.shape[0]} epochs but "
            f"{len(behavior)} behavioural trials"
        )

    good = np.flatnonzero(~is_bad_epoch)
    return dict(
        data=data[good],
        time=time,
        behavior=behavior.iloc[good].reset_index(drop=True),
        trial_ids=good + 1,  # 1-based, matches TW pickle trial axis
    )


def alpha_power(
    raw: np.ndarray,
    hz: float,
    low: float,
    high: float,
    batch_trials: int,
    method: str,
    mtm_window_mode: str = "adaptive",
    mtm_cycles: float = 5.0,
    mtm_fixed_window_ms: float = 500.0,
    mtm_frequency_step: float = 1.0,
) -> np.ndarray:
    """Return alpha power using Hilbert or sliding tapered Fourier analysis."""
    if method == "mtmconvol":
        return mtmconvol_power(
            raw,
            hz,
            low,
            high,
            batch_trials,
            window_mode=mtm_window_mode,
            cycles=mtm_cycles,
            fixed_window_ms=mtm_fixed_window_ms,
            frequency_step=mtm_frequency_step,
        )
    if method == "fir":
        filter_order = 3 * int(hz / low)
        if filter_order % 2:
            filter_order += 1
        numtaps = filter_order + 1
        coefficients = firwin(numtaps, [low, high], pass_zero=False, fs=hz, window="hamming")
        print(f"  FIR: {numtaps} taps, {numtaps / hz:.3f} s (3 cycles at {low:g} Hz)", flush=True)
    elif method == "iir":
        coefficients = butter(4, [low, high], btype="bandpass", fs=hz, output="sos")
        print(f"  IIR: 4th-order Butterworth SOS, {low:g}-{high:g} Hz", flush=True)
    else:
        raise ValueError(f"Unknown filter method: {method}")

    amplitude = np.empty(raw.shape, dtype=np.float32)
    for first in range(0, raw.shape[0], batch_trials):
        stop = min(first + batch_trials, raw.shape[0])
        raw_batch = raw[first:stop]
        if method == "fir":
            filtered = filtfilt(coefficients, [1.0], raw_batch, axis=-1)
        else:
            filtered = sosfiltfilt(coefficients, raw_batch, axis=-1)
        analytic = hilbert(filtered, axis=-1)
        amplitude[first:stop] = np.square(np.abs(analytic)).astype(np.float32, copy=False)
        print(f"  alpha extraction trials {first + 1}-{stop}/{raw.shape[0]}", flush=True)
    return amplitude


def mtmconvol_power(
    raw: np.ndarray,
    hz: float,
    low: float,
    high: float,
    batch_trials: int,
    window_mode: str,
    cycles: float,
    fixed_window_ms: float,
    frequency_step: float,
) -> np.ndarray:
    """Hanning-tapered sliding Fourier power, averaged across frequencies."""
    if frequency_step <= 0:
        raise ValueError("mtm-frequency-step must be positive")
    if cycles <= 0:
        raise ValueError("mtm-cycles must be positive")
    if fixed_window_ms <= 0:
        raise ValueError("mtm-fixed-window-ms must be positive")

    frequencies = np.arange(low, high + frequency_step / 2, frequency_step, dtype=np.float64)
    frequencies = frequencies[frequencies <= high + 1e-12]
    if frequencies.size == 0:
        raise ValueError("No mtmconvol frequencies fall inside the alpha band")

    power_sum = np.zeros(raw.shape, dtype=np.float32)
    print(
        f"  MTMCONVOL: Hanning taper, frequencies="
        f"{np.array2string(frequencies, precision=3)}, window={window_mode}",
        flush=True,
    )
    for frequency in frequencies:
        if window_mode == "adaptive":
            window_seconds = cycles / frequency
        elif window_mode == "fixed":
            window_seconds = fixed_window_ms / 1000
        else:
            raise ValueError(f"Unknown mtmconvol window mode: {window_mode}")

        window_samples = max(2, round(window_seconds * hz))
        taper = np.hanning(window_samples)
        taper /= np.sqrt(np.square(taper).sum())
        relative_time = (np.arange(window_samples) - (window_samples - 1) / 2) / hz
        kernel = (taper * np.exp(2j * np.pi * frequency * relative_time))[None, None, :]
        pad_left = (window_samples - 1) // 2
        pad_right = window_samples - 1 - pad_left
        print(
            f"    {frequency:g} Hz: {window_samples} samples "
            f"({1000 * window_samples / hz:.1f} ms)",
            flush=True,
        )

        for first in range(0, raw.shape[0], batch_trials):
            stop = min(first + batch_trials, raw.shape[0])
            padded = np.pad(
                raw[first:stop], ((0, 0), (0, 0), (pad_left, pad_right)), mode="reflect"
            )
            fourier = fftconvolve(padded, kernel, mode="valid", axes=-1)
            power_sum[first:stop] += np.square(np.abs(fourier)).astype(np.float32, copy=False)
        print(f"    completed {frequency:g} Hz for {raw.shape[0]} trials", flush=True)

    power_sum /= frequencies.size
    return power_sum


def output_is_complete(path: Path, sessions) -> bool:
    if not path.is_file():
        return False
    required = {"time_dec", "dec_cue", "dec_uncue", "dist_cue", "dist_uncue"}
    required |= {f"sess{s}_trial_ids" for s in sessions}
    required |= {f"sess{s}_trial_dec_{k}" for s in sessions for k in ("cue", "uncue")}
    try:
        with np.load(path, allow_pickle=True) as saved:
            return required.issubset(saved.files)
    except (OSError, ValueError, EOFError):
        return False


def main() -> None:
    args = parse_args()
    if args.filter_method == "mtmconvol":
        mtm_tag = "adaptive" if args.mtm_window_mode == "adaptive" else "fixed"
        output_name = f"guven_alpha_decoding_discrete_mtmconvol_{mtm_tag}"
    else:
        output_name = f"guven_alpha_decoding_discrete_{args.filter_method}"
    if args.cv_method == "loocv":
        output_name = output_name.replace("discrete_", "discrete_loocv_")
    if args.decoding_feature_mode == "timepoint":
        output_name += "_timepoint"
    output_name += "_pooled_sessions" if args.pool_sessions else "_separate_sessions"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_DIR / "results" / output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DecodeConfig(
        n_folds=args.n_folds,
        reps=args.reps,
        step_ms=args.step_ms,
        span_ms=args.span_ms,
        window_ms=args.window_ms,
        smooth_ms=args.smooth_ms,
        toi=tuple(args.toi),
    )

    for subject in args.subjects:
        output = output_dir / f"subject_{subject:02d}_alpha_torch.npz"
        if not args.overwrite and output_is_complete(output, args.sessions):
            print(f"skipping completed {output}", flush=True)
            continue

        session_results = []
        for session in args.sessions:
            fif_path = EPOCH_DIR / f"sub{subject:02d}_session{session}.fif"
            if not fif_path.is_file():
                print(f"[skip] missing {fif_path}", flush=True)
                continue
            print(f"subject {subject:02d}, session {session}", flush=True)
            session_data = load_session(subject, session)
            hz = float(1.0 / np.median(np.diff(session_data["time"])))

            power = alpha_power(
                session_data["data"],
                hz,
                args.alpha_low,
                args.alpha_high,
                args.filter_batch_trials,
                args.filter_method,
                mtm_window_mode=args.mtm_window_mode,
                mtm_cycles=args.mtm_cycles,
                mtm_fixed_window_ms=args.mtm_fixed_window_ms,
                mtm_frequency_step=args.mtm_frequency_step,
            )
            del session_data["data"]

            # Deliberately identical to raw-voltage feature construction:
            # subtract each channel's window-mean, average within spans,
            # concatenate channel x segment features.
            features, time_dec = make_sliding_features(
                power,
                session_data["time"],
                config,
                mode=args.decoding_feature_mode,
            )
            del power
            print(f"  feature shape {features.shape}", flush=True)

            behavior = session_data["behavior"]
            session_results.append(
                {
                    "session": session,
                    "features": features,
                    "behavior": behavior,
                    "trial_ids": session_data["trial_ids"],
                    "cue_loc": behavior["cue_loc"].to_numpy(),
                    "rt": behavior["rt"].to_numpy(),
                    "acc": behavior["acc"].to_numpy(),
                }
            )

        if not session_results:
            print(f"[skip] subject {subject:02d}: no sessions found", flush=True)
            continue

        if args.pool_sessions:
            pooled_features = np.concatenate(
                [entry["features"] for entry in session_results], axis=0
            )
            session_strata = np.concatenate(
                [
                    np.full(len(entry["trial_ids"]), entry["session"], dtype=int)
                    for entry in session_results
                ]
            )
            decoded = {}
            for name, column in (("cue", "cue_item"), ("uncue", "uncue_item")):
                theta = item_theta_rad(
                    np.concatenate(
                        [entry["behavior"][column].to_numpy() for entry in session_results]
                    )
                )
                decode_seed = args.seed + 1000 * subject
                decode_seed += 1 if name == "uncue" else 0
                if args.cv_method == "loocv":
                    print(
                        f" pooled decoding {name}: 6-category discrete "
                        "LOOCV, session-aware train-only scaling",
                        flush=True,
                    )
                    decoded[name] = decode_discrete_loocv(
                        pooled_features,
                        theta,
                        config,
                        device=args.device,
                        dtype=args.dtype,
                        seed=decode_seed,
                        return_trialwise=True,
                        strata=session_strata,
                    )
                else:
                    print(
                        f" pooled decoding {name}: 6-category hard k-fold, "
                        f"stratified by session x category, "
                        f"n_folds={args.n_folds}, reps={args.reps}",
                        flush=True,
                    )
                    decoded[name] = decode_discrete_repetitions(
                        pooled_features,
                        theta,
                        config,
                        device=args.device,
                        dtype=args.dtype,
                        seed=decode_seed,
                        return_trialwise=True,
                        strata=session_strata,
                    )

            dec_cue, dist_cue = decoded["cue"][:2]
            dec_uncue, dist_uncue = decoded["uncue"][:2]
            first = 0
            for entry in session_results:
                stop = first + len(entry["trial_ids"])
                entry["decoded"] = {
                    "cue": decoded["cue"][2][first:stop],
                    "uncue": decoded["uncue"][2][first:stop],
                }
                first = stop
                del entry["features"], entry["behavior"]
            del pooled_features
        else:
            for entry in session_results:
                session_decoded = {}
                for name, column in (("cue", "cue_item"), ("uncue", "uncue_item")):
                    theta = item_theta_rad(entry["behavior"][column].to_numpy())
                    decode_seed = (
                        args.seed + 1000 * subject + 100 * entry["session"]
                        + (1 if name == "uncue" else 0)
                    )
                    if args.cv_method == "loocv":
                        print(
                            f" session {entry['session']} decoding {name}: "
                            "6-category discrete LOOCV",
                            flush=True,
                        )
                        session_decoded[name] = decode_discrete_loocv(
                            entry["features"],
                            theta,
                            config,
                            device=args.device,
                            dtype=args.dtype,
                            seed=decode_seed,
                            return_trialwise=True,
                        )
                    else:
                        print(
                            f" session {entry['session']} decoding {name}: "
                            f"6-category hard k-fold, n_folds={args.n_folds}, "
                            f"reps={args.reps}",
                            flush=True,
                        )
                        session_decoded[name] = decode_discrete_repetitions(
                            entry["features"],
                            theta,
                            config,
                            device=args.device,
                            dtype=args.dtype,
                            seed=decode_seed,
                            return_trialwise=True,
                        )
                entry["decoded_full"] = session_decoded
                entry["decoded"] = {
                    name: session_decoded[name][2] for name in ("cue", "uncue")
                }
                del entry["features"], entry["behavior"]

            dec_cue = np.mean(
                [entry["decoded_full"]["cue"][0] for entry in session_results],
                axis=0,
            )
            dec_uncue = np.mean(
                [entry["decoded_full"]["uncue"][0] for entry in session_results],
                axis=0,
            )
            dist_cue = np.mean(
                [entry["decoded_full"]["cue"][1] for entry in session_results],
                axis=0,
            )
            dist_uncue = np.mean(
                [entry["decoded_full"]["uncue"][1] for entry in session_results],
                axis=0,
            )
            for entry in session_results:
                del entry["decoded_full"]

        save_kwargs = dict(
            time_dec=time_dec,
            dec_cue=dec_cue,
            dec_uncue=dec_uncue,
            dist_cue=dist_cue,
            dist_uncue=dist_uncue,
            channels=np.asarray(CHANNELS),
            alpha_band=np.asarray([args.alpha_low, args.alpha_high]),
            filter_method=args.filter_method,
            mtm_window_mode=args.mtm_window_mode,
            mtm_cycles=args.mtm_cycles,
            mtm_fixed_window_ms=args.mtm_fixed_window_ms,
            mtm_frequency_step=args.mtm_frequency_step,
            n_folds=args.n_folds,
            reps=(1 if args.cv_method == "loocv" else args.reps),
            cv_method=args.cv_method,
            decoding_feature_mode=args.decoding_feature_mode,
            session_pooling=(
                "pooled_session_category_stratified"
                if args.pool_sessions else "separate_sessions"
            ),
            session_feature_standardization=(
                "train_fold_zscore_per_feature_time"
                if args.pool_sessions else "none"
            ),
            template_category_balancing="random_subsample_to_fold_minimum",
            config=np.array(config.__dict__, dtype=object),
        )
        for entry in session_results:
            sess = entry["session"]
            save_kwargs[f"sess{sess}_trial_ids"] = entry["trial_ids"]
            save_kwargs[f"sess{sess}_trial_dec_cue"] = entry["decoded"]["cue"]
            save_kwargs[f"sess{sess}_trial_dec_uncue"] = entry["decoded"]["uncue"]
            save_kwargs[f"sess{sess}_cue_loc"] = entry["cue_loc"]
            save_kwargs[f"sess{sess}_rt"] = entry["rt"]
            save_kwargs[f"sess{sess}_acc"] = entry["acc"]

        temporary_output = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_output, **save_kwargs)
        temporary_output.replace(output)
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
