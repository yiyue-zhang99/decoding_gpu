#!/usr/bin/env python3
"""Pooled-session alpha time-cross decoding for Guven data.

Cue and uncued-item orientations are six discrete categories.  Accordingly,
this script uses the discrete half-cosine-basis K-fold time-generalization
decoder rather than Micheal's continuous-angle decoder. Sessions are pooled
with session-stratified cross-validation and output matrices are train-time x
test-time. Optional MNE resampling is performed before alpha extraction.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from mne.filter import resample as mne_resample


PROJECT_DIR = Path("/home/dilay/project2/tw")
DECODER_DIR = PROJECT_DIR / "functions" / "decoding" / "decoding_gpu"
GUVEN_DIR = PROJECT_DIR / "travelling_waves" / "tw" / "fft" / "guven"
GUVEN_CORR_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(GUVEN_DIR))
sys.path.insert(0, str(GUVEN_CORR_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helper import DecodeConfig, make_sliding_features  # noqa: E402
from mahal_discrete_basis_kfold_torch_time_cross_generalization import (  # noqa: E402
    decode_discrete_basis_time_cross_repetitions,
)
from guven_behavior import item_theta_rad  # noqa: E402
from guven_decoding_gpu import (  # noqa: E402
    CHANNELS,
    EPOCH_DIR,
    alpha_power,
    load_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guven alpha decoding: pooled sessions, timepoint features, "
            "discrete-basis K-fold time generalization"
        )
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 31)))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--step-ms",
        type=float,
        default=10.0,
        help=(
            "Spacing between decoded single-sample timepoints. The default "
            "10 ms keeps every sample after the default 100-Hz resampling."
        ),
    )
    parser.add_argument("--smooth-ms", type=float, default=0.0)
    parser.add_argument("--toi", type=float, nargs=2, default=(0.75, 1.85))
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=100.0,
        help=(
            "Resample raw EEG with MNE before alpha filtering (default: "
            "100 Hz). Pass 0 to retain the original sampling rate."
        ),
    )

    parser.add_argument("--alpha-low", type=float, default=8.0)
    parser.add_argument("--alpha-high", type=float, default=12.0)
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir", "mtmconvol"],
        default="fir",
    )
    parser.add_argument(
        "--mtm-window-mode", choices=["adaptive", "fixed"], default="adaptive"
    )
    parser.add_argument("--mtm-cycles", type=float, default=5.0)
    parser.add_argument("--mtm-fixed-window-ms", type=float, default=500.0)
    parser.add_argument("--mtm-frequency-step", type=float, default=1.0)
    parser.add_argument("--filter-batch-trials", type=int, default=32)

    parser.add_argument("--basis-exponent", type=float, default=5.0)
    parser.add_argument("--basis-reps", type=int, default=100)
    parser.add_argument(
        "--pool-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pool available sessions using session x category stratified "
            "cross-validation (default). --no-pool-sessions is intentionally "
            "unsupported by this time-cross workflow."
        ),
    )
    parser.add_argument(
        "--trialwise-only",
        action="store_true",
        help="Save trialwise cue/uncue matrices but omit mean and distance maps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_is_complete(
    path: Path, sessions: list[int], trialwise_only: bool = False
) -> bool:
    if not path.is_file():
        return False
    required = {"time_dec", "train_time", "test_time"}
    if not trialwise_only:
        required |= {"dec_cue", "dec_uncue", "dist_cue", "dist_uncue"}
    for session in sessions:
        required |= {
            f"sess{session}_trial_ids",
            f"sess{session}_trial_dec_cue",
            f"sess{session}_trial_dec_uncue",
        }
    try:
        with np.load(path, allow_pickle=True) as saved:
            return required.issubset(saved.files)
    except (OSError, ValueError, EOFError):
        return False


def prepare_session(
    subject: int,
    session: int,
    args: argparse.Namespace,
    config: DecodeConfig,
) -> dict:
    """Load one session and create timepoint features without decoding it."""
    print(f"subject {subject:02d}, session {session}: preparing", flush=True)
    session_data = load_session(subject, session)
    time = np.asarray(session_data["time"], dtype=np.float64)
    original_hz = float(1.0 / np.median(np.diff(time)))
    hz = original_hz

    raw = np.asarray(session_data["data"], dtype=np.float32)
    if args.resample_hz > 0:
        target_hz = float(args.resample_hz)
        if target_hz > hz * (1 + 1e-9):
            raise ValueError(
                f"--resample-hz={target_hz:g} exceeds original {hz:g} Hz"
            )
        if args.alpha_high >= target_hz / 2:
            raise ValueError(
                f"alpha-high must be below the resampled Nyquist frequency "
                f"({target_hz / 2:g} Hz)"
            )
        old_n_times = raw.shape[-1]
        raw = mne_resample(
            raw.astype(np.float64, copy=False),
            up=target_hz,
            down=hz,
            npad="auto",
            verbose=False,
        ).astype(np.float32, copy=False)
        time = time[0] + np.arange(raw.shape[-1], dtype=np.float64) / target_hz
        hz = target_hz
        print(
            f"  MNE resample before alpha FIR: {original_hz:.3f} -> "
            f"{hz:.3f} Hz; {old_n_times} -> {raw.shape[-1]} samples",
            flush=True,
        )
    else:
        print(f"  retaining original sampling rate: {hz:.3f} Hz", flush=True)

    power = alpha_power(
        raw,
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
    del raw, session_data["data"]

    # Genuine single-sample features: no sliding window or temporal feature
    # concatenation. Optionally retain one original sample every --step-ms.
    features, time_dec = make_sliding_features(power, time, config, mode="timepoint")
    del power
    if args.step_ms <= 0:
        raise ValueError("--step-ms must be positive")
    native_step_ms = float(np.median(np.diff(time_dec)) * 1000)
    if args.step_ms < native_step_ms * (1 - 1e-6):
        raise ValueError(
            f"--step-ms={args.step_ms:g} is finer than the available "
            f"{native_step_ms:g} ms/sample grid"
        )
    if args.step_ms > native_step_ms * (1 + 1e-6):
        requested = np.arange(
            time_dec[0],
            time_dec[-1] + args.step_ms / 2000,
            args.step_ms / 1000,
        )
        indices = np.unique(
            [int(np.argmin(np.abs(time_dec - value))) for value in requested]
        )
        features = features[:, :, indices]
        time_dec = time_dec[indices]
    print(f"  timepoint feature shape {features.shape}", flush=True)
    print(
        f"  decoding time {time_dec[0]:.3f}-{time_dec[-1]:.3f} s, "
            f"{time_dec.size} selected single-sample timepoints, "
        f"{np.median(np.diff(time_dec)) * 1000:.3f} ms/sample",
        flush=True,
    )

    behavior = session_data["behavior"]
    return {
        "session": session,
        "features": features,
        "time_dec": time_dec,
        "trial_ids": session_data["trial_ids"],
        "cue_loc": behavior["cue_loc"].to_numpy(),
        "rt": behavior["rt"].to_numpy(),
        "acc": behavior["acc"].to_numpy(),
        "behavior": behavior,
        "sfreq": hz,
        "original_sfreq": original_hz,
    }


def main() -> None:
    args = parse_args()
    if args.alpha_low <= 0 or args.alpha_high <= args.alpha_low:
        raise ValueError("alpha band must satisfy 0 < alpha-low < alpha-high")
    if args.basis_exponent <= 0 or args.basis_reps <= 0:
        raise ValueError("basis-exponent and basis-reps must be positive")
    if args.step_ms <= 0:
        raise ValueError("--step-ms must be positive")
    if len(args.toi) != 2 or args.toi[0] >= args.toi[1]:
        raise ValueError("--toi must contain increasing start and stop times")
    if not args.pool_sessions:
        raise ValueError(
            "This script implements pooled-session time-cross decoding; "
            "remove --no-pool-sessions"
        )
    if not np.isfinite(args.resample_hz) or args.resample_hz < 0:
        raise ValueError("--resample-hz must be non-negative and finite")

    output_dir = args.output_dir or (
        PROJECT_DIR
        / "results"
        / (
            f"guven_alpha_time_cross_discrete_basis_kfold_{args.filter_method}_"
            "timepoint_pooled_sessions"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DecodeConfig(
        n_folds=args.n_folds,
        reps=args.basis_reps,
        step_ms=args.step_ms,
        smooth_ms=args.smooth_ms,
        toi=tuple(args.toi),
    )

    for subject in args.subjects:
        output = output_dir / f"subject_{subject:02d}_alpha_time_cross_torch.npz"
        if not args.overwrite and output_is_complete(
            output, args.sessions, args.trialwise_only
        ):
            print(f"skipping completed {output}", flush=True)
            continue
        if output.exists():
            print(f"recomputing incomplete output {output}", flush=True)

        missing = [
            session
            for session in args.sessions
            if not (EPOCH_DIR / f"sub{subject:02d}_session{session}.fif").is_file()
        ]
        if missing:
            print(f"[skip] subject {subject:02d}: missing sessions {missing}", flush=True)
            continue

        sessions = [
            prepare_session(subject, session, args, config)
            for session in args.sessions
        ]
        reference_time = sessions[0]["time_dec"]
        if any(not np.array_equal(entry["time_dec"], reference_time) for entry in sessions[1:]):
            raise ValueError("Decoding time vectors differ between sessions")

        pooled_features = np.concatenate(
            [entry["features"] for entry in sessions], axis=0
        )
        session_strata = np.concatenate(
            [
                np.full(len(entry["trial_ids"]), entry["session"], dtype=int)
                for entry in sessions
            ]
        )
        decoded = {}
        for offset, (name, column) in enumerate(
            (("cue", "cue_item"), ("uncue", "uncue_item"))
        ):
            theta = item_theta_rad(
                np.concatenate(
                    [entry["behavior"][column].to_numpy() for entry in sessions]
                )
            )
            decode_seed = args.seed + 1000 * subject + offset
            print(
                f"subject {subject:02d}, pooled decoding {name}: "
                f"{pooled_features.shape[0]} trials from {len(sessions)} sessions, "
                f"6-category basis, {config.n_folds}-fold, "
                f"half-cosine^{args.basis_exponent:g}, "
                f"{config.reps} repetitions; folds stratified by session x "
                "category",
                flush=True,
            )
            decoded[name] = decode_discrete_basis_time_cross_repetitions(
                pooled_features,
                theta,
                config,
                basis_exponent=args.basis_exponent,
                device=args.device,
                dtype=args.dtype,
                seed=decode_seed,
                return_trialwise=True,
                strata=session_strata,
            )
        del pooled_features

        first = 0
        for entry in sessions:
            stop = first + len(entry["trial_ids"])
            entry["trial_dec_cue"] = decoded["cue"][2][first:stop]
            entry["trial_dec_uncue"] = decoded["uncue"][2][first:stop]
            del entry["features"], entry["behavior"]
            first = stop

        payload = {
            "time_dec": reference_time,
            "train_time": reference_time,
            "test_time": reference_time,
            "channels": np.asarray(CHANNELS),
            "alpha_band": np.asarray([args.alpha_low, args.alpha_high]),
            "filter_method": args.filter_method,
            "mtm_window_mode": args.mtm_window_mode,
            "mtm_cycles": args.mtm_cycles,
            "mtm_fixed_window_ms": args.mtm_fixed_window_ms,
            "mtm_frequency_step": args.mtm_frequency_step,
            "angle_method": "discrete-basis-kfold-time-cross-generalization",
            "n_folds": args.n_folds,
            "basis_exponent": args.basis_exponent,
            "basis_reps": args.basis_reps,
            "basis_function": "row_normalized_(0.5+0.5*cos(theta-mu))^exponent",
            "decoding_feature_mode": "timepoint",
            "original_sfreq": sessions[0]["original_sfreq"],
            "resample_hz_requested": (
                np.nan if args.resample_hz == 0 else args.resample_hz
            ),
            "effective_sfreq": sessions[0]["sfreq"],
            "timepoint_stride_samples": int(
                round(args.step_ms * sessions[0]["sfreq"] / 1000.0)
            ),
            "session_pooling": "pooled_session_category_stratified",
            "session_feature_standardization": "train_fold_zscore_per_feature_time",
            "time_axis_order": "train_time,test_time",
            "storage_mode": "trialwise_only" if args.trialwise_only else "full",
            "config": np.array(config.__dict__, dtype=object),
        }
        if not args.trialwise_only:
            payload.update(
                {
                    "dec_cue": decoded["cue"][0],
                    "dec_uncue": decoded["uncue"][0],
                    "dist_cue": decoded["cue"][1],
                    "dist_uncue": decoded["uncue"][1],
                }
            )
        for entry in sessions:
            session = entry["session"]
            payload.update(
                {
                    f"sess{session}_trial_ids": entry["trial_ids"],
                    f"sess{session}_trial_dec_cue": entry["trial_dec_cue"],
                    f"sess{session}_trial_dec_uncue": entry["trial_dec_uncue"],
                    f"sess{session}_cue_loc": entry["cue_loc"],
                    f"sess{session}_rt": entry["rt"],
                    f"sess{session}_acc": entry["acc"],
                }
            )

        temporary_output = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_output, **payload)
        temporary_output.replace(output)
        print(f"saved {output}", flush=True)

        del sessions, payload, decoded, session_strata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
