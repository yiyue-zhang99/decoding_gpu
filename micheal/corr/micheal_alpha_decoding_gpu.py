#!/usr/bin/env python3
"""Alpha-power decoding with the same window features as raw voltage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import (
    butter,
    fftconvolve,
    filtfilt,
    firwin,
    hilbert,
    sosfiltfilt,
)


PROJECT_DIR = Path("/home/dilay/project2/tw")
sys.path.insert(
    0, str(PROJECT_DIR / "functions" / "decoding" / "decoding_gpu")
)

from helper import (  # noqa: E402
    DecodeConfig,
    make_sliding_features,
)
CHANNELS = [ "PZ", "POZ", "OZ", "P1", "PO3", "O1", "P7", "PO7", "P5", "P3", "P2", "PO4", "O2", "P8", "PO8", "P4", "P6", ]
# CHANNELS = [
#     # # Midline
#     "OZ", "POZ", "PZ", "CPZ", "CZ", "FCZ", "FZ",

#     # Left hemisphere
#     "O1", "PO3", "P1", "CP1", "C1", "FC1", "F1",
#     "PO7", "P5", "CP3", "C3", "P3", "FC3", "F3",

#     # Right hemisphere
#     "O2", "PO4", "P2", "CP2", "C2", "FC2", "F2",
#     "PO8", "P6", "CP4", "C4", "P4", "FC4", "F4",
# ]
EOG_CHANNELS = {"VEOG", "HEOG"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1)
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
        "--angle-method",
        choices=["kernel", "hard-loocv", "basis-kfold"],
        default="kernel",
        help=(
            "Continuous kernel, trial-relative hard LOOCV, or the 16-bin "
            "repeated 8-fold half-cosine-basis method"
        ),
    )
    parser.add_argument(
        "--kernel-width-deg",
        type=float,
        default=15.0,
        help="Gaussian SD in original 0-180 orientation degrees",
    )
    parser.add_argument(
        "--kernel-templates",
        type=int,
        default=16,
        help="Number of evaluation templates; trials are not binned into them",
    )
    parser.add_argument(
        "--kernel-cv",
        choices=["kfold", "loocv"],
        default="kfold",
        help="Cross-validation scheme for --angle-method kernel",
    )
    parser.add_argument(
        "--hard-width-deg",
        type=float,
        default=15.0,
        help="Hard-bin half-width in original 0-180 orientation degrees",
    )
    parser.add_argument(
        "--hard-templates",
        type=int,
        default=12,
        help="Number of trial-relative hard templates for LOOCV",
    )
    parser.add_argument("--basis-bins", type=int, default=16)
    parser.add_argument("--basis-exponent", type=float, default=15.0)
    parser.add_argument("--basis-orientation-spaces", type=int, default=8)
    parser.add_argument("--basis-reps", type=int, default=100)
    parser.add_argument(
    "--all-eeg",
    action="store_true",
    help="Use all EEG channels except VEOG and HEOG",
)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--span-ms", type=float, default=10.0)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument(
        "--decoding-feature-mode",
        choices=["sliding-window", "timepoint"],
        default="sliding-window",
        help=(
            "Use the current temporally concatenated sliding-window features "
            "or only the channels at each individual decoding timepoint"
        ),
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
    parser.add_argument(
        "--mtm-cycles",
        type=float,
        default=5.0,
        help="For adaptive mtmconvol: number of cycles per frequency",
    )
    parser.add_argument(
        "--mtm-fixed-window-ms",
        type=float,
        default=500.0,
        help="For fixed mtmconvol: common Fourier window length",
    )
    parser.add_argument(
        "--mtm-frequency-step",
        type=float,
        default=1,
        help="Frequency spacing in Hz for mtmconvol",
    )
    parser.add_argument("--filter-batch-trials", type=int, default=32)
    parser.add_argument("--smooth-ms", type=float, default=0.0)
    parser.add_argument("--toi", type=float, nargs=2, default=(-0.05, 6.0))
    parser.add_argument(
        "--pool-sessions",
        action="store_true",
        help=(
            "Pool both sessions for any decoding method; train-only "
            "feature standardisation remain stratified by session"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute subjects whose complete output file already exists",
    )
    
    return parser.parse_args()


def load_session(path: Path) -> dict:
    return loadmat(
        path, variable_names=["ft_mem"], simplify_cells=True
    )["ft_mem"]


def channel_indices(
    labels: np.ndarray,
    all_eeg: bool = False,
) -> np.ndarray:
    available = {
        str(label).strip().upper(): index
        for index, label in enumerate(np.asarray(labels).reshape(-1))
    }

    if all_eeg:
        return np.asarray( [ index for name, index in available.items() if name not in EOG_CHANNELS ], dtype=int, )
    missing = [name for name in CHANNELS if name not in available]
    if missing:
        raise ValueError(f"Missing requested channels: {missing}")

    return np.asarray( [available[name] for name in CHANNELS], dtype=int, )



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

    # Equivalent MNE IIR implementation (reference only; not executed):
    #
    # filtered = mne.filter.filter_data(
    #     raw_batch.astype(np.float64, copy=False),
    #     sfreq=hz,
    #     l_freq=low,
    #     h_freq=high,
    #     method="iir",
    #     iir_params={
    #         "order": 4,
    #         "ftype": "butter",
    #         "output": "sos",
    #     },
    #     phase="zero",
    #     pad="reflect_limited",
    #     verbose=False,
    # )


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
        coefficients = firwin( numtaps, [low, high], pass_zero=False, fs=hz, window="hamming", )
        print( f"  FIR: {numtaps} taps, {numtaps / hz:.3f} s " f"(3 cycles at {low:g} Hz)", flush=True, )
    elif method == "iir":
        coefficients = butter( 4, [low, high], btype="bandpass", fs=hz, output="sos" )
        print( f"  IIR: 4th-order Butterworth SOS, {low:g}-{high:g} Hz", flush=True, )
    else:
        raise ValueError(f"Unknown filter method: {method}")

    amplitude = np.empty(raw.shape, dtype=np.float32)

    for first in range(0, raw.shape[0], batch_trials):
        stop = min(first + batch_trials, raw.shape[0])

        raw_batch = raw[first:stop]
        if method == "fir":
            filtered = filtfilt( coefficients, [1.0], raw_batch, axis=-1 )
        else:
            filtered = sosfiltfilt( coefficients, raw_batch, axis=-1 )

        analytic = hilbert(filtered, axis=-1)

        amplitude[first:stop] = np.square(np.abs(analytic)).astype(np.float32, copy=False)
        print( f"  alpha extraction trials {first + 1}-{stop}/{raw.shape[0]}", flush=True, )
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

    frequencies = np.arange(
        low,
        high + frequency_step / 2,
        frequency_step,
        dtype=np.float64,
    )
    frequencies = frequencies[frequencies <= high + 1e-12]
    if frequencies.size == 0:
        raise ValueError("No mtmconvol frequencies fall inside the alpha band")

    power_sum = np.zeros(raw.shape, dtype=np.float32)
    print(
        f"  MTMCONVOL: Hanning taper, frequencies="
        f"{np.array2string(frequencies, precision=3)}, "
        f"window={window_mode}",
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
        relative_time = (
            np.arange(window_samples) - (window_samples - 1) / 2
        ) / hz
        kernel = (
            taper * np.exp(2j * np.pi * frequency * relative_time)
        )[None, None, :]
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
                raw[first:stop],
                ((0, 0), (0, 0), (pad_left, pad_right)),
                mode="reflect",
            )
            fourier = fftconvolve(
                padded,
                kernel,
                mode="valid",
                axes=-1,
            )
            power_sum[first:stop] += np.square(np.abs(fourier)).astype(
                np.float32,
                copy=False,
            )
        print(
            f"    completed {frequency:g} Hz for {raw.shape[0]} trials",
            flush=True,
        )

    power_sum /= frequencies.size
    return power_sum


def output_is_complete(path: Path) -> bool:
    """Return True only for a readable output containing all expected arrays."""
    if not path.is_file():
        return False

    required = {
        "time_dec",
        "dec_early",
        "dec_late",
        "dist_early",
        "dist_late",
        "sess1_trial_ids",
        "sess1_trial_dec_early",
        "sess1_trial_dec_late",
        "sess2_trial_ids",
        "sess2_trial_dec_early",
        "sess2_trial_dec_late",
    }
    try:
        with np.load(path, allow_pickle=True) as saved:
            return required.issubset(saved.files)
    except (OSError, ValueError, EOFError):
        return False


def main() -> None:
    args = parse_args()
    # Import only the selected decoder. Besides reducing startup overhead,
    # this lets other scripts reuse the alpha extraction helpers without
    # loading unrelated decoding implementations.
    if args.angle_method == "kernel":
        from mahal_theta_kernel_torch import (
            decode_kernel_loocv,
            decode_kernel_repetitions,
        )
    elif args.angle_method == "hard-loocv":
        from mahal_theta_hard_loocv_torch import mahal_hard_loocv
    else:
        from mahal_theta_basis_kfold_torch import decode_basis_kfold_repetitions

    data_dir = PROJECT_DIR / "Data" / "Micheal_Data_exp2"
    if args.angle_method == "basis-kfold":
        output_name = (
            f"michael_alpha_decoding_basis_kfold_{args.filter_method}"
        )
    elif args.angle_method == "hard-loocv":
        if args.filter_method == "mtmconvol":
            if args.mtm_window_mode == "adaptive":
                mtm_tag = f"adaptive"
            else:
                mtm_tag = f"fixed"
            output_name = (
                "michael_alpha_decoding_hard_loocv_mtmconvol_"
                f"{mtm_tag}"
            )
        else:
            output_name = (
                f"michael_alpha_decoding_hard_loocv_"
                f"{args.filter_method}"
            )
    else:
        output_name = (
            f"michael_alpha_decoding_kernel_{args.filter_method}"
        )
    if args.decoding_feature_mode == "timepoint":
        output_name += "_timepoint"
    if args.pool_sessions:
        output_name += "_pooled_sessions"
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
        if not args.overwrite and output_is_complete(output):
            print(f"skipping completed {output}", flush=True)
            continue
        if output.exists():
            print(f"recomputing incomplete output {output}", flush=True)

        session_results = []
        for session in (1, 2):
            print(f"subject {subject:02d}, session {session}", flush=True)
            input_file = data_dir / (
                f"MemImp3_mem_whole_sess{session}_{subject}.mat"
            )
            ft_mem = load_session(input_file)
            labels = np.asarray(ft_mem["label"]).reshape(-1)
            indices = channel_indices( labels, all_eeg=args.all_eeg, )
            selected_channels = labels[indices]
            n_trials = ft_mem["trial"].shape[0]
            bad = np.asarray(ft_mem["bad_trials_mem"], dtype=int).reshape(-1) - 1
            good = np.setdiff1d(np.arange(n_trials), bad)
            time = np.asarray(ft_mem["time"], dtype=np.float64).reshape(-1)
            hz = round(1 / np.median(np.diff(time)))
            raw = np.asarray(
                ft_mem["trial"][np.ix_(good, indices, np.arange(time.size))],
                dtype=np.float32,
            )
            results = np.asarray(ft_mem["Results"])[good]
            del ft_mem

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
            del raw

            # This is deliberately identical to voltage feature construction:
            # subtract each channel's 100-ms temporal mean, average within
            # 10-ms segments, and concatenate channel x segment features.
            features, time_dec = make_sliding_features(
                power,
                time,
                config,
                mode=args.decoding_feature_mode,
            )
            del power
            print(
                f"  feature mode {args.decoding_feature_mode}; "
                f"shape {features.shape}",
                flush=True,
            )

            decoded = {}
            if not args.pool_sessions:
              for name, column in (("early", 5), ("late", 6)):
                continuous_theta = results[:, column] * 2
                decode_seed = (
                    args.seed + 1000 * subject + 100 * session + column
                )
                if args.angle_method == "kernel":
                    # The orientation is doubled to span a full circle, so
                    # the kernel SD must be doubled in circular-angle space.
                    kernel_sigma = np.deg2rad(
                        2 * args.kernel_width_deg
                    )
                    print(
                        f" decoding {name}: continuous angles, "
                        f"kernel SD={args.kernel_width_deg:g} orientation deg, "
                        f"{args.kernel_templates} soft templates",
                        flush=True,
                    )
                    if args.kernel_cv == "loocv":
                        decoded[name] = decode_kernel_loocv(
                            features,
                            continuous_theta,
                            config,
                            kernel_sigma=kernel_sigma,
                            n_templates=args.kernel_templates,
                            device=args.device,
                            dtype=args.dtype,
                            return_trialwise=True,
                        )
                    else:
                        decoded[name] = decode_kernel_repetitions(
                            features,
                            continuous_theta,
                            config,
                            kernel_sigma=kernel_sigma,
                            n_templates=args.kernel_templates,
                            device=args.device,
                            dtype=args.dtype,
                            seed=decode_seed,
                            return_trialwise=True,
                        )
                elif args.angle_method == "hard-loocv":
                    # Orientations were doubled to span a full circle, so the
                    # requested half-width must also be doubled.
                    bin_half_width = np.deg2rad(
                        2 * args.hard_width_deg
                    )
                    print(
                        f" decoding {name}: MATLAB-style hard relative bins, "
                        f"LOOCV, half-width={args.hard_width_deg:g} "
                        f"orientation deg, {args.hard_templates} templates",
                        flush=True,
                    )
                    decoded[name] = mahal_hard_loocv(
                        features,
                        continuous_theta,
                        config,
                        bin_half_width=bin_half_width,
                        n_templates=args.hard_templates,
                        device=args.device,
                        dtype=args.dtype,
                        return_trialwise=True,
                    )
                elif args.angle_method == "basis-kfold":
                    print(
                        f" decoding {name}: {args.basis_bins} hard bins, "
                        f"stratified {config.n_folds}-fold, "
                        f"half-cosine^{args.basis_exponent:g}, "
                        f"{args.basis_orientation_spaces} orientation spaces, "
                        f"{args.basis_reps} repetitions per space",
                        flush=True,
                    )
                    decoded[name] = decode_basis_kfold_repetitions(
                        features,
                        continuous_theta,
                        config,
                        n_bins=args.basis_bins,
                        n_orientation_spaces=args.basis_orientation_spaces,
                        basis_exponent=args.basis_exponent,
                        repetitions=args.basis_reps,
                        device=args.device,
                        dtype=args.dtype,
                        seed=decode_seed,
                        return_trialwise=True,
                    )
            session_results.append(
                {
                    "decoded": decoded,
                    "features": features if args.pool_sessions else None,
                    "results": results if args.pool_sessions else None,
                    "session": session,
                    # One-based original row numbers, matching MATLAB trial
                    # numbering and the trial axis retained in TW pickle files.
                    "trial_ids": good + 1,
                }
            )
            if not args.pool_sessions:
                del features

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
            pooled_decoded = {}
            for name, column in (("early", 5), ("late", 6)):
                continuous_theta = np.concatenate(
                    [entry["results"][:, column] for entry in session_results]
                ) * 2
                if args.angle_method == "kernel":
                    kernel_sigma = np.deg2rad(2 * args.kernel_width_deg)
                    if args.kernel_cv == "loocv":
                        print(
                            f" pooled decoding {name}: continuous Gaussian "
                            f"kernel LOOCV, kernel SD="
                            f"{args.kernel_width_deg:g} orientation deg, "
                            f"{args.kernel_templates} soft templates",
                            flush=True,
                        )
                        pooled_decoded[name] = decode_kernel_loocv(
                            pooled_features,
                            continuous_theta,
                            config,
                            kernel_sigma=kernel_sigma,
                            n_templates=args.kernel_templates,
                            device=args.device,
                            dtype=args.dtype,
                            return_trialwise=True,
                            strata=session_strata,
                        )
                    else:
                        print(
                            f" pooled decoding {name}: continuous kernel, "
                            f"session-stratified {config.n_folds}-fold, "
                            f"kernel SD={args.kernel_width_deg:g} orientation deg, "
                            f"{args.kernel_templates} soft templates",
                            flush=True,
                        )
                        pooled_decoded[name] = decode_kernel_repetitions(
                            pooled_features,
                            continuous_theta,
                            config,
                            kernel_sigma=kernel_sigma,
                            n_templates=args.kernel_templates,
                            device=args.device,
                            dtype=args.dtype,
                            seed=args.seed + 1000 * subject + column,
                            return_trialwise=True,
                            strata=session_strata,
                        )
                elif args.angle_method == "hard-loocv":
                    bin_half_width = np.deg2rad(2 * args.hard_width_deg)
                    print(
                        f" pooled decoding {name}: MATLAB-style hard "
                        f"relative-bin LOOCV, half-width="
                        f"{args.hard_width_deg:g} orientation deg, "
                        f"{args.hard_templates} templates",
                        flush=True,
                    )
                    pooled_decoded[name] = mahal_hard_loocv(
                        pooled_features,
                        continuous_theta,
                        config,
                        bin_half_width=bin_half_width,
                        n_templates=args.hard_templates,
                        device=args.device,
                        dtype=args.dtype,
                        return_trialwise=True,
                        strata=session_strata,
                    )
                else:
                    print(
                        f" pooled decoding {name}: {args.basis_bins} hard bins, "
                        f"session x bin stratified {config.n_folds}-fold, "
                        f"half-cosine^{args.basis_exponent:g}, "
                        f"{args.basis_orientation_spaces} orientation spaces, "
                        f"{args.basis_reps} repetitions per space",
                        flush=True,
                    )
                    pooled_decoded[name] = decode_basis_kfold_repetitions(
                        pooled_features,
                        continuous_theta,
                        config,
                        n_bins=args.basis_bins,
                        n_orientation_spaces=args.basis_orientation_spaces,
                        basis_exponent=args.basis_exponent,
                        repetitions=args.basis_reps,
                        device=args.device,
                        dtype=args.dtype,
                        seed=args.seed + 1000 * subject + column,
                        return_trialwise=True,
                        strata=session_strata,
                    )
            first = 0
            for entry in session_results:
                stop = first + len(entry["trial_ids"])
                entry["decoded"] = {
                    name: (
                        pooled_decoded[name][0],
                        pooled_decoded[name][1],
                        pooled_decoded[name][2][first:stop],
                    )
                    for name in ("early", "late")
                }
                first = stop
                del entry["features"], entry["results"]
            del pooled_features

        if args.pool_sessions:
            dec_early, dist_early = pooled_decoded["early"][:2]
            dec_late, dist_late = pooled_decoded["late"][:2]
        else:
            dec_early = np.mean(
                [x["decoded"]["early"][0] for x in session_results], axis=0
            )
            dec_late = np.mean(
                [x["decoded"]["late"][0] for x in session_results], axis=0
            )
            dist_early = np.mean(
                [x["decoded"]["early"][1] for x in session_results], axis=0
            )
            dist_late = np.mean(
                [x["decoded"]["late"][1] for x in session_results], axis=0
            )
        temporary_output = output.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_output,
            time_dec=time_dec,
            dec_early=dec_early,
            dec_late=dec_late,
            dist_early=dist_early,
            dist_late=dist_late,
            channels=selected_channels,
            alpha_band=np.asarray([args.alpha_low, args.alpha_high]),
            filter_method=args.filter_method,
            mtm_window_mode=args.mtm_window_mode,
            mtm_cycles=args.mtm_cycles,
            mtm_fixed_window_ms=args.mtm_fixed_window_ms,
            mtm_frequency_step=args.mtm_frequency_step,
            angle_method=args.angle_method,
            kernel_width_deg=args.kernel_width_deg,
            kernel_templates=args.kernel_templates,
            kernel_cv=args.kernel_cv,
            hard_width_deg=args.hard_width_deg,
            hard_templates=args.hard_templates,
            basis_bins=args.basis_bins,
            basis_hard_bin_width_deg=180.0 / args.basis_bins,
            basis_hard_bin_half_width_deg=90.0 / args.basis_bins,
            basis_exponent=args.basis_exponent,
            basis_orientation_spaces=args.basis_orientation_spaces,
            basis_reps=args.basis_reps,
            basis_template_convolution="row_normalized_half_cosine",
            basis_training_bin_balancing="random_subsample_to_fold_minimum",
            decoding_feature_mode=args.decoding_feature_mode,
            session_pooling=(
                "pooled_session_stratified" if args.pool_sessions else "separate"
            ),
            session_feature_standardization=(
                "train_fold_zscore_per_feature_time"
                if args.pool_sessions else "none"
            ),
            config=np.array(config.__dict__, dtype=object),
            sess1_trial_ids=session_results[0]["trial_ids"],
            sess1_trial_dec_early=session_results[0]["decoded"]["early"][2],
            sess1_trial_dec_late=session_results[0]["decoded"]["late"][2],
            sess2_trial_ids=session_results[1]["trial_ids"],
            sess2_trial_dec_early=session_results[1]["decoded"]["early"][2],
            sess2_trial_dec_late=session_results[1]["decoded"]["late"][2],
        )
        temporary_output.replace(output)
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
