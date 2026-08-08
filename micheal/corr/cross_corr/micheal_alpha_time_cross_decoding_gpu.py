#!/usr/bin/env python3
"""Separate-session, timepoint alpha time-cross decoding for Micheal data.

Alpha power is extracted with the same FIR/Hilbert, IIR/Hilbert, or
Hanning-tapered MTMCONVOL implementations as ``micheal_alpha_decoding_gpu``.
The only decoder is the shifted-bin half-cosine basis K-fold time-cross
generalization decoder. Output matrices are train-time x test-time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import gc
import torch
import numpy as np
from mne.filter import resample as mne_resample


PROJECT_DIR = Path("/home/dilay/project2/tw")
DECODER_DIR = PROJECT_DIR / "functions" / "decoding" / "decoding_gpu"
CORR_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CORR_DIR))

from helper import DecodeConfig, make_sliding_features  # noqa: E402
from mahal_theta_basis_kfold_torch_time_cross_generalization import (  # noqa: E402
    decode_basis_kfold_time_cross_repetitions,
)

# Reuse the alpha extraction and data/channel helpers verbatim so this script
# cannot silently diverge from the established FIR, IIR, and MTMCONVOL paths.
from micheal_alpha_decoding_gpu import (  # noqa: E402
    alpha_power,
    channel_indices,
    load_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Micheal alpha decoding: separate sessions, timepoint features, "
            "basis K-fold time-cross generalization only"
        )
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--all-eeg", action="store_true")
    parser.add_argument(
        "--step-ms",
        type=float,
        default=50.0,
        help=(
            "Spacing between decoded timepoints. The feature at each selected "
            "time is still one sample on the current sampling grid (no sliding "
            "window). Use 2 for every sample at 500 Hz or 10 for every sample "
            "after --resample-hz 100."
        ),
    )
    parser.add_argument("--smooth-ms", type=float, default=0.0)
    parser.add_argument("--toi", type=float, nargs=2, default=(-0.05, 6.0))
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help=(
            "Optionally resample the raw EEG with MNE before alpha filtering. "
            "For example, use 100 for 100 Hz (10 ms/sample). By default the "
            "original 500-Hz data are retained. --step-ms independently "
            "controls which resampled timepoints are decoded."
        ),
    )

    parser.add_argument("--alpha-low", type=float, default=8.0)
    parser.add_argument("--alpha-high", type=float, default=12.0)
    parser.add_argument(
        "--filter-method",
        choices=["fir", "iir", "mtmconvol"],
        default="fir",
        help="FIR/Hilbert, IIR/Hilbert, or Hanning-tapered Fourier power",
    )
    parser.add_argument(
        "--mtm-window-mode", choices=["adaptive", "fixed"], default="adaptive"
    )
    parser.add_argument("--mtm-cycles", type=float, default=5.0)
    parser.add_argument("--mtm-fixed-window-ms", type=float, default=500.0)
    parser.add_argument("--mtm-frequency-step", type=float, default=1.0)
    parser.add_argument("--filter-batch-trials", type=int, default=32)
    parser.add_argument(
        "--train-time-batch",
        type=int,
        default=16,
        help=(
            "Number of training-time points processed together by the GPU "
            "time-cross Mahalanobis decoder. Larger values use more VRAM and "
            "can reduce runtime; try 32, then 64. Default: 16."
        ),
    )

    parser.add_argument("--basis-bins", type=int, default=16)
    parser.add_argument("--basis-exponent", type=float, default=15.0)
    parser.add_argument("--basis-orientation-spaces", type=int, default=8)
    parser.add_argument("--basis-reps", type=int, default=100)
    parser.add_argument(
        "--trialwise-only",
        action="store_true",
        help=(
            "Save only time vectors, trial IDs, and early/late trialwise "
            "train-by-test decoding matrices; omit distance and mean maps."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_is_complete(path: Path, trialwise_only: bool = False) -> bool:
    """Check that both separate-session time-cross results were saved."""
    if not path.is_file():
        return False
    trialwise_required = {
        "time_dec",
        "sess1_trial_ids",
        "sess1_trial_dec_early",
        "sess1_trial_dec_late",
        "sess2_trial_ids",
        "sess2_trial_dec_early",
        "sess2_trial_dec_late",
    }
    required = set(trialwise_required)
    if not trialwise_only:
        required |= {
            "dec_early", "dec_late", "dist_early", "dist_late",
            "sess1_dec_early", "sess1_dec_late",
            "sess1_dist_early", "sess1_dist_late",
            "sess2_dec_early", "sess2_dec_late",
            "sess2_dist_early", "sess2_dist_late",
        }
    try:
        with np.load(path, allow_pickle=True) as saved:
            return required.issubset(saved.files)
    except (OSError, ValueError, EOFError):
        return False


def decode_session(
    subject: int,
    session: int,
    args: argparse.Namespace,
    config: DecodeConfig,
    data_dir: Path,
) -> dict:
    """Load, extract alpha, and independently decode one session."""
    print(f"subject {subject:02d}, session {session}", flush=True)
    input_file = data_dir / f"MemImp3_mem_whole_sess{session}_{subject}.mat"
    ft_mem = load_session(input_file)
    labels = np.asarray(ft_mem["label"]).reshape(-1)
    indices = channel_indices(labels, all_eeg=args.all_eeg)
    selected_channels = labels[indices]

    n_trials = ft_mem["trial"].shape[0]
    bad = np.asarray(ft_mem["bad_trials_mem"], dtype=int).reshape(-1) - 1
    good = np.setdiff1d(np.arange(n_trials), bad)
    time = np.asarray(ft_mem["time"], dtype=np.float64).reshape(-1)
    original_hz = float(1 / np.median(np.diff(time)))
    hz = float(round(original_hz))
    raw = np.asarray(
        ft_mem["trial"][np.ix_(good, indices, np.arange(time.size))],
        dtype=np.float32,
    )
    results = np.asarray(ft_mem["Results"])[good]
    del ft_mem

    # Optional anti-aliased resampling is deliberately performed before the
    # alpha FIR/IIR/MTM extraction. MNE applies its resampling low-pass filter;
    # this is not simple decimation and not a 10-ms boxcar average.
    if args.resample_hz is not None:
        target_hz = float(args.resample_hz)
        if not np.isfinite(target_hz) or target_hz <= 0:
            raise ValueError("--resample-hz must be a positive finite number")
        if target_hz > hz * (1 + 1e-9):
            raise ValueError(
                f"--resample-hz={target_hz:g} exceeds the original {hz:g} Hz; "
                "this option is intended for downsampling"
            )
        if args.alpha_high >= target_hz / 2:
            raise ValueError(
                f"alpha-high={args.alpha_high:g} Hz must be below the "
                f"{target_hz / 2:g}-Hz Nyquist frequency after resampling"
            )
        old_n_times = raw.shape[-1]
        # This installed MNE version requires float64 input for its FFT
        # resampler. Convert only for resampling, then immediately return to
        # float32 so alpha extraction and decoding do not retain the 2x RAM cost.
        raw = mne_resample(
            raw.astype(np.float64, copy=False),
            up=target_hz,
            down=hz,
            npad="auto",
            verbose=False,
        ).astype(np.float32, copy=False)
        # Preserve the first sample's event-relative time and construct the
        # exact uniform time axis corresponding to the requested output rate.
        time = time[0] + np.arange(raw.shape[-1], dtype=np.float64) / target_hz
        hz = target_hz
        print(
            f"  MNE resample before alpha filtering: {original_hz:.3f} -> "
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
    del raw

    # First construct genuine single-sample features (never a sliding window),
    # then optionally retain one such original sample every --step-ms.  This
    # restores the established full-trial 50-ms workflow while still allowing
    # all 500-Hz samples with --step-ms 2.
    features, time_dec = make_sliding_features(
        power, time, config, mode="timepoint"
    )
    del power
    if args.step_ms <= 0:
        raise ValueError("--step-ms must be positive")
    native_step_ms = float(np.median(np.diff(time_dec)) * 1000)
    if args.step_ms < native_step_ms * (1 - 1e-6):
        raise ValueError(
            f"--step-ms={args.step_ms:g} is finer than the available "
            f"{native_step_ms:g} ms/sample grid; use --step-ms >= "
            f"{native_step_ms:g}"
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

    decoded = {}
    for name, column in (("early", 5), ("late", 6)):
        continuous_theta = results[:, column] * 2
        decode_seed = args.seed + 1000 * subject + 100 * session + column
        print(
            f"  decoding {name}: separate session, train-time x test-time, "
            f"{args.basis_bins} bins, {config.n_folds}-fold, "
            f"half-cosine^{args.basis_exponent:g}, "
            f"{args.basis_orientation_spaces} orientation spaces, "
            f"{args.basis_reps} repetitions per space, "
            f"train-time batch {args.train_time_batch}",
            flush=True,
        )
        decoded[name] = decode_basis_kfold_time_cross_repetitions(
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
            train_time_batch=args.train_time_batch,
        )
    del features

    return {
        "decoded": decoded,
        "time_dec": time_dec,
        "channels": selected_channels,
        "sfreq": hz,
        "original_sfreq": original_hz,
        # One-based source row IDs, consistent with the existing decoder.
        "trial_ids": good + 1,
    }


def main() -> None:
    args = parse_args()
    if args.alpha_low <= 0 or args.alpha_high <= args.alpha_low:
        raise ValueError("alpha band must satisfy 0 < alpha-low < alpha-high")
    if args.train_time_batch < 1:
        raise ValueError("--train-time-batch must be positive")

    data_dir = PROJECT_DIR / "Data" / "Micheal_Data_exp2"
    output_dir = args.output_dir or (
        PROJECT_DIR
        / "results"
        / (
            f"michael_alpha_time_cross_basis_kfold_{args.filter_method}_"
            "timepoint_separate_sessions"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DecodeConfig(
        n_folds=args.n_folds,
        # The basis decoder uses basis_reps; this field is metadata only here.
        reps=args.basis_reps,
        # This is the spacing between selected single-sample timepoints. It
        # does not introduce a window or concatenate neighbouring samples.
        step_ms=args.step_ms,
        smooth_ms=args.smooth_ms,
        toi=tuple(args.toi),
    )

    for subject in args.subjects:
        output = output_dir / f"subject_{subject:02d}_alpha_time_cross_torch.npz"
        if not args.overwrite and output_is_complete(output, args.trialwise_only):
            print(f"skipping completed {output}", flush=True)
            continue
        if output.exists():
            print(f"recomputing incomplete output {output}", flush=True)

        sessions = [
            decode_session(subject, session, args, config, data_dir)
            for session in (1, 2)
        ]
        if not np.array_equal(sessions[0]["time_dec"], sessions[1]["time_dec"]):
            raise ValueError("Decoding time vectors differ between sessions")
        if not np.array_equal(sessions[0]["channels"], sessions[1]["channels"]):
            raise ValueError("Selected channels differ between sessions")

        temporary_output = output.with_suffix(".tmp.npz")
        payload = {
            "time_dec": sessions[0]["time_dec"],
            "train_time": sessions[0]["time_dec"],
            "test_time": sessions[0]["time_dec"],
            "channels": sessions[0]["channels"],
            "alpha_band": np.asarray([args.alpha_low, args.alpha_high]),
            "filter_method": args.filter_method,
            "mtm_window_mode": args.mtm_window_mode,
            "mtm_cycles": args.mtm_cycles,
            "mtm_fixed_window_ms": args.mtm_fixed_window_ms,
            "mtm_frequency_step": args.mtm_frequency_step,
            "angle_method": "basis-kfold-time-cross-generalization",
            "basis_bins": args.basis_bins,
            "basis_hard_bin_width_deg": 180.0 / args.basis_bins,
            "basis_hard_bin_half_width_deg": 90.0 / args.basis_bins,
            "basis_exponent": args.basis_exponent,
            "basis_orientation_spaces": args.basis_orientation_spaces,
            "basis_reps": args.basis_reps,
            "train_time_batch": args.train_time_batch,
            "basis_template_convolution": "row_normalized_half_cosine",
            "basis_training_bin_balancing": "random_subsample_to_fold_minimum",
            "decoding_feature_mode": "timepoint",
            "original_sfreq": sessions[0]["original_sfreq"],
            "resample_hz_requested": (
                np.nan if args.resample_hz is None else args.resample_hz
            ),
            "effective_sfreq": sessions[0]["sfreq"],
            "timepoint_stride_samples": int(
                round(args.step_ms * sessions[0]["sfreq"] / 1000.0)
            ),
            "session_pooling": "separate",
            "time_axis_order": "train_time,test_time",
            "storage_mode": (
                "trialwise_only" if args.trialwise_only else "full"
            ),
            "config": np.array(config.__dict__, dtype=object),
        }
        if not args.trialwise_only:
            # Equally weighted mean of the two independently decoded sessions.
            payload.update(
                {
                    "dec_early": np.mean(
                        [entry["decoded"]["early"][0] for entry in sessions],
                        axis=0,
                    ),
                    "dec_late": np.mean(
                        [entry["decoded"]["late"][0] for entry in sessions],
                        axis=0,
                    ),
                    "dist_early": np.mean(
                        [entry["decoded"]["early"][1] for entry in sessions],
                        axis=0,
                    ),
                    "dist_late": np.mean(
                        [entry["decoded"]["late"][1] for entry in sessions],
                        axis=0,
                    ),
                }
            )
        for index, entry in enumerate(sessions, start=1):
            session_payload = {
                f"sess{index}_trial_ids": entry["trial_ids"],
                f"sess{index}_trial_dec_early": entry["decoded"]["early"][2],
                f"sess{index}_trial_dec_late": entry["decoded"]["late"][2],
            }
            if not args.trialwise_only:
                session_payload.update(
                    {
                        f"sess{index}_dec_early": entry["decoded"]["early"][0],
                        f"sess{index}_dist_early": entry["decoded"]["early"][1],
                        f"sess{index}_dec_late": entry["decoded"]["late"][0],
                        f"sess{index}_dist_late": entry["decoded"]["late"][1],
                    }
                )
            payload.update(session_payload)
        np.savez_compressed(temporary_output, **payload)
        temporary_output.replace(output)
        print(f"saved {output}", flush=True)
        
        # Release CPU objects and unused CUDA cache after each subject.
        del sessions, payload
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
