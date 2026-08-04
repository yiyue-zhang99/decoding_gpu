#!/usr/bin/env python3
"""Run batched PyTorch Mahalanobis decoding on Michael/Wolff exp. 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_DIR = Path('/home/dilay/project2/tw')

sys.path.insert(
    0,
    str(PROJECT_DIR / 'functions' / 'decoding' / 'decoding_gpu')
)

from helper import (  # noqa: E402
    DecodeConfig,
    make_sliding_features,
)
from mahal_theta_hard_loocv_torch import mahal_hard_loocv  # noqa: E402

CHANNELS = [ "PZ", "POZ", "OZ", "P1", "PO3", "O1", "P7", "PO7", "P5", "P3", "P2", "PO4", "O2", "P8", "PO8", "P4", "P6", ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--smooth-ms", type=float, default=0.0)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument(
        "--decoding-feature-mode",
        choices=["sliding-window", "timepoint"],
        default="sliding-window",
        help=(
            "Use the current temporally concatenated sliding-window features "
            "or only the channels at each individual decoding timepoint"
        ),
    )
    parser.add_argument("--toi", type=float, nargs=2, default=(-0.05, 6.0))
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
        help="Number of trial-relative templates for hard LOOCV",
    )
    return parser.parse_args()


def load_session(path: Path) -> dict:
    loaded = loadmat(path, variable_names=["ft_mem"], simplify_cells=True)
    return loaded["ft_mem"]


def channel_indices(labels: np.ndarray) -> np.ndarray:
    """Return the 17 posterior-channel indices in the requested order."""
    available = {
        str(label).strip().upper(): index
        for index, label in enumerate(np.asarray(labels).reshape(-1))
    }
    missing = [name for name in CHANNELS if name not in available]
    if missing:
        raise ValueError(f"Missing requested channels: {missing}")
    return np.asarray([available[name] for name in CHANNELS], dtype=int)


def main() -> None:
    args = parse_args()
    data_dir = PROJECT_DIR / "Data" / "Micheal_Data_exp2"
    output_name = "michael_voltage_decoding_hard_loocv_raw"
    if args.decoding_feature_mode == "timepoint":
        output_name += "_timepoint"
    output_dir = PROJECT_DIR / "results" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DecodeConfig(
        step_ms=args.step_ms,
        smooth_ms=args.smooth_ms,
        toi=tuple(args.toi),
    )

    for subject in args.subjects:
        session_results = []
        for session in (1, 2):
            print(f"subject {subject:02d}, session {session}", flush=True)
            path = data_dir / f"MemImp3_mem_whole_sess{session}_{subject}.mat"
            ft_mem = load_session(path)
            indices = channel_indices(ft_mem["label"])
            n_trials = ft_mem["trial"].shape[0]
            bad = np.asarray(ft_mem["bad_trials_mem"], dtype=int).reshape(-1) - 1
            good = np.setdiff1d(np.arange(n_trials), bad)
            raw = np.asarray(ft_mem["trial"])[good][:, indices, :]
            results = np.asarray(ft_mem["Results"])[good]
            print(
                f"  selected {len(CHANNELS)} channels: {', '.join(CHANNELS)}",
                flush=True,
            )
            features, time_dec = make_sliding_features(
                raw,
                np.asarray(ft_mem["time"]),
                config,
                mode=args.decoding_feature_mode,
            )
            del raw, ft_mem

            decoded = {}
            for name, column in (("early", 5), ("late", 6)):
                continuous_theta = results[:, column] * 2
                # Orientations are doubled to span a full circle, so the
                # requested orientation half-width is doubled as well.
                bin_half_width = np.deg2rad(2 * args.hard_width_deg)
                print(
                    f" decoding {name}: hard LOOCV, "
                    f"half-width={args.hard_width_deg:g} orientation deg, "
                    f"{args.hard_templates} templates",
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
            session_results.append(
                {
                    "decoded": decoded,
                    # One-based original trial rows for matching other results.
                    "trial_ids": good + 1,
                }
            )
            del features

        dec_early = np.mean(
            [item["decoded"]["early"][0] for item in session_results], axis=0
        )
        dec_late = np.mean(
            [item["decoded"]["late"][0] for item in session_results], axis=0
        )
        dist_early = np.mean(
            [item["decoded"]["early"][1] for item in session_results], axis=0
        )
        dist_late = np.mean(
            [item["decoded"]["late"][1] for item in session_results], axis=0
        )
        output = output_dir / f"subject_{subject:02d}_voltage_torch.npz"
        np.savez_compressed(
            output,
            time_dec=time_dec,
            dec_early=dec_early,
            dec_late=dec_late,
            dist_early=dist_early,
            dist_late=dist_late,
            channels=np.asarray(CHANNELS),
            angle_method="hard-loocv",
            hard_width_deg=args.hard_width_deg,
            hard_templates=args.hard_templates,
            decoding_feature_mode=args.decoding_feature_mode,
            config=np.array(config.__dict__, dtype=object),
            sess1_trial_ids=session_results[0]["trial_ids"],
            sess1_trial_dec_early=session_results[0]["decoded"]["early"][2],
            sess1_trial_dec_late=session_results[0]["decoded"]["late"][2],
            sess2_trial_ids=session_results[1]["trial_ids"],
            sess2_trial_dec_early=session_results[1]["decoded"]["early"][2],
            sess2_trial_dec_late=session_results[1]["decoded"]["late"][2],
        )
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
