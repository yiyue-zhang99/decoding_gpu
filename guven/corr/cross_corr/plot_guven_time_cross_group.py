#!/usr/bin/env python3
"""Plot Guven cue, uncue, and cue-minus-uncue group time-cross maps."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from guven_tw_cross_correlation import cluster_signflip_3d  # noqa: E402

SESSIONS = (1, 2, 3, 4)


def _subject_number(path: Path) -> int:
    match = re.search(r"subject_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject number from {path.name}")
    return int(match.group(1))


def load_subject_maps(result_dir: Path):
    """Recover each subject's pooled-trial cue and uncue mean matrices."""
    paths = sorted(result_dir.glob("subject_*_alpha_time_cross_torch.npz"))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two result files in {result_dir}")
    subjects, cue_maps, uncue_maps = [], [], []
    reference_time = None
    for path in paths:
        with np.load(path, allow_pickle=True) as saved:
            time = np.asarray(saved["time_dec"], dtype=float)
            if reference_time is None:
                reference_time = time
            elif not np.allclose(reference_time, time):
                raise ValueError(f"Time mismatch in {path.name}")
            cue_trials = np.concatenate(
                [
                    np.asarray(saved[f"sess{s}_trial_dec_cue"], dtype=np.float32)
                    for s in SESSIONS
                ],
                axis=0,
            )
            uncue_trials = np.concatenate(
                [
                    np.asarray(saved[f"sess{s}_trial_dec_uncue"], dtype=np.float32)
                    for s in SESSIONS
                ],
                axis=0,
            )
            if cue_trials.shape != uncue_trials.shape:
                raise ValueError(f"Cue/uncue shape mismatch in {path.name}")
            cue_maps.append(cue_trials.mean(axis=0, dtype=np.float64))
            uncue_maps.append(uncue_trials.mean(axis=0, dtype=np.float64))
            subjects.append(_subject_number(path))
        print(
            f"loaded subject {subjects[-1]:02d}: {cue_trials.shape[0]} pooled trials",
            flush=True,
        )
    cue_maps = np.asarray(cue_maps, dtype=np.float32)
    uncue_maps = np.asarray(uncue_maps, dtype=np.float32)
    return np.asarray(subjects), reference_time, {
        "cue": cue_maps,
        "uncue": uncue_maps,
        "cue_minus_uncue": cue_maps - uncue_maps,
    }


def test_map(subject_maps, permutations, seed):
    """Two-sided 2-D cluster-mass sign-flip via a singleton third axis."""
    mean_map, observed_t, clusters, null_max = cluster_signflip_3d(
        subject_maps[..., None],
        permutations=permutations,
        cluster_alpha=0.05,
        cluster_p=0.05,
        seed=seed,
        return_all=True,
    )
    for cluster in clusters:
        cluster["mask"] = cluster["mask"][..., 0]
    return mean_map[..., 0], observed_t[..., 0], clusters, null_max


def plot_map(mean_map, clusters, time, output, title):
    significant = [cluster for cluster in clusters if cluster["significant"]]
    positive = np.zeros(mean_map.shape, dtype=bool)
    negative = np.zeros(mean_map.shape, dtype=bool)
    for cluster in significant:
        if cluster["sign"] > 0:
            positive |= cluster["mask"]
        else:
            negative |= cluster["mask"]
    finite = np.abs(mean_map[np.isfinite(mean_map)])
    limit = float(np.percentile(finite, 99)) if finite.size else 1.0
    limit = max(limit, np.finfo(float).eps)

    fig, axis = plt.subplots(figsize=(7.2, 6.2), facecolor="white")
    image = axis.imshow(
        mean_map,
        origin="lower",
        aspect="equal",
        extent=[time[0], time[-1], time[0], time[-1]],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="bilinear",
    )
    if positive.any():
        axis.contour(
            time, time, positive.astype(float), levels=[0.5],
            colors="black", linewidths=1.8,
        )
    if negative.any():
        axis.contour(
            time, time, negative.astype(float), levels=[0.5],
            colors="black", linewidths=1.8, linestyles="--",
        )
    axis.plot([time[0], time[-1]], [time[0], time[-1]], color="0.3", lw=1.0,
              ls=":", label="Train = test")
    axis.set(
        xlabel="Test time (s)", ylabel="Training time (s)", title=title,
        xlim=(time[0], time[-1]), ylim=(time[0], time[-1]),
    )
    axis.spines[["top", "right"]].set_visible(False)
    colourbar = fig.colorbar(image, ax=axis, pad=0.025, shrink=0.88)
    colourbar.set_label("Group mean decoding evidence")
    fig.text(
        0.5, 0.012,
        "Solid/dashed black contour: positive/negative cluster-mass corrected p < .05",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects, time, maps = load_subject_maps(args.result_dir)
    figure_dir = args.result_dir / "fig"
    labels = {
        "cue": "Guven alpha time-cross decoding: cue item",
        "uncue": "Guven alpha time-cross decoding: uncued item",
        "cue_minus_uncue": "Guven alpha time-cross decoding: cue − uncued",
    }
    for index, key in enumerate(("cue", "uncue", "cue_minus_uncue")):
        print(f"\n=== {key}: {len(subjects)} subjects ===", flush=True)
        mean_map, observed_t, clusters, null_max = test_map(
            maps[key], args.permutations, args.seed + index
        )
        significant = [cluster for cluster in clusters if cluster["significant"]]
        output = figure_dir / f"group_time_cross_{key}_cluster.png"
        plot_map(mean_map, clusters, time, output, labels[key])
        print(f"significant clusters: {len(significant)}", flush=True)
        for cluster_index, cluster in enumerate(significant, start=1):
            rows, columns = np.where(cluster["mask"])
            print(
                f"  cluster {cluster_index}: sign={cluster['sign']:+d}, "
                f"corrected p={cluster['p']:.4f}, cells={cluster['n_cells']}; "
                f"training={time[rows.min()]:.3f}–{time[rows.max()]:.3f} s; "
                f"test={time[columns.min()]:.3f}–{time[columns.max()]:.3f} s",
                flush=True,
            )
        print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
