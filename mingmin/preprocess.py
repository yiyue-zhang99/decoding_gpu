#!/usr/bin/env python3
"""Preprocess Mingmin sequence BrainVision files through ICA ocular cleaning."""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


RAW_DIR = Path("/media/dilay/T7/mingmin_data/sequence/raw_dedup")
RAW_EEG_DIR = Path("/media/dilay/T7/mingmin_data/sequence/raw")
MONTAGE_PATH = Path("/home/dilay/project2/tw/travelling_waves/tw/fft/mingmin/sequence/standard_waveguard64_equidistant.elc")
OUT_ROOT = Path("/home/dilay/project2/tw/Data/Mingmin_data/sequency")
EPOCH_THRESHOLD_DIR = OUT_ROOT / "epoch_threshold"
ONLINE_REFERENCE_CH = "5Z"
EXCLUDE_THRESHOLD_CHS = {"0Z", "1L", "1R", "1LD", "1LC", "1RC", "1RD"}

L_FREQ = 0.1
H_FREQ = 40.0
ICA_FIT_L_FREQ = 1.0
TARGET_SFREQ = 500.0
TMIN = -1.5
TMAX = 7.5
RANDOM_STATE = 42

CORE_TRIGGER_CATEGORY = {
    101: "stim1",
    103: "stim1",
    102: "stim2",
    104: "stim2",
    105: "impulse1",
    107: "probe1",
    106: "impulse2",
    108: "probe2",
}
EXPECTED_CORE = ["stim1", "stim2", "impulse1", "probe1", "impulse2", "probe2"]
EPOCH_EVENT_ID = {"stim1": 1001}
TIMING_TOLERANCE_S = 0.05
EXPECTED_TRIGGER_TIMES = {
    "stim2_rel_s": 1.15,
    "impulse1_rel_s": 2.30,
    "probe1_rel_s": 2.90,
    "impulse2_rel_s": 4.45,
    "probe2_rel_s": 5.05,
}

# Only these sessions use AutoReject's default interpolation behavior. All
# other sessions keep the stricter detection-only grid n_interpolate=[0].
# AUTOREJECT_DEFAULT_INTERPOLATION_KEYS = {
#     *(f"sub08_session{session}" for session in range(1, 5)),
#     *(f"sub09_session{session}" for session in range(1, 5)),
#     *(f"sub26_session{session}" for session in range(1, 5)),
#     *(f"sub27_session{session}" for session in range(1, 5)),
# }


@dataclass(frozen=True)
class Marker:
    mk_index: int
    sample: int
    code: int
    desc: str
    category: str


def output_key(path: Path) -> str:
    match = re.search(r"seq_(WM\d+)_session([0-9.]+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return path.stem
    subject, session = match.groups()
    subject_num = re.search(r"\d+", subject).group(0)
    return f"sub{subject_num}_session{session}"


def category_for_code(code: int) -> str:
    if 1 <= code <= 100:
        return "fixation"
    return CORE_TRIGGER_CATEGORY.get(code, "other")


def read_stim_markers(vhdr_path: Path) -> list[Marker]:
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    markers: list[Marker] = []
    for line in vmrk_path.read_text(errors="replace").splitlines():
        marker_match = re.match(r"^Mk(\d+)=(.*)$", line.strip())
        if not marker_match:
            continue

        mk_index = int(marker_match.group(1))
        fields = marker_match.group(2).split(",")
        marker_type = fields[0] if len(fields) > 0 else ""
        desc = fields[1] if len(fields) > 1 else ""

        code_match = re.search(r"(?:^|/)s\s*(\d+)$", desc.strip(), flags=re.IGNORECASE)
        if marker_type != "Stimulus" or code_match is None:
            continue
        code = int(code_match.group(1))

        try:
            position = int(fields[2])
        except (IndexError, ValueError):
            continue

        markers.append(
            Marker(
                mk_index=mk_index,
                sample=position - 1,
                code=code,
                desc=f"s{code}",
                category=category_for_code(code),
            )
        )
    return sorted(markers, key=lambda marker: (marker.sample, marker.mk_index))


def group_markers_by_trial_start(markers: list[Marker]) -> list[list[Marker]]:
    starts = [idx for idx, marker in enumerate(markers) if marker.category == "fixation"]
    trials: list[list[Marker]] = []
    for trial_idx, start_idx in enumerate(starts):
        stop_idx = starts[trial_idx + 1] if trial_idx + 1 < len(starts) else len(markers)
        trials.append(markers[start_idx:stop_idx])
    return trials


def build_events_and_trigger_intervals(
    trials: list[list[Marker]],
    old_sfreq: float,
    new_sfreq: float,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    events = []
    trial_issues: list[dict[str, object]] = []
    scale = new_sfreq / old_sfreq

    for trial_index, trial in enumerate(trials, start=1):
        fixation = trial[0]
        stim1 = next((marker for marker in trial if marker.category == "stim1"), None)
        if stim1 is None:
            stim2 = next((marker for marker in trial if marker.category == "stim2"), None)
            trial_issues.append(
                {
                    "trial_index": trial_index,
                    "trial_code": fixation.desc,
                    "trial_start_sample_orig": fixation.sample,
                    "stim2_code": stim2.desc if stim2 is not None else "",
                    "stim2_sample_orig": stim2.sample if stim2 is not None else "",
                    "issue": "missing_stim1",
                }
            )
            continue

        event_sample = int(round(stim1.sample * scale))
        events.append([event_sample, 0, EPOCH_EVENT_ID["stim1"]])
        row: dict[str, object] = {
            "trial_index": trial_index,
            "trial_code": fixation.desc, #orignial trigger
            "trial_start_sample_orig": fixation.sample,
            "trial_start_sample_resampled": int(round(fixation.sample * scale)),
            "stim1_code": stim1.desc,
            "stim1_sample_orig": stim1.sample,
            "stim1_sample_resampled": event_sample,
        }

        for category in EXPECTED_CORE[1:]:
            marker = next((item for item in trial if item.category == category), None)
            prefix = category
            if marker is None:
                row[f"{prefix}_code"] = ""
                row[f"{prefix}_sample_orig"] = np.nan
                row[f"{prefix}_sample_resampled"] = np.nan
                row[f"{prefix}_rel_s"] = np.nan
                continue

            row[f"{prefix}_code"] = marker.desc
            row[f"{prefix}_sample_orig"] = marker.sample
            row[f"{prefix}_sample_resampled"] = int(round(marker.sample * scale))
            row[f"{prefix}_rel_s"] = (marker.sample - stim1.sample) / old_sfreq

        other_markers = [marker for marker in trial if marker.category == "other"]
        row["other_trigger_codes"] = ",".join(marker.desc for marker in other_markers)
        row["other_trigger_rel_s"] = ",".join(
            f"{(marker.sample - stim1.sample) / old_sfreq:.6f}" for marker in other_markers
        )
        rows.append(row)

    if not events:
        raise RuntimeError("No stim1 events available for epoching.")

    return np.asarray(events, dtype=int), pd.DataFrame(rows), trial_issues


def prepare_raw(path: Path, montage_path: Path) -> tuple[mne.io.BaseRaw, list[Marker]]:
    raw_vhdr_path = RAW_EEG_DIR / path.name
    if not raw_vhdr_path.exists():
        raise FileNotFoundError(f"Cannot find original raw EEG header: {raw_vhdr_path}")

    raw = mne.io.read_raw_brainvision(raw_vhdr_path, preload=True, verbose="ERROR")
    markers = read_stim_markers(path)

    if ONLINE_REFERENCE_CH not in raw.ch_names:
        raw = mne.add_reference_channels(
            raw,
            ref_channels=[ONLINE_REFERENCE_CH],
            copy=False,
        )

    # Apply the montage while EOG is still typed "eeg" so it gets a real
    # position from the montage file too, then relabel channel types
    # afterward. Switching type doesn't clear the position already set.
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    channel_types = {}
    if "EOG" in raw.ch_names:
        channel_types["EOG"] = "eog"
    if "photodiode" in raw.ch_names:
        channel_types["photodiode"] = "misc"
    if channel_types:
        raw.set_channel_types(channel_types, verbose=False)

    raw.filter(L_FREQ, None, picks=["eeg", "eog"], verbose=False)
    raw.filter(None, H_FREQ, picks=["eeg", "eog"], verbose=False)
    return raw, markers


def fit_ica_and_find_eog(
    raw: mne.io.BaseRaw,
    eog_chs: list[str],
    n_components: float | int | None,
) -> tuple[ICA, list[int], dict[str, list[int]]]:
    fit_raw = raw.copy().filter(ICA_FIT_L_FREQ, None, picks="eeg", verbose=False)
    ica = ICA(
        n_components=n_components,
        method="picard",
        fit_params={"extended": True},
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(fit_raw, picks="eeg", verbose=False)

    eog_component_map: dict[str, list[int]] = {}
    excluded: set[int] = set()
    for ch in eog_chs:
        if ch not in raw.ch_names:
            continue
        inds, _ = ica.find_bads_eog(raw, ch_name=ch, verbose=False)
        eog_component_map[ch] = [int(ind) for ind in inds]
        excluded.update(int(ind) for ind in inds)

    ica.exclude = sorted(excluded)
    return ica, ica.exclude, eog_component_map


def find_threshold_bad_epochs(epochs: mne.Epochs, eeg_reject_uv: float) -> np.ndarray:
    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, misc=False)
    eeg_picks = [
        pick for pick in eeg_picks
        if epochs.ch_names[pick] not in EXCLUDE_THRESHOLD_CHS
    ]
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found for threshold rejection.")

    data = epochs.get_data(picks=eeg_picks, units="uV", verbose=False)
    peak_to_peak = np.ptp(data, axis=2)
    return np.any(peak_to_peak > eeg_reject_uv, axis=1)


def make_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    reject_method: str,
    eeg_reject_uv: float,
    autoreject_default_interpolation: bool = False,
) -> tuple[mne.Epochs, list[int]]:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EPOCH_EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        picks=["eeg", "eog"],
        reject=None,
        reject_by_annotation=False,
        verbose=False,
    )

    kept_after_boundary = [int(idx) for idx in epochs.selection]
    boundary_dropped = sorted(set(range(len(events))) - set(kept_after_boundary))
    dropped_after_boundary: list[int] = []
    if reject_method == "peak_to_peak":
        bad_mask = find_threshold_bad_epochs(epochs, eeg_reject_uv)
        dropped_after_boundary = [
            kept_after_boundary[int(idx)] for idx in np.flatnonzero(bad_mask)
        ]
    elif reject_method == "autoreject":
        try:
            from autoreject import AutoReject
        except ImportError as exc:
            raise RuntimeError("autoreject is not installed in this environment.") from exc
        ar_kwargs = {
            "picks": mne.pick_types(epochs.info, eeg=True),
            # Kept at 1: file-level parallelism (main()'s Parallel(n_jobs=...))
            # already saturates the machine. A second parallel layer here
            # would oversubscribe (n_jobs files x this n_jobs each).
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
            "verbose": False,
        }
        if not autoreject_default_interpolation:
            ar_kwargs["n_interpolate"] = [0]
        ar = AutoReject(**ar_kwargs)
        ar.fit(epochs)
        reject_log = ar.get_reject_log(epochs)
        bad_mask = reject_log.bad_epochs
        dropped_after_boundary = [
            kept_after_boundary[int(idx)] for idx in np.flatnonzero(bad_mask)
        ]
        # Intentionally not calling ar.transform(epochs): we keep every trial
        # in the returned epochs and only record which ones are bad.
    else:
        raise ValueError(f"Unknown reject method: {reject_method}")

    dropped_all = sorted(set(boundary_dropped + dropped_after_boundary))
    bad_set = set(dropped_after_boundary)
    epochs.metadata = pd.DataFrame(
        {
            "orig_trial_index": kept_after_boundary,
            "is_bad_epoch": [idx in bad_set for idx in kept_after_boundary],
        }
    )
    return epochs, dropped_all


def annotate_timing_issues(trigger_intervals: pd.DataFrame) -> pd.DataFrame:
    trigger_intervals = trigger_intervals.copy()
    timing_issue_flags: list[bool] = []
    issue_details: list[str] = []

    for _, row in trigger_intervals.iterrows():
        bad_parts = []
        for col, expected in EXPECTED_TRIGGER_TIMES.items():
            actual = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(actual):
                bad_parts.append(f"{col}:missing")
                continue

            diff = float(actual) - expected
            if abs(diff) > TIMING_TOLERANCE_S:
                bad_parts.append(f"{col}:{actual:.3f},diff={diff:+.3f}")

        timing_issue_flags.append(bool(bad_parts))
        issue_details.append("; ".join(bad_parts))

    trigger_intervals["has_timing_issue"] = timing_issue_flags
    trigger_intervals["timing_issue_detail"] = issue_details
    return trigger_intervals


def process_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    key = output_key(path)
    epoch_prefix = EPOCH_THRESHOLD_DIR / key
    epochs_clean_path = epoch_prefix.with_suffix(".fif")
    trigger_intervals_path = epoch_prefix.with_name(f"{epoch_prefix.name}_intervals.csv")

    if epochs_clean_path.exists() and not args.overwrite:
        print(f"[skip] {key} already done")
        return {"file": path.name, "key": key, "status": "skipped"}

    EPOCH_THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[start] {key}")

    raw, markers = prepare_raw(path, args.montage)
    old_sfreq = float(raw.info["sfreq"])
    trials = group_markers_by_trial_start(markers)

    stim1_events_orig, trigger_intervals, trial_issues = build_events_and_trigger_intervals(
        trials,
        old_sfreq=old_sfreq,
        new_sfreq=args.sfreq,
    )
    raw.resample(args.sfreq, verbose=False)
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    missing_stim1_path = None
    if trial_issues:
        missing_stim1_path = epoch_prefix.with_name(f"{epoch_prefix.name}_missing_stim1.csv")

    ica, excluded, eog_component_map = fit_ica_and_find_eog(
        raw,
        eog_chs=args.eog_chs,
        n_components=args.ica_n_components,
    )
    auto_excluded = list(excluded)

    raw_clean = raw.copy()
    ica.apply(raw_clean, exclude=excluded, verbose=False)

    epochs_clean, dropped_trials = make_epochs(
        raw_clean,
        stim1_events_orig,
        reject_method=args.bad_epoch_method,
        eeg_reject_uv=args.eeg_reject_uv,
        # autoreject_default_interpolation=(
        #     key in AUTOREJECT_DEFAULT_INTERPOLATION_KEYS
        # ),
    )
    epochs_clean.save(epochs_clean_path, overwrite=True, verbose=False)

    dropped_set = set(dropped_trials)
    trigger_intervals = trigger_intervals.copy()
    trigger_intervals["is_bad_epoch"] = [
        idx in dropped_set for idx in range(len(trigger_intervals))
    ]
    trigger_intervals["dropped_reason"] = [
        args.bad_epoch_method if idx in dropped_set else ""
        for idx in range(len(trigger_intervals))
    ]
    trigger_intervals = annotate_timing_issues(trigger_intervals)
    trigger_intervals["drop_by_timing_or_bad_epoch"] = (
        trigger_intervals["has_timing_issue"] | trigger_intervals["is_bad_epoch"]
    )
    trigger_intervals.to_csv(trigger_intervals_path, index=False)
    if missing_stim1_path is not None:
        pd.DataFrame(trial_issues).to_csv(missing_stim1_path, index=False)
    else:
        old_missing_stim1_path = epoch_prefix.with_name(f"{epoch_prefix.name}_missing_stim1.csv")
        if old_missing_stim1_path.exists():
            old_missing_stim1_path.unlink()

    summary = {
        "file": path.name,
        "key": key,
        "status": "done",
        "n_trials_from_markers": len(trials),
        "n_epochs_saved": len(epochs_clean),
        "n_missing_stim1_trials": len(trial_issues),
        "missing_stim1_trials": trial_issues,
        "auto_ica_excluded_components": auto_excluded,
        "ica_excluded_components": excluded,
        "ica_eog_component_map": eog_component_map,
        "bad_epoch_method": args.bad_epoch_method,
        "eeg_reject_uv": args.eeg_reject_uv,
        "excluded_threshold_channels": ",".join(sorted(EXCLUDE_THRESHOLD_CHS)),
        # "autoreject_default_interpolation": (
        #     key in AUTOREJECT_DEFAULT_INTERPOLATION_KEYS
        # ),
        "bad_trial_indices_zero_based": dropped_trials,
        "epochs_threshold": str(epochs_clean_path),
        "trigger_intervals": str(trigger_intervals_path),
        "missing_stim1_trials_csv": str(missing_stim1_path) if missing_stim1_path is not None else "",
        "runtime_s": round(time.time() - t0, 3),
    }

    print(
        f"[done] {key}: epochs={len(epochs_clean)}, "
        f"ICA exclude={excluded}, missing_stim1={len(trial_issues)}, "
        f"{time.time() - t0:.1f}s"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess sequence raw_dedup BrainVision files and remove EOG-related ICA components."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional .vhdr files. Defaults to all raw_dedup files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--montage", type=Path, default=MONTAGE_PATH)
    parser.add_argument("--sfreq", type=float, default=TARGET_SFREQ)
    parser.add_argument("--eog-chs", nargs="*", default=["EOG"])
    parser.add_argument("--ica-n-components", type=int, default=30)
    parser.add_argument(
        "--bad-epoch-method",
        choices=["peak_to_peak", "autoreject"],
        default="peak_to_peak",
        help="Bad epoch flagging method after ICA cleaning. Epochs are flagged, not dropped.",
    )
    parser.add_argument("--eeg-reject-uv", type=float, default=250.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of files to process in parallel.")
    return parser


def process_file_safely(path: Path, args: argparse.Namespace) -> dict[str, object]:
    try:
        return process_file(path, args)
    except Exception as exc:
        import traceback

        print(f"[error] {path.name}: {exc}")
        return {
            "file": path.name,
            "key": output_key(path),
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mne.set_log_level("WARNING")

    paths = args.paths or sorted(
        args.raw_dir.glob("*.vhdr"),
        key=lambda path: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    if args.limit is not None:
        paths = paths[: args.limit]

    from joblib import Parallel, delayed, parallel_config

    inner_threads = max(1, (os.cpu_count() or 1) // args.n_jobs)
    print(
        f"Processing {len(paths)} file(s) with n_jobs={args.n_jobs} "
        f"(inner_max_num_threads={inner_threads})"
    )
    with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
        summary_rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(process_file_safely)(path, args) for path in paths
        )

    summary_path = OUT_ROOT / "sequence_threshold_preprocess_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
