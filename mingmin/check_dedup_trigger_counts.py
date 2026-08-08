#!/usr/bin/env python3
"""Check trigger counts in deduplicated Mingmin sequence BrainVision files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


RAW_DIR = Path("/media/dilay/T7/mingmin_data/sequence/raw_dedup")
OUT_DIR = Path(__file__).resolve().parent

TRIAL_START = set(range(1, 101))
STIM1 = {101, 103}
STIM2 = {102, 104}
IMPULSE1 = {105}
PROBE1 = {107}
IMPULSE2 = {106}
PROBE2 = {108}
RESPONSE = {200, 201, 210, 211}
EXPECTED_ORDER = [
    "fixation",
    "stim1",
    "stim2",
    "impulse1",
    "probe1",
    "impulse2",
    "probe2",
]


@dataclass(frozen=True)
class Marker:
    index: int
    marker_type: str
    description: str
    code: int | None
    sample: int | None
    category: str


def parse_ini_value(lines: list[str], key: str) -> str | None:
    prefix = key.lower() + "="
    for line in lines:
        if line.lower().startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def marker_file_for_vhdr(path: Path) -> Path:
    lines = path.read_text(errors="replace").splitlines()
    vmrk_name = parse_ini_value(lines, "MarkerFile")
    if vmrk_name is None:
        raise ValueError(f"Missing MarkerFile= line in {path}")
    return path.parent / vmrk_name


def split_marker_line(line: str) -> tuple[int, list[str]] | None:
    match = re.match(r"^Mk(\d+)=(.*)$", line.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2).split(",")


def parse_code(description: str) -> int | None:
    match = re.search(r"(?:^|/)s\s*(\d+)$", description.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def category_for_code(code: int | None) -> str:
    if code is None:
        return "non_numeric"
    if code in TRIAL_START:
        return "fixation"
    if code in STIM1:
        return "stim1"
    if code in STIM2:
        return "stim2"
    if code in IMPULSE1:
        return "impulse1"
    if code in PROBE1:
        return "probe1"
    if code in IMPULSE2:
        return "impulse2"
    if code in PROBE2:
        return "probe2"
    if code in RESPONSE:
        return "response"
    return "unknown"


def read_markers(vhdr_path: Path) -> list[Marker]:
    vmrk_path = marker_file_for_vhdr(vhdr_path)
    markers: list[Marker] = []
    for line in vmrk_path.read_text(errors="replace").splitlines():
        parsed = split_marker_line(line)
        if parsed is None:
            continue

        marker_index, fields = parsed
        marker_type = fields[0] if len(fields) > 0 else ""
        description = fields[1] if len(fields) > 1 else ""
        code = parse_code(description)
        try:
            sample = int(fields[2]) if len(fields) > 2 and fields[2] else None
        except ValueError:
            sample = None

        markers.append(
            Marker(
                index=marker_index,
                marker_type=marker_type,
                description=description,
                code=code,
                sample=sample,
                category=category_for_code(code),
            )
        )
    return markers


def compact_marker(marker: Marker) -> str:
    code = f"s{marker.code}" if marker.code is not None else marker.description
    sample = marker.sample if marker.sample is not None else "NA"
    return f"{code}@{sample}"


def split_trials(stim_markers: list[Marker]) -> list[list[Marker]]:
    start_indices = [
        idx for idx, marker in enumerate(stim_markers) if marker.category == "fixation"
    ]
    trials: list[list[Marker]] = []
    for trial_idx, start_idx in enumerate(start_indices):
        stop_idx = start_indices[trial_idx + 1] if trial_idx + 1 < len(start_indices) else len(stim_markers)
        trials.append(stim_markers[start_idx:stop_idx])
    return trials


def issue_row(path: Path, issue_type: str, message: str, **extra: object) -> dict[str, object]:
    return {"file": path.name, "issue_type": issue_type, "message": message, **extra}


def analyse_file(path: Path, expected_trials: Optional[int], expected_trial_codes: int) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    markers = read_markers(path)
    stim_markers = [marker for marker in markers if marker.marker_type == "Stimulus"]
    code_counts = Counter(marker.code for marker in stim_markers if marker.code is not None)
    category_counts = Counter(marker.category for marker in stim_markers)
    issues: list[dict[str, object]] = []

    trial_numbers = sorted(
        marker.code for marker in stim_markers if marker.category == "fixation" and marker.code is not None
    )
    trial_number_counts = Counter(trial_numbers)
    missing_trial_numbers = [
        str(code) for code in range(1, expected_trial_codes + 1) if trial_number_counts[code] == 0
    ]
    duplicate_trial_numbers = [
        f"s{code}x{count}" for code, count in sorted(trial_number_counts.items()) if count > 1
    ]
    unexpected_trial_numbers = [
        str(code) for code in trial_numbers if code < 1 or code > expected_trial_codes
    ]

    if expected_trials is not None and len(trial_numbers) != expected_trials:
        issues.append(
            issue_row(
                path,
                "trial_count",
                f"expected {expected_trials} fixation/trial-start triggers, got {len(trial_numbers)}",
                observed=len(trial_numbers),
                expected=expected_trials,
            )
        )
    if missing_trial_numbers:
        issues.append(
            issue_row(
                path,
                "missing_trial_numbers",
                "missing fixation codes: " + ",".join(missing_trial_numbers),
            )
        )
    if unexpected_trial_numbers:
        issues.append(
            issue_row(
                path,
                "unexpected_trial_numbers",
                "unexpected fixation codes: " + ",".join(unexpected_trial_numbers),
            )
        )

    expected_core_count = expected_trials if expected_trials is not None else len(trial_numbers)
    for category in EXPECTED_ORDER:
        observed = category_counts[category]
        if observed != expected_core_count:
            issues.append(
                issue_row(
                    path,
                    f"{category}_count",
                    f"expected {expected_core_count} {category} triggers, got {observed}",
                    observed=observed,
                    expected=expected_core_count,
                )
            )

    unknown = [marker for marker in stim_markers if marker.category == "unknown"]
    if unknown:
        unknown_counts = Counter(marker.code for marker in unknown)
        issues.append(
            issue_row(
                path,
                "unknown_trigger_codes",
                "unknown trigger code counts: "
                + ";".join(f"s{code}:{count}" for code, count in sorted(unknown_counts.items())),
            )
        )

    trials = split_trials(stim_markers)
    bad_trial_count = 0
    for trial_idx, trial_markers in enumerate(trials, start=1):
        counts = Counter(marker.category for marker in trial_markers)
        problems = [
            f"{category}={counts[category]}" for category in EXPECTED_ORDER if counts[category] != 1
        ]
        if not problems:
            continue

        bad_trial_count += 1
        start = trial_markers[0]
        issues.append(
            issue_row(
                path,
                "trial_core_count",
                "core category counts in trial are not all 1: " + ";".join(problems),
                trial_index=trial_idx,
                trial_code=f"s{start.code}",
                trial_start_sample=start.sample,
                trigger_sequence=",".join(
                    compact_marker(marker)
                    for marker in trial_markers
                    if marker.category in EXPECTED_ORDER
                ),
            )
        )

    count_rows = [
        {
            "file": path.name,
            "code": f"s{code}",
            "category": category_for_code(code),
            "count": count,
        }
        for code, count in sorted(code_counts.items())
    ]

    summary = {
        "file": path.name,
        "n_marker_rows": len(markers),
        "n_stimulus_rows": len(stim_markers),
        "n_trials": len(trial_numbers),
        "expected_trials": expected_trials if expected_trials is not None else "",
        "expected_core_count": expected_core_count,
        "expected_trial_codes": expected_trial_codes,
        "n_bad_trials_by_core_count": bad_trial_count,
        "n_unknown_stimulus": len(unknown),
        "has_problem": bool(issues),
        "fixation_count": category_counts["fixation"],
        "stim1_count": category_counts["stim1"],
        "stim2_count": category_counts["stim2"],
        "impulse1_count": category_counts["impulse1"],
        "probe1_count": category_counts["probe1"],
        "impulse2_count": category_counts["impulse2"],
        "probe2_count": category_counts["probe2"],
        "response_count": category_counts["response"],
        "unknown_count": category_counts["unknown"],
        "non_numeric_count": category_counts["non_numeric"],
        "missing_trial_numbers": ",".join(missing_trial_numbers),
        "duplicate_trial_numbers": ",".join(duplicate_trial_numbers),
    }
    return summary, issues, count_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check trigger counts in deduplicated sequence BrainVision marker files."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--expected-trials",
        type=int,
        default=None,
        help=(
            "Optional expected total number of trials per file. If omitted, the "
            "script uses the observed fixation count as the expected core-trigger count."
        ),
    )
    parser.add_argument(
        "--expected-trial-codes",
        type=int,
        default=100,
        help="Expected fixation code range is s1 through this value.",
    )
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("*.vhdr"))
    summary_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path.name}")
        summary, issues, counts = analyse_file(path, args.expected_trials, args.expected_trial_codes)
        summary_rows.append(summary)
        issue_rows.extend(issues)
        count_rows.extend(counts)

    write_csv(args.out_dir / "trigger_count_by_file.csv", summary_rows)
    write_csv(args.out_dir / "trigger_count_issues.csv", issue_rows)
    write_csv(
        args.out_dir / "trigger_count_by_code.csv",
        count_rows,
        ["file", "code", "category", "count"],
    )

    problem_files = [row for row in summary_rows if row["has_problem"]]
    print(f"\nFiles checked: {len(summary_rows)}")
    print(f"Files with trigger count problems: {len(problem_files)}")
    print(f"Issue rows: {len(issue_rows)}")
    print(f"Summary CSV: {args.out_dir / 'trigger_count_by_file.csv'}")
    print(f"Issue CSV: {args.out_dir / 'trigger_count_issues.csv'}")
    print(f"Code count CSV: {args.out_dir / 'trigger_count_by_code.csv'}")


if __name__ == "__main__":
    main()
