#!/usr/bin/env python3
"""Check sequence trigger timing relative to stim1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EPOCH_CLEAN_DIR = Path("/home/dilay/project2/tw/Data/Mingmin_data/sequency/epoch_all")
OUT_ROOT = Path("/home/dilay/project2/tw/Data/Mingmin_data/sequency")
TOLERANCE_S = 0.05

EXPECTED_TIMES = {
    "stim2_rel_s": 1.15,
    "impulse1_rel_s": 2.30,
    "probe1_rel_s": 2.90,
    "impulse2_rel_s": 4.45,
    "probe2_rel_s": 5.05,
}


def main() -> None:
    trial_rows = []
    summary_rows = []

    interval_files = sorted(EPOCH_CLEAN_DIR.glob("*_intervals.csv"))
    for path in interval_files:
        session = path.name.replace("_intervals.csv", "")
        df = pd.read_csv(path)

        timing_issue_flags = []
        issue_details = []

        for _, row in df.iterrows():
            bad_parts = []
            for col, expected in EXPECTED_TIMES.items():
                actual = pd.to_numeric(row.get(col), errors="coerce")
                if pd.isna(actual):
                    bad_parts.append(f"{col}:missing")
                    continue

                diff = float(actual) - expected
                if abs(diff) > TOLERANCE_S:
                    bad_parts.append(f"{col}:{actual:.3f},diff={diff:+.3f}")

            timing_issue_flags.append(bool(bad_parts))
            issue_details.append("; ".join(bad_parts))

        timing_issue = pd.Series(timing_issue_flags, index=df.index)
        is_bad_epoch = df["is_bad_epoch"].astype(str).str.lower().isin(["true", "1"])
        drop_union = timing_issue | is_bad_epoch

        for idx, row in df.iterrows():
            if not timing_issue.iloc[idx]:
                continue
            trial_rows.append(
                {
                    "session": session,
                    "trial_index": int(row["trial_index"]),
                    "timing_issue": True,
                    "timing_issue_detail": issue_details[idx],
                    "is_bad_epoch": bool(is_bad_epoch.iloc[idx]),
                    "drop_by_timing_or_bad_epoch": bool(drop_union.iloc[idx]),
                    "stim2_rel_s": row.get("stim2_rel_s"),
                    "impulse1_rel_s": row.get("impulse1_rel_s"),
                    "probe1_rel_s": row.get("probe1_rel_s"),
                    "impulse2_rel_s": row.get("impulse2_rel_s"),
                    "probe2_rel_s": row.get("probe2_rel_s"),
                }
            )

        summary_rows.append(
            {
                "session": session,
                "n_epochable_trials": len(df),
                "n_timing_issue_trials": int(timing_issue.sum()),
                "n_bad_epoch_trials": int(is_bad_epoch.sum()),
                "n_union_drop_trials": int(drop_union.sum()),
                "n_keep_after_union": int((~drop_union).sum()),
                "timing_issue_percent": round(float(timing_issue.mean() * 100), 1),
                "union_drop_percent": round(float(drop_union.mean() * 100), 1),
            }
        )

    trial_out = OUT_ROOT / "timing_issue_trials.csv"
    summary_out = OUT_ROOT / "timing_issue_summary.csv"
    pd.DataFrame(trial_rows).to_csv(trial_out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    summary = pd.DataFrame(summary_rows)
    print(f"Checked {len(interval_files)} interval files.")
    print(f"Wrote {trial_out}")
    print(f"Wrote {summary_out}")
    print()
    print(summary.sort_values("n_timing_issue_trials", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
