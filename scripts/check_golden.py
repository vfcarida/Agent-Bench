"""Golden benchmarks scorecard assertion check.

Validates that the latest run of pix_basic_v1 yields a functional_score within
the expected deterministic scripted band [0.55, 0.65] (nominal 0.60).
Fails with exit code 1 if scores drift outside the designated band.
"""

import json
import sys
from pathlib import Path

EXPECTED_BAND_MIN = 0.55
EXPECTED_BAND_MAX = 0.65


def check_golden_run(run_file: Path | None = None) -> int:
    runs_dir = Path("data/runs")
    if run_file is None:
        json_files = list(runs_dir.glob("*.json"))
        if not json_files:
            print(f"[ERROR] No run artifacts found in {runs_dir}", file=sys.stderr)
            return 1
        # Get the most recently modified run file
        run_file = max(json_files, key=lambda f: f.stat().st_mtime)

    print(f"Inspecting run artifact: {run_file}")
    with open(run_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    suite_id = data.get("suite_id")
    if suite_id != "pix_basic_v1":
        print(f"[WARN] Suite is {suite_id}, expected pix_basic_v1. Proceeding with scorecard check.")

    scorecards = data.get("scorecards", [])
    if not scorecards:
        print("[ERROR] No scorecards found in run artifact.", file=sys.stderr)
        return 1

    all_passed = True
    for sc in scorecards:
        system_id = sc.get("system_id", "unknown")
        domain = sc.get("domain", "unknown")
        func_score = sc.get("functional_score", 0.0)

        print(
            f"System: {system_id} | Domain: {domain} | "
            f"Functional Score: {func_score:.4f} (Expected band: [{EXPECTED_BAND_MIN}, {EXPECTED_BAND_MAX}])"
        )

        if not (EXPECTED_BAND_MIN <= func_score <= EXPECTED_BAND_MAX):
            print(
                f"[ERROR] Functional score {func_score:.4f} for system '{system_id}' "
                f"is outside expected band [{EXPECTED_BAND_MIN}, {EXPECTED_BAND_MAX}]!",
                file=sys.stderr,
            )
            all_passed = False

    if all_passed:
        print("[SUCCESS] All systems within expected golden scorecard band.")
        return 0
    else:
        print("[FAIL] Golden scorecard assertion failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(check_golden_run(target))
