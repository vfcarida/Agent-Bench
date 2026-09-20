#!/usr/bin/env python3
"""Run contamination and data leakage checks between holdout and dev/synthetic datasets."""

import argparse
import sys
from pathlib import Path

# Ensure src is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_bench.validators.contamination import check_contamination


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for benchmark dataset contamination and holdout leakage.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Jaccard similarity threshold for near-duplicate detection (default: 0.80)",
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=None,
        help="Custom holdout dataset directory",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=None,
        help="Custom dev dataset directory",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="Custom synthetic dataset directory",
    )

    args = parser.parse_args()

    report = check_contamination(
        holdout_dir=args.holdout_dir,
        dev_dir=args.dev_dir,
        synthetic_dir=args.synthetic_dir,
        threshold=args.threshold,
    )

    print(report.summary_text)

    if not report.passed:
        print("\nFAILED: Contamination gate detected leakage into dev/synthetic sets.")
        sys.exit(1)
    else:
        print("\nPASSED: Zero contamination detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
