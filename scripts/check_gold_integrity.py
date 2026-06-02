#!/usr/bin/env python3
"""Check integrity of all gold dataset files."""

import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_bench.validators.pipeline import run_validation_pipeline


def main() -> None:
    gold_dir = Path(__file__).resolve().parent.parent / "datasets" / "gold"

    if not gold_dir.exists():
        print(f"Gold directory not found: {gold_dir}")
        sys.exit(1)

    yaml_files = list(gold_dir.rglob("*.yaml")) + list(gold_dir.rglob("*.yml"))

    if not yaml_files:
        print("No YAML files found in datasets/gold/")
        sys.exit(0)

    total_errors = 0
    total_warnings = 0
    total_cases = 0

    for filepath in sorted(yaml_files):
        rel_path = filepath.relative_to(gold_dir)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            continue

        cases: list[dict]
        if isinstance(data, list):
            cases = data
        elif isinstance(data, dict) and "cases" in data:
            cases = data["cases"]
        else:
            cases = [data]

        report = run_validation_pipeline(cases)
        total_cases += report.total
        total_errors += len(report.errors)
        total_warnings += len(report.warnings)

        if report.errors or report.warnings:
            print(f"\n{rel_path} ({report.total} cases):")
            for err in report.errors:
                print(f"  ERROR: {err}")
            for warn in report.warnings:
                print(f"  WARN:  {warn}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Gold Integrity Check Summary")
    print(f"  Files scanned: {len(yaml_files)}")
    print(f"  Total cases: {total_cases}")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print(f"{'='*60}")

    if total_errors > 0:
        print("\nFAILED: integrity issues found.")
        sys.exit(1)
    else:
        print("\nPASSED: all gold cases are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
