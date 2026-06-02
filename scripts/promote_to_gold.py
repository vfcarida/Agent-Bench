#!/usr/bin/env python3
"""Promote validated eval cases to the gold dataset."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_bench.validators.pipeline import run_validation_pipeline


def load_cases(source_path: Path) -> list[dict]:
    """Load cases from a YAML file."""
    with open(source_path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    return [data]


def save_cases(target_path: Path, cases: list[dict]) -> None:
    """Append cases to a YAML file (or create it)."""
    existing: list[dict] = []
    if target_path.exists():
        with open(target_path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            existing = data
        elif isinstance(data, dict) and "cases" in data:
            existing = data["cases"]

    existing.extend(cases)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote eval cases to gold dataset")
    parser.add_argument("source", type=Path, help="Source YAML file with cases")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "holdout", "calibration"],
        help="Target split for promoted cases",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: source file not found: {args.source}")
        sys.exit(1)

    cases = load_cases(args.source)
    print(f"Loaded {len(cases)} cases from {args.source}")

    # Validate
    report = run_validation_pipeline(cases)

    if report.errors:
        print(f"\nValidation errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"  - {err}")

    valid_cases: list[dict] = []
    invalid_ids: set[str] = set()

    # Identify invalid case IDs from errors
    for err in report.errors:
        # Errors are formatted as "[case_id] message"
        if err.startswith("[") and "]" in err:
            case_id = err[1 : err.index("]")]
            invalid_ids.add(case_id)

    now_iso = datetime.now(timezone.utc).isoformat()

    for case in cases:
        case_id = case.get("id", "unknown")
        if case_id in invalid_ids:
            continue
        # Promote
        case["source_type"] = "human_gold"
        case["review_status"] = "approved"
        case["promoted_to_gold_at"] = now_iso
        valid_cases.append(case)

    if not valid_cases:
        print("\nNo valid cases to promote.")
        sys.exit(1)

    # Group by domain and save
    by_domain: dict[str, list[dict]] = {}
    for case in valid_cases:
        domain = case.get("domain", "unknown")
        by_domain.setdefault(domain, []).append(case)

    gold_base = Path(__file__).resolve().parent.parent / "datasets" / "gold" / args.split
    for domain, domain_cases in by_domain.items():
        target_path = gold_base / f"{domain}.yaml"
        save_cases(target_path, domain_cases)
        print(f"  Saved {len(domain_cases)} cases to {target_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total loaded: {len(cases)}")
    print(f"  Promoted: {len(valid_cases)}")
    print(f"  Rejected: {len(cases) - len(valid_cases)}")


if __name__ == "__main__":
    main()
