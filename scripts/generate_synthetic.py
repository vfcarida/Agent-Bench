#!/usr/bin/env python3
"""CLI script to generate synthetic benchmark cases."""
import argparse
import os
import sys
from pathlib import Path

# Add src to path so we can import the generators
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_bench.generators.base import GenerationResult
from agent_bench.generators.business_horizon import BusinessHorizonGenerator
from agent_bench.generators.knowledge_rag import KnowledgeRagGenerator
from agent_bench.generators.security import SecurityGenerator
from agent_bench.generators.transactional import TransactionalGenerator

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


GENERATORS = {
    "transactional_tools": {
        "class": TransactionalGenerator,
        "domain": "pix_like",
    },
    "knowledge_rag_reasoning": {
        "class": KnowledgeRagGenerator,
        "domain": "investment_like",
    },
    "business_long_horizon": {
        "class": BusinessHorizonGenerator,
        "domain": "sme_advisor_like",
    },
    "security_guardrails": {
        "class": SecurityGenerator,
        "domain": "adversarial",
    },
}


def _dump_yaml(data: dict, path: Path) -> None:
    """Write data as YAML, falling back to a simple serializer if PyYAML unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    else:
        # Minimal YAML-like output without PyYAML dependency
        import json
        with open(path, "w", encoding="utf-8") as f:
            # Use JSON as a fallback (valid YAML superset)
            json.dump(data, f, ensure_ascii=False, indent=2)


def generate_family(
    family: str,
    count: int,
    seed: int,
    output_dir: Path,
) -> dict[str, int]:
    """Generate cases for one family. Returns stats dict."""
    config = GENERATORS[family]
    generator = config["class"]()
    domain = config["domain"]

    results: list[GenerationResult] = generator.generate_batch(count, seed=seed)

    accepted = [r for r in results if not r.rejected]
    rejected = [r for r in results if r.rejected]

    # Build dataset structure
    tasks = [r.case for r in accepted]
    dataset = {
        "domain": domain,
        "family": family,
        "version": "1.0.0",
        "source_type": results[0].case.get("source_type", "synthetic_shadow") if results else "synthetic_shadow",
        "description": f"Synthetic {family} cases generated deterministically",
        "tasks": tasks,
    }

    # Save
    out_path = output_dir / family / f"{domain}.yaml"
    _dump_yaml(dataset, out_path)

    return {
        "family": family,
        "domain": domain,
        "generated": len(results),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark cases for agent-bench"
    )
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=list(GENERATORS.keys()) + ["all"],
        help="Family to generate (or 'all')",
    )
    parser.add_argument(
        "--count-per-family",
        type=int,
        default=150,
        help="Number of cases per family (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/synthetic/shadow",
        help="Output directory (default: datasets/synthetic/shadow)",
    )

    args = parser.parse_args()

    # Resolve output dir relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir

    families = list(GENERATORS.keys()) if args.family == "all" else [args.family]

    print(f"Generating synthetic cases...")
    print(f"  Families: {', '.join(families)}")
    print(f"  Count per family: {args.count_per_family}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print()

    all_stats = []
    for family in families:
        stats = generate_family(
            family=family,
            count=args.count_per_family,
            seed=args.seed,
            output_dir=output_dir,
        )
        all_stats.append(stats)

    # Print summary
    print("=" * 60)
    print(f"{'Family':<30} {'Generated':>10} {'Accepted':>10} {'Rejected':>10}")
    print("-" * 60)
    total_gen = total_acc = total_rej = 0
    for stats in all_stats:
        print(f"{stats['family']:<30} {stats['generated']:>10} {stats['accepted']:>10} {stats['rejected']:>10}")
        total_gen += stats["generated"]
        total_acc += stats["accepted"]
        total_rej += stats["rejected"]
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_gen:>10} {total_acc:>10} {total_rej:>10}")
    print("=" * 60)
    print()
    for stats in all_stats:
        print(f"  -> {stats['output_path']}")


if __name__ == "__main__":
    main()
