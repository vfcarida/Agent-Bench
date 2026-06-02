"""CLI entry point for agent-bench."""

import asyncio
from pathlib import Path

import click
from rich.console import Console

from agent_bench.core.config import load_config

console = Console()


@click.group()
@click.option("--config-dir", default="configs", type=click.Path(exists=False))
@click.pass_context
def cli(ctx: click.Context, config_dir: str) -> None:
    """agent-bench: Corporate benchmark for LLMs and agent systems."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = Path(config_dir)


@cli.command()
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """Validate all configuration files."""
    config_dir = ctx.obj["config_dir"]
    if not config_dir.exists():
        console.print(f"[red]Config directory not found: {config_dir}[/red]")
        raise SystemExit(1)
    try:
        config = load_config(config_dir)
        console.print("[green]Config valid.[/green]")
        console.print(f"  Models: {len(config.models)}")
        console.print(f"  Systems: {len(config.systems)}")
        console.print(f"  Suites: {len(config.suites)}")
        console.print(f"  Hash: {config.config_hash()}")
    except Exception as e:
        console.print(f"[red]Config validation failed: {e}[/red]")
        raise SystemExit(1)


@cli.command()
@click.option("--fixtures-dir", default="data/fixtures", type=click.Path())
def validate_datasets(fixtures_dir: str) -> None:
    """Validate all dataset files for schema correctness."""
    from agent_bench.datasets.validator import validate_all_datasets

    results = validate_all_datasets(Path(fixtures_dir))
    all_valid = True
    for r in results:
        status = "[green]VALID[/green]" if r.valid else "[red]INVALID[/red]"
        console.print(f"  {status} {r.path.name} ({r.task_count} tasks, {len(r.warnings)} warnings, {len(r.errors)} errors)")
        for issue in r.errors:
            console.print(f"    [red]ERROR[/red] [{issue.task_id}] {issue.field}: {issue.message}")
        for issue in r.warnings[:5]:
            console.print(f"    [yellow]WARN[/yellow] [{issue.task_id}] {issue.field}: {issue.message}")
        if not r.valid:
            all_valid = False

    if all_valid:
        console.print(f"\n[green]All {len(results)} datasets valid.[/green]")
    else:
        console.print(f"\n[red]Validation failed.[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("suite_id")
@click.option("--repeat", "-n", default=None, type=int, help="Override repeat_n")
@click.option("--seed", default=None, type=int, help="Override seed")
@click.option("--output-dir", default="data/runs", type=click.Path())
@click.pass_context
def run_suite(
    ctx: click.Context, suite_id: str, repeat: int | None, seed: int | None, output_dir: str
) -> None:
    """Run a complete benchmark suite."""
    from agent_bench.runners.suite_runner import run_suite as _run_suite

    config_dir = ctx.obj["config_dir"]
    config = load_config(config_dir)
    suite_cfg = next((s for s in config.suites if s.suite_id == suite_id), None)
    if not suite_cfg:
        console.print(f"[red]Suite '{suite_id}' not found in config.[/red]")
        raise SystemExit(1)

    if repeat is not None:
        suite_cfg.repeat_n = repeat
    if seed is not None:
        suite_cfg.seed = seed

    artifact = asyncio.run(_run_suite(suite_cfg, config, Path(output_dir)))
    console.print(f"[green]Suite completed. Run ID: {artifact.run_id}[/green]")
    console.print(f"  Passed: {artifact.tasks_passed}/{artifact.tasks_total}")

    # Update leaderboard
    scorecards = artifact.metrics.get("scorecards", [])
    if scorecards:
        from agent_bench.reports.leaderboard import update_leaderboard
        update_leaderboard(artifact.run_id, suite_cfg.suite_id, scorecards)


@cli.command()
@click.argument("task_id")
@click.option("--system", required=True, help="System ID to use")
@click.option("--domain", required=True, help="Domain ID")
@click.option("--output-dir", default="data/runs", type=click.Path())
@click.pass_context
def run_case(
    ctx: click.Context, task_id: str, system: str, domain: str, output_dir: str
) -> None:
    """Run a single benchmark case."""
    from agent_bench.runners.case_runner import run_single_case

    config_dir = ctx.obj["config_dir"]
    config = load_config(config_dir)
    result = asyncio.run(run_single_case(task_id, system, domain, config, Path(output_dir)))
    if result:
        console.print(f"[green]Task {task_id}: PASSED[/green]")
    else:
        console.print(f"[red]Task {task_id}: FAILED[/red]")


@cli.command()
@click.argument("traces_file", type=click.Path(exists=True))
@click.option("--domain", default=None, help="Domain for evaluation (auto-detected if not set)")
@click.option("--weighting-profile", default="transactional_high_risk", help="Weighting profile")
@click.option("--output-dir", default="data/runs", type=click.Path())
@click.pass_context
def run_online_eval(
    ctx: click.Context, traces_file: str, domain: str | None, weighting_profile: str, output_dir: str
) -> None:
    """Evaluate pre-collected production traces (online eval)."""
    from agent_bench.runners.online_eval import run_online_eval as _run_online

    config_dir = ctx.obj["config_dir"]
    config = load_config(config_dir)
    result = asyncio.run(_run_online(Path(traces_file), config, Path(output_dir), domain=domain, weighting_profile=weighting_profile))
    console.print(f"[green]Online eval completed. Run ID: {result.run_id}[/green]")
    console.print(f"  Evaluated: {result.tasks_total} traces")
    console.print(f"  Passed: {result.tasks_passed}/{result.tasks_total}")


@cli.command()
@click.argument("run_ids", nargs=-1)
@click.option("--output", default="data/reports/comparison.md", type=click.Path())
def compare_runs(run_ids: tuple[str, ...], output: str) -> None:
    """Compare multiple benchmark runs."""
    from agent_bench.reports.comparator import compare

    if len(run_ids) < 2:
        console.print("[red]Need at least 2 run IDs to compare.[/red]")
        raise SystemExit(1)
    compare(list(run_ids), Path(output))
    console.print(f"[green]Comparison saved to {output}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "html"]))
@click.option("--output-dir", default="data/reports", type=click.Path())
def generate_report(run_id: str, fmt: str, output_dir: str) -> None:
    """Generate a report from a run."""
    if fmt == "html":
        from agent_bench.reports.html_report import generate_html_report
        out = generate_html_report(run_id, Path(output_dir))
    else:
        from agent_bench.reports.generator import generate
        out = generate(run_id, fmt, Path(output_dir))
    console.print(f"[green]Report generated: {out}[/green]")


@cli.command()
@click.argument("domain")
@click.option("--output", default=None, type=click.Path())
def export_dataset_template(domain: str, output: str | None) -> None:
    """Export a dataset template for a domain."""
    from agent_bench.datasets.exporter import export_template

    out_path = Path(output) if output else Path(f"data/fixtures/{domain}_template.yaml")
    export_template(domain, out_path)
    console.print(f"[green]Template exported: {out_path}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--task", default=None, help="Filter by task ID")
@click.option("--limit", default=50, type=int, help="Max events to show")
def view_traces(run_id: str, task: str | None, limit: int) -> None:
    """View execution traces for a run."""
    from agent_bench.cli.trace_viewer import view_traces as _view

    _view(run_id, task_id=task, limit=limit)


@cli.command()
@click.option("--domain", default=None, help="Filter by domain")
@click.option("--top", default=20, type=int, help="Number of entries to show")
def leaderboard(domain: str | None, top: int) -> None:
    """Show the local leaderboard."""
    from agent_bench.reports.leaderboard import render_leaderboard_markdown

    md = render_leaderboard_markdown(domain=domain, top_n=top)
    console.print(md)


@cli.command()
@click.argument("run_id")
@click.option("--min-global", default=0.6, type=float, help="Minimum global score")
@click.option("--min-functional", default=0.5, type=float, help="Minimum functional score")
@click.option("--min-risk", default=0.7, type=float, help="Minimum risk score")
@click.option("--max-failures", default=None, type=int, help="Maximum allowed failures")
def gate(run_id: str, min_global: float, min_functional: float, min_risk: float, max_failures: int | None) -> None:
    """CI gate: check if a run passes quality thresholds."""
    from agent_bench.cli.gate import check_gate

    passed = check_gate(run_id, min_global, min_functional, min_risk, max_failures)
    if not passed:
        raise SystemExit(1)


@cli.command()
@click.argument("version")
@click.option("--change", "-c", multiple=True, required=True, help="Change description (repeatable)")
@click.option("--author", default="", help="Author of the version")
@click.option("--version-file", default="data/governance/versions.json", type=click.Path())
def version_bump(version: str, change: tuple[str, ...], author: str, version_file: str) -> None:
    """Record a new benchmark version."""
    from agent_bench.governance.versioning import BenchmarkVersioning

    versioning = BenchmarkVersioning(Path(version_file))
    entry = versioning.record_version(version, config_hash="", changes=list(change), author=author)
    console.print(f"[green]Version recorded: {entry.version}[/green]")


@cli.command()
@click.option("--version-file", default="data/governance/versions.json", type=click.Path())
@click.option("--since", default=None, help="Show changes since this version")
def changelog(version_file: str, since: str | None) -> None:
    """Show the benchmark changelog."""
    from agent_bench.governance.versioning import BenchmarkVersioning

    versioning = BenchmarkVersioning(Path(version_file))
    md = versioning.render_changelog_markdown(since_version=since)
    console.print(md)


@cli.command()
@click.option("--fixtures-dir", default="data/fixtures", type=click.Path())
@click.option("--registry-file", default="data/governance/provenance.json", type=click.Path())
@click.option("--version", default="1.0.0", help="Version tag for registration")
def register_datasets(fixtures_dir: str, registry_file: str, version: str) -> None:
    """Register dataset files in the provenance registry."""
    from agent_bench.governance.provenance import ProvenanceRegistry

    registry = ProvenanceRegistry(Path(registry_file))
    fixtures_path = Path(fixtures_dir)

    if not fixtures_path.exists():
        console.print(f"[red]Fixtures directory not found: {fixtures_path}[/red]")
        raise SystemExit(1)

    count = 0
    for yaml_file in sorted(fixtures_path.glob("*.yaml")):
        domain = yaml_file.stem
        registry.register(yaml_file, domain=domain, version=version)
        count += 1

    console.print(f"[green]Registered {count} dataset files.[/green]")


@cli.command()
@click.option("--type", "plugin_type", default=None, help="Filter by plugin type (model/judge/tool/retrieval)")
def list_plugins(plugin_type: str | None) -> None:
    """List all registered plugins."""
    from agent_bench.utils.plugins import get_registry

    registry = get_registry()
    registry.discover_entry_points()
    plugins = registry.list_plugins(plugin_type)

    if not plugins:
        console.print("[yellow]No plugins found.[/yellow]")
        return

    console.print(f"[bold]Registered plugins ({len(plugins)}):[/bold]")
    for p in plugins:
        builtin = " [dim](builtin)[/dim]" if p.metadata.get("builtin") else ""
        console.print(f"  [{p.plugin_type}] {p.name} -> {p.module_path}:{p.class_name}{builtin}")


@cli.command()
@click.option("--runs-dir", default="data/runs", type=click.Path())
@click.option("--by", "group_by", default="system", type=click.Choice(["system", "domain"]))
@click.option("--system", default=None, help="Filter by system ID (for trends)")
@click.option("--domain", default=None, help="Filter by domain (for cross-system)")
def analytics(runs_dir: str, group_by: str, system: str | None, domain: str | None) -> None:
    """Run analytics queries across benchmark runs."""
    from agent_bench.storage.analytics import (
        aggregate_by_domain,
        aggregate_by_system,
        cross_system_comparison,
        trend_over_runs,
    )

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        console.print("[yellow]No runs directory found.[/yellow]")
        return

    if domain:
        results = cross_system_comparison(runs_path, domain)
        console.print(f"[bold]Cross-system comparison for domain: {domain}[/bold]")
    elif system:
        results = trend_over_runs(runs_path, system)
        console.print(f"[bold]Score trend for system: {system}[/bold]")
    elif group_by == "domain":
        results = aggregate_by_domain(runs_path)
        console.print("[bold]Aggregation by domain:[/bold]")
    else:
        results = aggregate_by_system(runs_path)
        console.print("[bold]Aggregation by system:[/bold]")

    if not results:
        console.print("  [yellow]No data found.[/yellow]")
        return

    for r in results:
        parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()]
        console.print(f"  {' | '.join(parts)}")


if __name__ == "__main__":
    cli()
