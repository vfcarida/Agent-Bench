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
    except (ValueError, OSError, FileNotFoundError) as e:
        console.print(f"[red]Config validation failed: {e}[/red]")
        raise SystemExit(1)


@cli.command()
@click.option("--fixtures-dir", "--data-dir", default="datasets/gold/dev", type=click.Path(), help="Directory containing dataset files")
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
        console.print("\n[red]Validation failed.[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("suite_id")
@click.option("--repeat", "-n", default=None, type=int, help="Override repeat_n")
@click.option("--seed", default=None, type=int, help="Override seed")
@click.option("--split", default=None, help="Override dataset split (e.g., dev, holdout, calibration)")
@click.option("--output-dir", default="data/runs", type=click.Path())
@click.option(
    "--runner",
    default="auto",
    type=click.Choice(["auto", "scripted", "stub"]),
    help="Agent runner implementation (default: auto)",
)
@click.option(
    "--enable-llm-judge",
    is_flag=True,
    default=False,
    help="[EXPERIMENTAL] Enable SemanticJudge in Phase 2. Requires --llm-judge-system."
         " Only use after a GO decision from the calibration experiment.",
)
@click.option(
    "--llm-judge-system",
    default=None,
    help="System ID whose model acts as the LLM judge (requires --enable-llm-judge).",
)
@click.option(
    "--concurrency",
    "-c",
    default=1,
    type=int,
    help="Maximum concurrent tasks to evaluate simultaneously (default: 1).",
)
@click.pass_context
def run_suite(
    ctx: click.Context,
    suite_id: str,
    repeat: int | None,
    seed: int | None,
    split: str | None,
    output_dir: str,
    runner: str,
    enable_llm_judge: bool,
    llm_judge_system: str | None,
    concurrency: int,
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
    if split is not None:
        suite_cfg.split = split

    from agent_bench.models.factory import ConfigError

    if enable_llm_judge and not llm_judge_system:
        console.print("[red]--enable-llm-judge requires --llm-judge-system to be specified.[/red]")
        raise SystemExit(1)

    try:
        artifact = asyncio.run(
            _run_suite(
                suite_cfg,
                config,
                Path(output_dir),
                runner_type=runner,
                enable_llm_judge=enable_llm_judge,
                llm_judge_system_id=llm_judge_system,
                concurrency=concurrency,
            )
        )
    except ConfigError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise SystemExit(1)
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
@click.option("--split", default="dev", help="Dataset split (e.g., dev, holdout, calibration)")
@click.option("--output-dir", default="data/runs", type=click.Path())
@click.pass_context
def run_case(
    ctx: click.Context, task_id: str, system: str, domain: str, split: str, output_dir: str
) -> None:
    """Run a single benchmark case."""
    from agent_bench.models.factory import ConfigError
    from agent_bench.runners.case_runner import run_single_case

    config_dir = ctx.obj["config_dir"]
    config = load_config(config_dir)
    try:
        result = asyncio.run(run_single_case(task_id, system, domain, config, Path(output_dir), split=split))
    except ConfigError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise SystemExit(1)
    if result:
        console.print(f"[green]Task {task_id}: PASSED[/green]")
    else:
        console.print(f"[red]Task {task_id}: FAILED[/red]")


@cli.command()
@click.option("--threshold", default=0.80, type=float, help="Jaccard similarity threshold")
@click.option("--holdout-dir", default=None, type=click.Path(), help="Holdout directory path")
@click.option("--dev-dir", default=None, type=click.Path(), help="Dev directory path")
@click.option("--synthetic-dir", default=None, type=click.Path(), help="Synthetic directory path")
def check_contamination(
    threshold: float,
    holdout_dir: str | None,
    dev_dir: str | None,
    synthetic_dir: str | None,
) -> None:
    """Check for data leakage/contamination between holdout and dev/synthetic datasets."""
    from agent_bench.validators.contamination import check_contamination as _check_contamination

    report = _check_contamination(
        holdout_dir=Path(holdout_dir) if holdout_dir else None,
        dev_dir=Path(dev_dir) if dev_dir else None,
        synthetic_dir=Path(synthetic_dir) if synthetic_dir else None,
        threshold=threshold,
    )
    console.print(report.summary_text)
    if not report.passed:
        console.print("\n[red]Contamination check failed: Data leakage detected![/red]")
        raise SystemExit(1)
    console.print("\n[green]Contamination check passed: Zero leakage detected.[/green]")


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
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
def generate_report(run_id: str, fmt: str, output_dir: str, runs_dir: str) -> None:
    """Generate a report from a run."""
    if fmt == "html":
        from agent_bench.reports.html_report import generate_html_report
        out = generate_html_report(run_id, Path(output_dir), runs_dir=Path(runs_dir))
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
@click.option("--domain", default=None, help="Filter by domain (e.g. pix_assist, cyber_sandbox). Omit for all.")
@click.option("--split", default="dev", type=click.Choice(["dev", "holdout"]), help="Dataset split")
@click.option("--output", default=None, type=click.Path(), help="Target JSONL file path")
def export_inspect(domain: str | None, split: str, output: str | None) -> None:
    """Export benchmark tasks to UK AISI Inspect AI dataset JSONL format."""
    from agent_bench.datasets.loader import load_domain_tasks
    from agent_bench.export.inspect_ai import export_tasks_to_inspect_dataset

    domains = [domain] if domain else ["pix_assist", "investment_advisor", "sme_business_advisor", "cyber_sandbox"]
    all_tasks = []
    for d in domains:
        try:
            tasks = load_domain_tasks(d, split=split)
            all_tasks.extend(tasks)
        except Exception:
            pass

    if not all_tasks:
        console.print("[yellow]No tasks found matching criteria.[/yellow]")
        return

    out_path = Path(output) if output else Path(f"data/reports/inspect_{domain or 'all'}_{split}.jsonl")
    samples = export_tasks_to_inspect_dataset(all_tasks, output_path=out_path)
    console.print(f"[green]Exported {len(samples)} tasks to Inspect AI dataset: {out_path}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
@click.option("--output", default=None, type=click.Path(), help="Target Inspect log JSON file path")
def export_inspect_log(run_id: str, runs_dir: str, output: str | None) -> None:
    """Export a run artifact and execution traces to Inspect AI evaluation log JSON."""
    import json
    from datetime import UTC, datetime
    from typing import Any

    from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
    from agent_bench.export.inspect_ai import run_artifact_to_inspect_log

    run_path = Path(runs_dir) / run_id
    if not run_path.exists():
        console.print(f"[red]Run directory '{run_path}' not found.[/red]")
        return

    manifest_file = run_path / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_file.exists():
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)

    traces_file = run_path / "traces.jsonl"
    traces: list[TraceEvent] = []
    if traces_file.exists():
        with open(traces_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                et = d.get("event_type", "system_event")
                try:
                    event_type = TraceEventType(et)
                except ValueError:
                    event_type = TraceEventType.SYSTEM_EVENT
                traces.append(
                    TraceEvent(
                        event_type=event_type,
                        timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(UTC),
                        data=d.get("data", {}),
                        event_id=d.get("event_id", ""),
                        parent_id=d.get("parent_id"),
                    )
                )

    artifact = RunArtifact(
        run_id=manifest.get("run_id", run_id),
        suite_id=manifest.get("suite_id", ""),
        system_id=manifest.get("system_id", ""),
        model_id=manifest.get("model_id", ""),
        tasks_total=manifest.get("tasks_total", 0),
        tasks_passed=manifest.get("tasks_passed", 0),
        tasks_failed=manifest.get("tasks_failed", 0),
        traces=traces,
        metrics=manifest.get("metrics", {}),
        metadata=manifest.get("metadata", {}),
    )
    artifact.finalize()

    log_data = run_artifact_to_inspect_log(artifact)
    out_file = Path(output) if output else Path(f"data/reports/inspect_log_{run_id}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Inspect AI evaluation log exported: {out_file}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--task-id", "--task", "task_id", default=None, help="Filter by task ID")
@click.option("--step", default=None, type=int, help="Filter by step index")
@click.option("--limit", default=50, type=int, help="Max events to show")
def view_traces(run_id: str, task_id: str | None, step: int | None, limit: int) -> None:
    """View execution traces for a run."""
    from agent_bench.cli.trace_viewer import view_traces as _view

    _view(run_id, task_id=task_id, step=step, limit=limit)


@cli.command()
@click.option("--domain", default=None, help="Filter leaderboard by domain (e.g. pix_assist, cyber_sandbox)")
@click.option("--top", "top_n", default=20, type=int, help="Number of top entries to display")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "markdown", "html", "json"]),
    help="Output presentation format",
)
@click.option(
    "--leaderboard-path",
    type=click.Path(),
    default="data/reports/leaderboard.json",
    help="Path to leaderboard JSON file",
)
def leaderboard(domain: str | None, top_n: int, output_format: str, leaderboard_path: str) -> None:
    """Display system ranking leaderboard across historical benchmark runs."""
    import json

    from rich.table import Table

    from agent_bench.reports.leaderboard import (
        get_leaderboard,
        render_leaderboard_html,
        render_leaderboard_markdown,
    )

    lb_path = Path(leaderboard_path)
    entries = get_leaderboard(domain=domain, top_n=top_n, leaderboard_path=lb_path)

    if not entries:
        console.print(
            f"[yellow]No leaderboard entries found at '{leaderboard_path}'. Run benchmark suites first to generate rankings.[/yellow]"
        )
        return

    if output_format == "markdown":
        console.print(render_leaderboard_markdown(domain=domain, top_n=top_n, leaderboard_path=lb_path))
        return

    if output_format == "html":
        console.print(render_leaderboard_html(domain=domain, top_n=top_n, leaderboard_path=lb_path))
        return

    if output_format == "json":
        console.print(json.dumps(entries, indent=2))
        return

    title = "Agent-Bench Leaderboard"
    if domain:
        title += f" — Domain: {domain}"

    table = Table(title=title)
    table.add_column("Rank", style="bold", justify="center")
    table.add_column("System ID", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Global", style="bold green", justify="right")
    table.add_column("Functional", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Reliability", justify="right")
    table.add_column("Run ID", style="dim")

    for i, e in enumerate(entries, 1):
        medal = {1: "1 🥇", 2: "2 🥈", 3: "3 🥉"}.get(i, str(i))
        table.add_row(
            medal,
            str(e.get("system_id", "—")),
            str(e.get("domain", "—")),
            f"{e.get('global_score', 0.0):.3f}",
            f"{e.get('functional_score', 0.0):.3f}",
            f"{e.get('risk_score', 0.0):.3f}",
            f"{e.get('cost_score', 0.0):.3f}",
            f"{e.get('latency_score', 0.0):.3f}",
            f"{e.get('reliability_score', 0.0):.3f}",
            str(e.get("run_id", "—"))[:8],
        )

    console.print(table)


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
@click.option("--fixtures-dir", "--data-dir", default="datasets/gold/dev", type=click.Path(), help="Directory containing dataset files")
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


@cli.command()
@click.option(
    "--annotations",
    type=click.Path(exists=True),
    default="datasets/gold/calibration/annotator_agreement.yaml",
    help="Path to YAML/JSON multi-annotator ratings file",
)
def check_agreement(annotations: str) -> None:
    """Calculate inter-annotator agreement (Cohen's Kappa & Krippendorff's Alpha)."""
    from rich.table import Table

    from agent_bench.metrics.inter_annotator import evaluate_annotation_dataset, interpret_agreement

    report = evaluate_annotation_dataset(Path(annotations))

    console.print(f"\n[bold]Inter-Annotator Agreement Report[/bold] ({report['total_items']} items, {len(report['raters'])} raters)")
    console.print(f"  Raters: {', '.join(report['raters'])}")
    console.print(f"  Krippendorff's Alpha: [bold green]{report['krippendorffs_alpha']:.4f}[/bold green] ([dim]{report['alpha_interpretation']}[/dim])")
    console.print(f"  Mean Cohen's Kappa:   [bold green]{report['mean_cohens_kappa']:.4f}[/bold green] ([dim]{report['kappa_interpretation']}[/dim])\n")

    table = Table(title="Pairwise Cohen's Kappa Matrix")
    table.add_column("Rater Pair", style="cyan")
    table.add_column("Kappa", style="green", justify="right")
    table.add_column("Interpretation", style="yellow")

    for pair, score in report["pairwise_cohens_kappa"].items():
        table.add_row(pair, f"{score:.4f}", interpret_agreement(score))

    console.print(table)


if __name__ == "__main__":
    cli()



