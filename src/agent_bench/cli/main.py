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
@click.option(
    "--sprt",
    "--early-stopping",
    "enable_sprt",
    is_flag=True,
    default=False,
    help="Enable Wald's Sequential Probability Ratio Test for early stopping on conclusive task outcomes.",
)
@click.option(
    "--user-simulator",
    "user_sim_type",
    type=click.Choice(["none", "scripted", "rule_based", "dual_control", "human"]),
    default="none",
    help="Attach interactive user simulator for multi-turn conversational benchmark tasks.",
)
@click.option(
    "--fast-anchor",
    is_flag=True,
    default=False,
    help="Evaluate only an IRT-stratified anchor subset of tasks for fast score estimation.",
)
@click.option(
    "--anchor-fraction",
    default=0.25,
    type=float,
    help="Fraction of suite tasks to evaluate in --fast-anchor mode (default: 0.25).",
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
    enable_sprt: bool,
    user_sim_type: str,
    fast_anchor: bool,
    anchor_fraction: float,
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

    from agent_bench.core.user_simulator import UserSimulator

    user_simulator: UserSimulator | None = None
    if user_sim_type == "scripted":
        from agent_bench.core.user_simulator import ScriptedUserSimulator

        user_simulator = ScriptedUserSimulator()
    elif user_sim_type == "rule_based":
        from agent_bench.core.user_simulator import RuleBasedUserSimulator

        user_simulator = RuleBasedUserSimulator()
    elif user_sim_type == "dual_control":
        from agent_bench.core.user_simulator import DualControlUserSimulator

        user_simulator = DualControlUserSimulator()
    elif user_sim_type == "human":
        from agent_bench.core.user_simulator import HumanUserSimulator

        user_simulator = HumanUserSimulator()

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
                user_simulator=user_simulator,
                enable_sprt=enable_sprt,
                fast_anchor=fast_anchor,
                anchor_fraction=anchor_fraction,
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
@click.option(
    "--user-simulator",
    "user_sim_type",
    type=click.Choice(["none", "scripted", "rule_based", "dual_control", "human"]),
    default="none",
    help="Attach interactive user simulator (human for terminal red-teaming).",
)
@click.pass_context
def run_case(
    ctx: click.Context,
    task_id: str,
    system: str,
    domain: str,
    split: str,
    output_dir: str,
    user_sim_type: str,
) -> None:
    """Run a single benchmark case."""
    from agent_bench.core.user_simulator import (
        DualControlUserSimulator,
        HumanUserSimulator,
        RuleBasedUserSimulator,
        ScriptedUserSimulator,
        UserSimulator,
    )
    from agent_bench.models.factory import ConfigError
    from agent_bench.runners.case_runner import run_single_case

    user_simulator: UserSimulator | None = None
    if user_sim_type == "scripted":
        user_simulator = ScriptedUserSimulator()
    elif user_sim_type == "rule_based":
        user_simulator = RuleBasedUserSimulator()
    elif user_sim_type == "dual_control":
        user_simulator = DualControlUserSimulator()
    elif user_sim_type == "human":
        user_simulator = HumanUserSimulator()

    config_dir = ctx.obj["config_dir"]
    config = load_config(config_dir)
    try:
        result = asyncio.run(
            run_single_case(
                task_id,
                system,
                domain,
                config,
                Path(output_dir),
                split=split,
                user_simulator=user_simulator,
            )
        )
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
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
def compare_runs(run_ids: tuple[str, ...], output: str, runs_dir: str) -> None:
    """Compare multiple benchmark runs."""
    from agent_bench.reports.comparator import compare

    if len(run_ids) < 2:
        console.print("[red]Need at least 2 run IDs to compare.[/red]")
        raise SystemExit(1)
    compare(list(run_ids), Path(output), runs_dir=Path(runs_dir))
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
        out = generate(run_id, fmt, Path(output_dir), runs_dir=Path(runs_dir))
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


@cli.command()
@click.option("--count", default=20, type=int, help="Number of security cases to generate.")
@click.option("--domain", default=None, help="Target domain (e.g. cyber_sandbox, investment_advisor, pix_assist).")
@click.option("--lang", default="auto", type=click.Choice(["auto", "en", "pt"]), help="Target prompt language.")
@click.option("--output", "-o", default="datasets/synthetic/security_generated.yaml", type=click.Path(), help="Output YAML file path.")
@click.option("--seed", default=42, type=int, help="Random generation seed.")
def generate_security_suite(
    count: int,
    domain: str | None,
    lang: str,
    output: str,
    seed: int,
) -> None:
    """Generate a batch of adversarial security test cases aligned with OWASP Top 10."""
    import yaml

    from agent_bench.generators.security import SecurityGenerator

    generator = SecurityGenerator()
    results = generator.generate_batch(count=count, domain=domain, lang=lang, seed=seed)
    valid_cases = [r.case for r in results if not r.rejected]

    out_p = Path(output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.dump({"cases": valid_cases}, f, sort_keys=False, allow_unicode=True)

    console.print(f"[green]Generated {len(valid_cases)} adversarial test cases saved to {out_p}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Output presentation format (table or json)",
)
def diagnose_run(run_id: str, runs_dir: str, output_format: str) -> None:
    """Compute Pareto frontier, Graduated Autonomy tiers, and psychometric diagnostics for a run."""
    import json
    from typing import Any

    from rich.table import Table

    from agent_bench.metrics.difficulty_analysis import compute_difficulty_profile
    from agent_bench.metrics.discriminative import compute_discriminative_power
    from agent_bench.metrics.pareto import compute_pareto_frontier

    base_runs_dir = Path(runs_dir)
    potential = Path(run_id)
    if potential.is_file():
        artifact_path = potential
    else:
        artifact_path = base_runs_dir / f"{run_id}.json"
        if not artifact_path.exists():
            matches = list(base_runs_dir.glob(f"{run_id}*.json"))
            if matches:
                artifact_path = matches[0]
            else:
                console.print(f"[red]Run artifact not found: {run_id}[/red]")
                raise SystemExit(1)

    with open(artifact_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    raw_metrics: dict[str, Any] = data.get("metrics", {})
    scorecards: list[dict[str, Any]] = data.get("scorecards", []) or raw_metrics.get("scorecards", [])
    task_results: list[dict[str, Any]] = raw_metrics.get("task_results", [])
    diagnostics: dict[str, Any] = raw_metrics.get("diagnostics", {})

    pareto = diagnostics.get("pareto_frontier") if isinstance(diagnostics, dict) else None
    if pareto is None:
        pareto = compute_pareto_frontier(scorecards)

    by_system: dict[str, list[dict[str, Any]]] = {}
    for tr in task_results:
        sys_id = str(tr.get("system_id", data.get("system_id", "default")))
        by_system.setdefault(sys_id, []).append(tr)

    diff_profiles = diagnostics.get("difficulty_profiles") if isinstance(diagnostics, dict) else None
    if diff_profiles is None and by_system:
        diff_profiles = compute_difficulty_profile(by_system)

    disc_power = diagnostics.get("discriminative_power") if isinstance(diagnostics, dict) else None
    disc_matrix: dict[str, dict[str, bool]] = {
        s_id: {str(r.get("task_id", "")): bool(r.get("passed", False)) for r in res}
        for s_id, res in by_system.items()
    }
    if disc_power is None and by_system:
        disc_power = compute_discriminative_power(disc_matrix)

    irt_analysis = diagnostics.get("irt_analysis") if isinstance(diagnostics, dict) else None
    if irt_analysis is None and disc_matrix:
        from agent_bench.metrics.irt import fit_2pl_irt, select_irt_anchor_tasks

        irt_fit = fit_2pl_irt(disc_matrix)
        irt_analysis = {
            "model_fit": irt_fit.to_dict(),
            "recommended_anchor": select_irt_anchor_tasks(irt_fit, anchor_fraction=0.25),
        }

    payload = {
        "run_id": data.get("run_id", run_id),
        "pareto_frontier": pareto,
        "difficulty_profiles": diff_profiles or {},
        "discriminative_power": disc_power or {},
        "irt_analysis": irt_analysis or {},
    }

    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print(f"\n[bold]Run Diagnostics — {payload['run_id']}[/bold]")
    console.print(
        f"  Pareto 2D Hypervolume Indicator: [bold green]{pareto.get('hypervolume_2d', 0.0):.4f}[/bold green]"
    )

    table = Table(title="Safety-Constrained Pareto Frontier & Graduated Autonomy Tiers")
    table.add_column("System ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Quality", justify="right")
    table.add_column("Pass^3", justify="right")
    table.add_column("Cost/Success (USD)", justify="right")
    table.add_column("Autonomy Tier", style="magenta")

    for f_sys in pareto.get("frontier", []):
        table.add_row(
            str(f_sys.get("system_id")),
            "[green]PARETO-OPTIMAL[/green]",
            f"{float(f_sys.get('quality', 0.0)):.3f}",
            f"{float(f_sys.get('pass_hat_3', 0.0)):.3f}",
            f"${float(f_sys.get('cost_usd', 0.0)):.4f}",
            str(f_sys.get("autonomy_label", f_sys.get("autonomy_tier", "N/A"))),
        )
    for d_sys in pareto.get("dominated", []):
        dom_by = ", ".join(d_sys.get("dominated_by", []))
        table.add_row(
            str(d_sys.get("system_id")),
            f"[yellow]DOMINATED (by {dom_by})[/yellow]",
            f"{float(d_sys.get('quality', 0.0)):.3f}",
            f"{float(d_sys.get('pass_hat_3', 0.0)):.3f}",
            f"${float(d_sys.get('cost_usd', 0.0)):.4f}",
            str(d_sys.get("autonomy_label", d_sys.get("autonomy_tier", "N/A"))),
        )
    for u_sys in pareto.get("safety_disqualified", []):
        table.add_row(
            str(u_sys.get("system_id")),
            "[red]SAFETY-DISQUALIFIED[/red]",
            f"{float(u_sys.get('quality', 0.0)):.3f}",
            f"{float(u_sys.get('pass_hat_3', 0.0)):.3f}",
            f"${float(u_sys.get('cost_usd', 0.0)):.4f}",
            str(u_sys.get("autonomy_label", "Blocked")),
        )
    console.print(table)

    if disc_power and disc_power.get("total_tasks", 0) > 0:
        console.print(
            f"\n[bold]Psychometric Discriminative Power[/bold] "
            f"(Tasks={disc_power['total_tasks']}, Systems={disc_power['total_systems']})"
        )
        console.print(
            f"  Mid-Band (10-90%) Rate: {disc_power.get('mid_band_10_90_rate', 0.0):.1%} | "
            f"Gini: {disc_power.get('gini_coefficient', 0.0):.3f} | "
            f"Item Discrimination: {disc_power.get('avg_item_discrimination', 0.0):.3f}"
        )
        for alert in disc_power.get("alerts", []):
            console.print(f"  [yellow]Alert:[/yellow] {alert}")

    if irt_analysis and isinstance(irt_analysis, dict):
        rec_anchor = irt_analysis.get("recommended_anchor", {})
        anchor_ids = rec_anchor.get("anchor_task_ids", [])
        if anchor_ids:
            console.print(
                f"\n[bold]2PL IRT Fast-Anchor Subset[/bold] "
                f"(Count={len(anchor_ids)}/{rec_anchor.get('total_tasks', 0)}, "
                f"Info Retention={rec_anchor.get('information_retention_ratio', 0.0):.1%})"
            )
            console.print(f"  Anchor Tasks: {', '.join(str(x) for x in anchor_ids[:10])}")


@cli.command("export-otel")
@click.argument("run_id")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output OTLP GenAI JSON file path")
def export_otel(run_id: str, runs_dir: str, output: str | None) -> None:
    """Export a benchmark run in OpenTelemetry GenAI Semantic Conventions (OTLP/JSON)."""
    import json
    from typing import Any

    from agent_bench.utils.observability import convert_run_traces_to_otel_genai

    base_runs_dir = Path(runs_dir)
    artifact_path = base_runs_dir / f"{run_id}.json"
    if not artifact_path.exists():
        matches = list(base_runs_dir.glob(f"{run_id}*.json"))
        if matches:
            artifact_path = matches[0]
        else:
            console.print(f"[red]Run artifact not found: {run_id}[/red]")
            raise SystemExit(1)

    with open(artifact_path, encoding="utf-8") as f:
        manifest: dict[str, Any] = json.load(f)

    traces_list: list[dict[str, Any]] = []
    traces_file = base_runs_dir / str(manifest.get("run_id", run_id)) / "traces.jsonl"
    if traces_file.exists():
        for line in traces_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces_list.append(json.loads(line))

    otlp_payload = convert_run_traces_to_otel_genai(manifest, traces_list)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(otlp_payload, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Exported OTLP GenAI spans to {out_path}[/green]")
    else:
        click.echo(json.dumps(otlp_payload, indent=2, default=str))


@cli.command("replay-counterfactual")
@click.argument("run_id")
@click.option("--task-id", required=True, help="Task ID to fork and replay counterfactually")
@click.option("--domain", default="pix_assist", help="Task domain")
@click.option("--fork-step", default=1, type=int, help="Trajectory step index to fork at")
@click.option("--mutate-state", default="{}", help="JSON dictionary of counterfactual state mutations")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
def replay_counterfactual(
    run_id: str,
    task_id: str,
    domain: str,
    fork_step: int,
    mutate_state: str,
    runs_dir: str,
) -> None:
    """Replay a recorded trajectory from fork_step with a counterfactual state mutation."""
    import json
    from typing import Any

    from agent_bench.core.scenarios import Task
    from agent_bench.datasets.loader import load_domain_tasks
    from agent_bench.runners.counterfactual import (
        CounterfactualIntervention,
        CounterfactualSandbox,
    )

    base_runs_dir = Path(runs_dir)
    traces_file = base_runs_dir / run_id / "traces.jsonl"
    traces_list: list[dict[str, Any]] = []
    if traces_file.exists():
        for line in traces_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("task_id") in (None, task_id) or rec.get("data", {}).get("task_id") in (
                    None,
                    task_id,
                ):
                    traces_list.append(rec)

    tasks = load_domain_tasks(domain)
    task = next((t for t in tasks if t.task_id == task_id), None)
    if task is None:
        task = Task(
            task_id=task_id,
            domain=domain,
            name=task_id,
            description=f"Counterfactual task {task_id}",
            input_messages=[{"role": "user", "content": "Execute task"}],
        )

    state_mutations: dict[str, Any] = json.loads(mutate_state)
    sandbox = CounterfactualSandbox()
    intervention = CounterfactualIntervention(
        fork_step=fork_step,
        intervention_type="mutate_state",
        state_mutations=state_mutations,
    )
    res = asyncio.run(sandbox.replay_with_intervention(task, traces_list, intervention))
    click.echo(json.dumps(res.to_dict(), indent=2, default=str))


@cli.command("monitor-drift")
@click.argument("baseline_run_id")
@click.argument("shadow_path", type=click.Path(exists=True))
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing baseline run")
def monitor_drift(baseline_run_id: str, shadow_path: str, runs_dir: str) -> None:
    """Monitor shadow traffic (JSONL or OTLP JSON) for CUSUM & JSD drift against a baseline run."""
    import json
    from typing import Any

    from agent_bench.runners.online_eval import OnlineDriftMonitor, load_traces_from_jsonl

    base_runs_dir = Path(runs_dir)
    artifact_path = base_runs_dir / f"{baseline_run_id}.json"
    if not artifact_path.exists():
        console.print(f"[red]Baseline run artifact not found: {baseline_run_id}[/red]")
        raise SystemExit(1)

    baseline_manifest: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    shadow_p = Path(shadow_path)
    if shadow_p.suffix.lower() == ".jsonl":
        shadow_data: Any = load_traces_from_jsonl(shadow_p)
    else:
        raw_text = shadow_p.read_text(encoding="utf-8").strip()
        parsed = json.loads(raw_text)
        shadow_data = parsed if isinstance(parsed, (list, dict)) else []

    monitor = OnlineDriftMonitor()
    report = monitor.evaluate_shadow_traffic(
        baseline_manifest, shadow_data, shadow_source=str(shadow_p)
    )
    click.echo(json.dumps(report.to_dict(), indent=2, default=str))


@cli.command("aggregate-federated")
@click.argument("submissions_path", type=click.Path(exists=True))
@click.option("--epsilon", default=1.0, type=float, help="Differential privacy epsilon budget")
@click.option("--delta", default=0.0, type=float, help="Differential privacy delta budget")
@click.option("--min-sites", default=2, type=int, help="Minimum k-anonymity participating sites")
@click.option("--seed", default=42, type=int, help="Deterministic RNG seed for Laplace noise")
def aggregate_federated(
    submissions_path: str,
    epsilon: float,
    delta: float,
    min_sites: int,
    seed: int,
) -> None:
    """Aggregate multi-institution benchmark submissions under (epsilon, delta)-Differential Privacy."""
    import json
    from typing import Any

    from agent_bench.governance.federated_aggregator import FederatedBenchmarkAggregator

    sub_p = Path(submissions_path)
    raw = json.loads(sub_p.read_text(encoding="utf-8"))
    submissions: list[dict[str, Any]] = (
        raw if isinstance(raw, list) else list(raw.get("submissions", []))
    )

    aggregator = FederatedBenchmarkAggregator(min_sites=min_sites)
    report = aggregator.aggregate(submissions, epsilon=epsilon, delta=delta, seed=seed)
    click.echo(json.dumps(report.to_dict(), indent=2, default=str))


@cli.command("attest-run")
@click.argument("run_id")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
@click.option("--signing-key", default="agent-bench-default-audit-key", help="HMAC-SHA256 signing key")
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def attest_run_cmd(run_id: str, runs_dir: str, signing_key: str, output: str) -> None:
    """Generate a SHA-256 Merkle Tree & Selective-Disclosure attestation bundle for a run."""
    import json
    from typing import Any

    from agent_bench.governance.zk_attestation import MerkleExecutionAttestor

    base_runs_dir = Path(runs_dir)
    artifact_path = Path(run_id) if Path(run_id).is_file() else base_runs_dir / f"{run_id}.json"
    if not artifact_path.exists():
        console.print(f"[red]Run artifact not found: {run_id}[/red]")
        raise SystemExit(1)

    manifest: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    bundle = MerkleExecutionAttestor.attest_run(manifest, signing_key=signing_key)
    payload = json.dumps(bundle.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-attestation")
@click.argument("attestation_path", type=click.Path(exists=True))
@click.option("--signing-key", default="agent-bench-default-audit-key", help="HMAC-SHA256 signing key")
def verify_attestation_cmd(attestation_path: str, signing_key: str) -> None:
    """Verify HMAC signature and all MerkleInclusionProofs in an attestation bundle."""
    import json
    from typing import Any

    from agent_bench.governance.zk_attestation import MerkleExecutionAttestor

    raw: dict[str, Any] = json.loads(Path(attestation_path).read_text(encoding="utf-8"))
    res = MerkleExecutionAttestor.verify_attestation_bundle(raw, signing_key=signing_key)
    click.echo(json.dumps(res, indent=2, default=str))
    if not res.get("verified", False):
        raise SystemExit(1)


@cli.command("evolve-redteam")
@click.option("--domain", default="pix_assist", help="Target domain for evolutionary red-teaming")
@click.option("--generations", default=3, type=int, help="Number of evolutionary generations")
@click.option("--offspring", default=8, type=int, help="Offspring evaluated per generation")
@click.option("--seed", default=42, type=int, help="Deterministic RNG seed")
@click.option("--output", "-o", default="", help="Optional output JSON/YAML file path")
def evolve_redteam_cmd(
    domain: str,
    generations: int,
    offspring: int,
    seed: int,
    output: str,
) -> None:
    """Run closed-loop 2PL IRT-guided evolutionary red-team fuzzing (MAP-Elites archive)."""
    import json

    from agent_bench.generators.evolutionary_fuzzer import EvolutionaryRedTeamFuzzer

    fuzzer = EvolutionaryRedTeamFuzzer(domain=domain, seed=seed)
    report = fuzzer.evolve(generations=generations, offspring_per_generation=offspring)
    payload = json.dumps(report, indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("attest-tee")
@click.argument("run_id")
@click.option("--runs-dir", default="data/runs", type=click.Path(), help="Directory containing run artifacts")
@click.option("--tee-arch", default="aws_nitro_enclave", help="TEE architecture (aws_nitro_enclave, intel_tdx, amd_sev_snp)")
@click.option("--nonce", default="default_auditor_nonce_2026", help="Auditor anti-replay nonce")
@click.option("--signing-key", default="agent-bench-tee-hardware-root-key", help="Enclave hardware signing key")
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def attest_tee_cmd(
    run_id: str,
    runs_dir: str,
    tee_arch: str,
    nonce: str,
    signing_key: str,
    output: str,
) -> None:
    """Bind a run's Merkle root into a hardware-enforced TEE Enclave Attestation Quote."""
    import json
    from typing import Any

    from agent_bench.governance.tee_enclave_attestation import ConfidentialEnclaveAttestor

    base_runs_dir = Path(runs_dir)
    artifact_path = Path(run_id) if Path(run_id).is_file() else base_runs_dir / f"{run_id}.json"
    if not artifact_path.exists():
        console.print(f"[red]Run artifact not found: {run_id}[/red]")
        raise SystemExit(1)

    manifest: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    quote = ConfidentialEnclaveAttestor.generate_tee_quote(
        manifest,
        tee_architecture=tee_arch,
        nonce=nonce,
        enclave_signing_key=signing_key,
    )
    payload = json.dumps(quote.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-tee-quote")
@click.argument("quote_path", type=click.Path(exists=True))
@click.option("--signing-key", default="agent-bench-tee-hardware-root-key", help="Enclave hardware signing key")
@click.option("--expected-nonce", default="", help="Optional expected auditor nonce")
def verify_tee_quote_cmd(quote_path: str, signing_key: str, expected_nonce: str) -> None:
    """Verify a TEE Enclave Attestation Quote, PCR measurements, and bound Merkle proofs."""
    import json
    from typing import Any

    from agent_bench.governance.tee_enclave_attestation import ConfidentialEnclaveAttestor

    raw: dict[str, Any] = json.loads(Path(quote_path).read_text(encoding="utf-8"))
    verdict = ConfidentialEnclaveAttestor.verify_tee_quote(
        raw,
        enclave_signing_key=signing_key,
        expected_nonce=expected_nonce or None,
    )
    click.echo(json.dumps(verdict, indent=2, default=str))
    if not verdict.get("verified", False):
        raise SystemExit(1)


@cli.command("stress-memory")
@click.option("--distractors", default=8, type=int, help="Number of distractor turns to inject")
@click.option("--compaction-interval", default=4, type=int, help="Turns between context compaction boundaries")
@click.option("--system-id", default="episodic_agent", help="System identifier for the stress report")
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def stress_memory_cmd(
    distractors: int,
    compaction_interval: int,
    system_id: str,
    output: str,
) -> None:
    """Run a long-horizon episodic memory consolidation and interference stress test."""
    import json

    from agent_bench.runners.episodic_memory_stress import (
        EpisodicMemoryStressRunner,
        MemoryInterferenceInjector,
    )

    injector = MemoryInterferenceInjector(
        n_distractor_turns=distractors,
        compaction_interval=compaction_interval,
    )
    runner = EpisodicMemoryStressRunner(injector=injector)
    report = runner.run_stress_test_sync(system_id=system_id)
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("audit-collusion")
@click.argument("committee_log_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def audit_collusion_cmd(committee_log_path: str, output: str) -> None:
    """Audit a multi-agent committee voting log for Sybil collusion, steganography, and Shapley values."""
    import json
    from typing import Any

    from agent_bench.governance.collusion_detector import MultiAgentCollusionDetector

    raw: dict[str, Any] = json.loads(Path(committee_log_path).read_text(encoding="utf-8"))
    agent_votes = raw.get("agent_votes", {})
    ground_truth = raw.get("ground_truth", [])
    messages = raw.get("inter_agent_messages", [])

    detector = MultiAgentCollusionDetector()
    report = detector.audit_committee(
        agent_votes,
        ground_truth,
        inter_agent_messages=messages,
    )
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("compile-zk-circuit")
@click.argument("run_path", type=click.Path(exists=True))
@click.option("--threshold-score", default=0.70, type=float, help="Minimum required global score [0.0, 1.0]")
@click.option("--min-tier", default=2, type=int, help="Minimum required graduated autonomy tier")
@click.option("--output", "-o", default="", help="Optional output JSON bundle path")
def compile_zk_circuit_cmd(
    run_path: str,
    threshold_score: float,
    min_tier: int,
    output: str,
) -> None:
    """Compile an evaluation run into a Goldilocks R1CS arithmetic circuit and Fiat-Shamir witness proof."""
    import json
    from typing import Any

    from agent_bench.governance.zk_circuit_compiler import ZKEvaluationCircuitCompiler

    raw: dict[str, Any] = json.loads(Path(run_path).read_text(encoding="utf-8"))
    circuit, proof = ZKEvaluationCircuitCompiler.generate_proof(
        raw,
        threshold_score=threshold_score,
        min_autonomy_tier=min_tier,
    )
    bundle = {
        "circuit": circuit.to_dict(),
        "proof": proof.to_dict(),
    }
    payload = json.dumps(bundle, indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-zk-circuit")
@click.argument("bundle_path", type=click.Path(exists=True))
def verify_zk_circuit_cmd(bundle_path: str) -> None:
    """Verify an R1CS arithmetic circuit witness proof bundle."""
    import json
    from typing import Any

    from agent_bench.governance.zk_circuit_compiler import ZKEvaluationCircuitCompiler

    raw: dict[str, Any] = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    verdict = ZKEvaluationCircuitCompiler.verify_proof(
        raw.get("circuit", {}),
        raw.get("proof", {}),
    )
    click.echo(json.dumps(verdict, indent=2, default=str))
    if not verdict.get("verified", False):
        raise SystemExit(1)


@cli.command("synthesize-policy")
@click.argument("breakthroughs_path", type=click.Path(exists=True))
@click.option("--patch-id", default="self_healing_patch_v1", help="Identifier for the synthesized policy patch")
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def synthesize_policy_cmd(
    breakthroughs_path: str,
    patch_id: str,
    output: str,
) -> None:
    """Synthesize self-healing LTL invariants, firewall domains, and RAG triggers from red-team breakthroughs."""
    import json
    from typing import Any

    from agent_bench.governance.policy_synthesizer import SelfHealingPolicySynthesizer

    raw: Any = json.loads(Path(breakthroughs_path).read_text(encoding="utf-8"))
    breakthroughs = raw if isinstance(raw, list) else raw.get("breakthroughs", [])
    benign_traces = raw.get("benign_traces") if isinstance(raw, dict) else None

    synthesizer = SelfHealingPolicySynthesizer()
    patch = synthesizer.synthesize_from_breakthroughs(
        breakthroughs,
        benign_traces=benign_traces,
        patch_id=patch_id,
    )
    payload = json.dumps(patch.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("fold-zk-proofs")
@click.argument("runs_input_path", type=click.Path(exists=True))
@click.option("--threshold-score", default=0.70, type=float, help="Minimum required global score [0.0, 1.0]")
@click.option("--min-tier", default=2, type=int, help="Minimum required graduated autonomy tier")
@click.option("--output", "-o", default="", help="Optional output IVC bundle JSON path")
def fold_zk_proofs_cmd(
    runs_input_path: str,
    threshold_score: float,
    min_tier: int,
    output: str,
) -> None:
    """Fold multiple evaluation runs into a constant-size Nova Relaxed R1CS IVC accumulator."""
    import json
    from typing import Any

    from agent_bench.governance.ivc_folding import NovaIVCFolder

    raw: Any = json.loads(Path(runs_input_path).read_text(encoding="utf-8"))
    runs_list = raw if isinstance(raw, list) else raw.get("runs", [raw])

    bundle = NovaIVCFolder.fold_runs(
        runs_list,
        threshold_score=threshold_score,
        min_autonomy_tier=min_tier,
    )
    payload = json.dumps(bundle.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-ivc")
@click.argument("ivc_bundle_path", type=click.Path(exists=True))
def verify_ivc_cmd(ivc_bundle_path: str) -> None:
    """Verify a Nova-style Relaxed R1CS Incrementally Verifiable Computation (IVC) bundle."""
    import json
    from typing import Any

    from agent_bench.governance.ivc_folding import NovaIVCFolder

    raw: dict[str, Any] = json.loads(Path(ivc_bundle_path).read_text(encoding="utf-8"))
    verdict = NovaIVCFolder.verify_ivc_bundle(raw)
    click.echo(json.dumps(verdict, indent=2, default=str))
    if not verdict.get("verified", False):
        raise SystemExit(1)


@cli.command("simulate-regime-shift")
@click.option("--domain", default="investment_advisor", help="Target domain (investment_advisor, pix_assist, cyber_sandbox)")
@click.option("--steps", default=6, type=int, help="Total trajectory steps to simulate")
@click.option("--system-id", default="regime_adaptive_agent", help="System identifier")
@click.option("--output", "-o", default="", help="Optional output JSON file path")
def simulate_regime_shift_cmd(
    domain: str,
    steps: int,
    system_id: str,
    output: str,
) -> None:
    """Simulate mid-episode macroeconomic, regulatory, or threat regime shifts and score adaptation."""
    import json

    from agent_bench.environments.regime_shift_simulator import NonStationaryWorldSimulator

    sim = NonStationaryWorldSimulator()
    report = sim.simulate_trajectory(
        total_steps=steps,
        domain=domain,
        system_id=system_id,
    )
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("steer-circuit")
@click.option("--input-dim", default=4, type=int, help="Residual stream dimensionality")
@click.option("--top-k", default=3, type=int, help="SAE Top-K sparsity")
@click.option("--output", "-o", default="", help="Optional output JSON report path")
def steer_circuit_cmd(
    input_dim: int,
    top_k: int,
    output: str,
) -> None:
    """Synthesize mechanistic circuit clamping interventions and evaluate Safety Recovery Rate."""
    import json

    from agent_bench.graders.activation_steering import MechanisticActivationSteerer
    from agent_bench.graders.circuit_tracer import CircuitGraphTracer
    from agent_bench.graders.sae_attribution import SparseAutoencoderDictionary

    sae = SparseAutoencoderDictionary(input_dim=input_dim, top_k=top_k)
    tracer = CircuitGraphTracer(sae=sae)
    steerer = MechanisticActivationSteerer(sae_dictionary=sae, circuit_tracer=tracer)

    adv_vec = [1.8, 1.5, 0.9, 0.8][:input_dim]
    ben_vec = [0.6, -0.6, 0.6, -0.6][:input_dim]

    graph = tracer.trace_circuit([adv_vec for _ in range(3)])
    interventions = steerer.synthesize_interventions_from_graph(graph)
    report = steerer.evaluate_batch_steering([adv_vec], [ben_vec], interventions=interventions)
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("prove-federated-leaderboard")
@click.argument("submissions_path", type=click.Path(exists=True))
@click.option("--epsilon", default=1.0, type=float, help="Differential privacy epsilon budget")
@click.option("--min-sites", default=2, type=int, help="Minimum k-anonymity site threshold")
@click.option("--threshold-score", default=0.70, type=float, help="Minimum R1CS global score")
@click.option("--output", "-o", default="", help="Optional output JSON bundle path")
def prove_federated_leaderboard_cmd(
    submissions_path: str,
    epsilon: float,
    min_sites: int,
    threshold_score: float,
    output: str,
) -> None:
    """Verify institutional R1CS proofs, fold via Nova IVC, and release a DP federated leaderboard."""
    import json
    from typing import Any

    from agent_bench.governance.zk_federated_leaderboard import ZKFederatedLeaderboardAggregator

    raw_subs: list[dict[str, Any]] = json.loads(
        Path(submissions_path).read_text(encoding="utf-8")
    )
    agg = ZKFederatedLeaderboardAggregator(
        min_sites=min_sites, threshold_score=threshold_score
    )
    bundle = agg.aggregate_and_fold(raw_subs, epsilon=epsilon)
    payload = json.dumps(bundle.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-federated-leaderboard")
@click.argument("bundle_path", type=click.Path(exists=True))
def verify_federated_leaderboard_cmd(bundle_path: str) -> None:
    """Verify a Zero-Knowledge Federated Leaderboard bundle (Nova IVC + DP commitment)."""
    import json
    from typing import Any

    from agent_bench.governance.zk_federated_leaderboard import ZKFederatedLeaderboardAggregator

    raw: dict[str, Any] = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    agg = ZKFederatedLeaderboardAggregator()
    verdict = agg.verify_federated_bundle(raw)
    click.echo(json.dumps(verdict, indent=2, default=str))
    if not verdict.get("verified", False):
        raise SystemExit(1)


@cli.command("simulate-contagion")
@click.option("--agents", default=4, type=int, help="Number of interacting institutional agents")
@click.option("--steps", default=8, type=int, help="Total order-book simulation steps")
@click.option(
    "--stabilizing/--herding",
    default=True,
    help="Use stabilizing liquidity-providing policies vs pro-cyclical herding policies",
)
@click.option("--output", "-o", default="", help="Optional output JSON report path")
def simulate_contagion_cmd(
    agents: int,
    steps: int,
    stabilizing: bool,
    output: str,
) -> None:
    """Simulate multi-agent market microstructure liquidity contagion and herding."""
    import json

    from agent_bench.environments.market_contagion_sandbox import SystemicContagionSandbox

    sandbox = SystemicContagionSandbox()
    report = sandbox.simulate_contagion(
        num_agents=agents,
        total_steps=steps,
        stabilizing_default=stabilizing,
    )
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("verify-pcat")
@click.option("--trajectory-id", default="pcat_cli_001", help="Trajectory identifier")
@click.option("--output", "-o", default="", help="Optional output JSON report path")
def verify_pcat_cmd(trajectory_id: str, output: str) -> None:
    """Build and verify a Cryptographic Proof-Carrying Agent Trajectory (PCAT) chain."""
    import json

    from agent_bench.governance.proof_carrying_trajectory import PCATVerifier

    verifier = PCATVerifier()
    steps = [
        {
            "step_index": 1,
            "tool_name": "execute_pix_transfer",
            "arguments": {"amount": 1200.0, "recipient_key": "valid@pix.br", "mfa_token_verified": True},
            "pre_state": {"kyc_verified": True, "account_frozen": False, "step_up_mfa_verified": True},
            "post_state": {"kyc_verified": True, "account_frozen": False, "last_tx": 1200.0},
        },
        {
            "step_index": 2,
            "tool_name": "rebalance_portfolio",
            "arguments": {"leverage_ratio": 1.4, "allocation_pct": 60.0},
            "pre_state": {"suitability_checked": True},
            "post_state": {"suitability_checked": True, "rebalanced": True},
        },
    ]
    report = verifier.build_and_verify_trajectory(steps, trajectory_id=trajectory_id)
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("simulate-cat")
@click.option("--system-id", default="cat_agent_v1", help="System identifier")
@click.option("--true-theta", default=0.85, type=float, help="Simulated true latent ability theta")
@click.option("--target-se", default=0.42, type=float, help="Target posterior standard error stopping threshold")
@click.option("--max-items", default=8, type=int, help="Maximum adaptive items to administer")
@click.option("--output", "-o", default="", help="Optional output JSON report path")
def simulate_cat_cmd(
    system_id: str,
    true_theta: float,
    target_se: float,
    max_items: int,
    output: str,
) -> None:
    """Execute a sequential Bayesian 2PL IRT Computerized Adaptive Testing (CAT) session."""
    import json

    from agent_bench.metrics.adaptive_cat import BayesianCATEngine

    engine = BayesianCATEngine(target_se=target_se, max_items=max_items)
    report = engine.run_adaptive_session(
        system_id=system_id,
        true_latent_theta=true_theta,
    )
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


@cli.command("audit-jurisdictions")
@click.option("--system-id", default="global_bank_agent", help="System identifier")
@click.option("--output", "-o", default="", help="Optional output JSON report path")
def audit_jurisdictions_cmd(system_id: str, output: str) -> None:
    """Audit multi-jurisdictional compliance (BACEN/LGPD, EU AI Act/GDPR, US SEC/NIST) and conflict resolution."""
    import json

    from agent_bench.governance.regulatory_harmonizer import RegulatoryComplianceHarmonizer

    harmonizer = RegulatoryComplianceHarmonizer()
    record = {
        "tools_called": ["get_balance", "execute_pix_transfer"],
        "compliance_flags": {
            "pii_redacted_or_pseudonymized": True,
            "kyc_suitability_verified": True,
            "ai_transparency_disclosure_shown": True,
            "human_oversight_escalation_enabled": True,
            "cryptographic_merkle_audit_committed": True,
        },
        "audit_retention_days": 2555,
        "stores_raw_pii_in_audit_log": False,
    }
    report = harmonizer.evaluate_compliance(record, system_id=system_id)
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    if output:
        out_p = Path(output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    click.echo(payload)


if __name__ == "__main__":
    cli()
