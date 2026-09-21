"""Command-line interface for Showdown."""

import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
import uvicorn

from showdown.models import Candidate, CreateTournamentRequest, TaskType
from showdown.storage import Storage

app = typer.Typer(
    name="showdown",
    help="Showdown: Minimalist Human & LLM Output Ranking Arena",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the Showdown web server."""
    console.print(f"[bold green]Starting Showdown Arena at[/bold green] http://{host}:{port}")
    uvicorn.run("showdown.server:app", host=host, port=port, reload=reload)


@app.command()
def list_tournaments():
    """List all tournaments and candidate counts."""
    storage = Storage()
    tournaments = storage.list_tournaments()
    if not tournaments:
        console.print("[yellow]No tournaments found. Create one with `showdown create` or `showdown demo`.[/yellow]")
        return

    table = Table(title="Showdown Tournaments")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("Candidates", justify="right", style="green")
    table.add_column("Matches", justify="right", style="blue")

    for t in tournaments:
        table.add_row(t.id, t.title, t.task_type.value, str(len(t.candidates)), str(len(t.matches)))

    console.print(table)


@app.command()
def create(
    title: str = typer.Option(..., "--title", "-t", help="Tournament title"),
    task_type: TaskType = typer.Option(TaskType.TEXT, "--type", help="Task type (text, markdown, code, diff, image, json)"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="JSON file containing list of candidates"),
    item: Optional[List[str]] = typer.Option(None, "--item", "-i", help="Candidate item content (can be specified multiple times)"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Evaluation prompt or task instructions"),
    tournament_id: Optional[str] = typer.Option(None, "--id", help="Custom tournament ID"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port for URL preview"),
):
    """Create a new tournament from a JSON file or direct items."""
    candidates = []

    if item:
        candidates = [
            Candidate(
                id=f"item_{i+1}",
                label=f"Candidate #{i+1}",
                content=str(val),
            )
            for i, val in enumerate(item)
        ]
    elif file:
        if not file.exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(code=1)

        try:
            raw_data = json.loads(file.read_text())
            if isinstance(raw_data, dict) and "candidates" in raw_data:
                raw_candidates = raw_data["candidates"]
                if not prompt and "prompt" in raw_data:
                    prompt = raw_data["prompt"]
            elif isinstance(raw_data, list):
                raw_candidates = raw_data
            else:
                raise ValueError("Expected a JSON array of candidates or a dict with a 'candidates' key.")
        except Exception as e:
            console.print(f"[red]Error parsing candidate JSON: {e}[/red]")
            raise typer.Exit(code=1)

        candidates = [
            Candidate(
                id=str(c.get("id", f"cand_{i}")),
                label=c.get("label", c.get("id")),
                content=str(c.get("content", "")),
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(raw_candidates)
        ]
    else:
        console.print("[red]Error: Must provide either --file or one or more --item options.[/red]")
        raise typer.Exit(code=1)

    from showdown.server import create_tournament

    req = CreateTournamentRequest(
        id=tournament_id,
        title=title,
        prompt=prompt,
        task_type=task_type,
        candidates=candidates,
    )
    t = create_tournament(req)

    console.print(f"[bold green]Tournament created:[/bold green] [cyan]{t.id}[/cyan]")
    console.print(f"Candidates: [white]{len(t.candidates)}[/white]")
    console.print(f"Start ranking at: [underline blue]http://localhost:{port}/?t={t.id}[/underline blue]")


@app.command()
def export(
    tournament_id: Optional[str] = typer.Argument(None, help="Tournament ID (optional if --all is set)"),
    all_tournaments: bool = typer.Option(False, "--all", "-a", help="Export preferences across all tournaments"),
    format: str = typer.Option("dpo", "--format", help="Export format: 'dpo', 'kto', 'pairwise_margins', 'leaderboard', or 'raw'"),
    task_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by task type: 'code', 'text', 'svg', 'markdown'"),
    voter: Optional[str] = typer.Option(None, "--voter", "-v", help="Filter export to matches by specific evaluator"),
    consensus: Optional[str] = typer.Option(None, "--consensus", help="Consensus mode: 'strict' (unanimous >=2 voters) or 'majority'"),
    min_agreement: Optional[float] = typer.Option(None, "--min-agreement", help="Minimum agreement threshold (0.5 to 1.0)"),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Deduplicate identical preference pairs"),
    critique: bool = typer.Option(True, "--critique/--no-critique", help="Include evaluator notes and critiques"),
    split: Optional[float] = typer.Option(None, "--split", help="Train/val split ratio (e.g. 0.8 for 80% train / 20% val)"),
    split_by: str = typer.Option("lineage", "--split-by", help="Split partitioning strategy: 'lineage' or 'random'"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Directory to save split files: train.jsonl and val.jsonl"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Export as JSON Lines format"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path to save the export"),
):
    """Export tournament preferences for DPO/KTO/reward modeling with automated train/val splitting."""
    storage = Storage()

    if all_tournaments or not tournament_id:
        tournaments = storage.list_tournaments()
        if not tournaments:
            console.print("[yellow]No tournaments found in storage to export.[/yellow]")
            raise typer.Exit(code=0)
    else:
        t = storage.load_tournament(tournament_id)
        if not t:
            console.print(f"[red]Error: Tournament '{tournament_id}' not found.[/red]")
            raise typer.Exit(code=1)
        tournaments = [t]

    from showdown.export import export_dataset, format_jsonl

    if format == "leaderboard":
        if len(tournaments) == 1:
            t = tournaments[0]
            sorted_cands = sorted(t.candidates, key=lambda c: t.stats.get(c.id, CandidateStats()).elo, reverse=True)
            data = [
                {
                    "rank": i + 1,
                    "id": c.id,
                    "label": c.label,
                    "elo": t.stats.get(c.id, CandidateStats()).elo,
                    "matches": t.stats.get(c.id, CandidateStats()).matches,
                    "wins": t.stats.get(c.id, CandidateStats()).wins,
                    "losses": t.stats.get(c.id, CandidateStats()).losses,
                    "ties": t.stats.get(c.id, CandidateStats()).ties,
                }
                for i, c in enumerate(sorted_cands)
            ]
        else:
            console.print("[red]Error: 'leaderboard' format is only supported for single-tournament export.[/red]")
            raise typer.Exit(code=1)
    else:
        try:
            data = export_dataset(
                tournaments=tournaments,
                format=format,
                task_type=task_type,
                voter=voter,
                consensus=consensus,
                min_agreement=min_agreement,
                dedup=dedup,
                include_critique=critique,
                split=split,
                split_by=split_by,
            )
        except ValueError as e:
            console.print(f"[red]Export error:[/red] {e}")
            raise typer.Exit(code=1)

    # Handle directory output for splits
    if out_dir and isinstance(data, dict) and "train" in data and "val" in data:
        out_dir.mkdir(parents=True, exist_ok=True)
        train_file = out_dir / "train.jsonl"
        val_file = out_dir / "val.jsonl"
        info_file = out_dir / "dataset_info.json"

        train_file.write_text(format_jsonl(data["train"]))
        val_file.write_text(format_jsonl(data["val"]))
        info_file.write_text(json.dumps(data["split_stats"], indent=2))

        console.print(f"[bold green]Dataset split exported to {out_dir}:[/bold green]")
        console.print(f"  • Train: [cyan]{len(data['train'])}[/cyan] records -> {train_file.name}")
        console.print(f"  • Validation: [cyan]{len(data['val'])}[/cyan] records -> {val_file.name}")
        console.print(f"  • Split ratio: [white]{data['split_stats'].get('train_ratio', 0.0) * 100:.1f}% train[/white] (by {split_by})")
        return

    if jsonl:
        if isinstance(data, list):
            formatted_output = format_jsonl(data)
        elif isinstance(data, dict) and "train" in data and "val" in data:
            tagged_records = [{"split": "train", **r} for r in data["train"]] + [{"split": "val", **r} for r in data["val"]]
            formatted_output = format_jsonl(tagged_records)
        else:
            formatted_output = json.dumps(data, indent=2)
    else:
        formatted_output = json.dumps(data, indent=2)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(formatted_output)
        rec_count = len(data) if isinstance(data, list) else (len(data.get("train", [])) + len(data.get("val", [])))
        console.print(f"[green]Exported {rec_count} {format} records across {len(tournaments)} tournament(s) to[/green] {output}")
    else:
        console.print(formatted_output)


@app.command()
def demo(port: int = typer.Option(8000, "--port", "-p", help="Port to serve")):
    """Initialize a demo tournament and launch the ranking server."""
    storage = Storage()
    demo_id = "demo_code_refactor"
    if not storage.load_tournament(demo_id):
        demo_candidates = [
            {
                "id": "model_alpha",
                "label": "Claude 3.7 Sonnet",
                "content": "```python\ndef get_user_session(user_id: str) -> Optional[Session]:\n    # High efficiency session lookup with local TTL cache\n    if cached := cache.get(user_id):\n        return cached\n    session = db.query(Session).filter_by(user_id=user_id).first()\n    if session:\n        cache.set(user_id, session, ttl=300)\n    return session\n```",
                "metadata": {"model": "claude-3-7-sonnet"},
            },
            {
                "id": "model_beta",
                "label": "Qwen 2.5 32B (vLLM)",
                "content": "```python\ndef get_user_session(user_id: str):\n    session = cache.get(user_id)\n    if not session:\n        session = db.query(Session).filter(Session.user_id == user_id).first()\n        cache.set(user_id, session, 300)\n    return session\n```",
                "metadata": {"model": "qwen2.5-32b-instruct"},
            },
            {
                "id": "model_gamma",
                "label": "DeepSeek R1",
                "content": "```python\ndef get_user_session(user_id: str) -> Session | None:\n    \"\"\"Retrieve session with read-through Redis cache pattern.\"\"\"\n    val = redis_client.get(f'session:{user_id}')\n    if val is not None:\n        return Session.model_validate_json(val)\n    sess = db.session.get(Session, user_id)\n    if sess is not None:\n        redis_client.setex(f'session:{user_id}', 300, sess.model_dump_json())\n    return sess\n```",
                "metadata": {"model": "deepseek-r1"},
            },
        ]
        from showdown.client import create_tournament

        create_tournament(
            title="Python Session Cache Implementation",
            prompt="Write a concise, idiomatic Python helper function to fetch a user session with a 5-minute read-through cache.",
            task_type="code",
            tournament_id=demo_id,
            candidates=demo_candidates,
        )
        console.print("[green]Created demo tournament 'demo_code_refactor'[/green]")

    console.print(f"[bold cyan]Opening Showdown Demo Arena at[/bold cyan] http://127.0.0.1:{port}/?t={demo_id}")
    uvicorn.run("showdown.server:app", host="127.0.0.1", port=port)


@app.command()
def install_skill(
    target: str = typer.Option("auto", "--target", "-t", help="Target framework: 'auto', 'agents', 'claude', 'global-claude', 'global-agents'"),
    dest: Optional[Path] = typer.Option(None, "--dest", "-d", help="Custom destination directory"),
    symlink: bool = typer.Option(False, "--symlink", help="Symlink skill instead of copying"),
):
    """Install the Showdown agent skill into an agent workspace or config."""
    import shutil

    skill_src = Path(__file__).resolve().parent.parent / "skills" / "showdown"
    if not skill_src.exists():
        console.print(f"[red]Error: Skill source directory not found at {skill_src}[/red]")
        raise typer.Exit(code=1)

    targets = []
    if dest:
        targets.append(dest)
    elif target == "agents":
        targets.append(Path.cwd() / ".agents" / "skills" / "showdown")
    elif target == "claude":
        targets.append(Path.cwd() / ".claude" / "skills" / "showdown")
    elif target == "global-claude":
        targets.append(Path.home() / ".claude" / "skills" / "showdown")
    elif target == "global-agents":
        targets.append(Path.home() / ".agents" / "skills" / "showdown")
    elif target == "auto":
        found_any = False
        curr = Path.cwd().resolve()
        for parent in [curr] + list(curr.parents):
            if (parent / ".agents" / "skills").is_dir():
                targets.append(parent / ".agents" / "skills" / "showdown")
                found_any = True
            if (parent / ".claude" / "skills").is_dir():
                targets.append(parent / ".claude" / "skills" / "showdown")
                found_any = True
            if found_any:
                break
        if not found_any:
            targets.append(curr / ".agents" / "skills" / "showdown")

    for target_dir in targets:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.is_symlink() or target_dir.exists():
            if target_dir.is_symlink() or target_dir.is_file():
                target_dir.unlink()
            elif target_dir.is_dir():
                shutil.rmtree(target_dir)

        if symlink:
            try:
                target_dir.symlink_to(skill_src.resolve(), target_is_directory=True)
                console.print(f"[green]Symlinked skill to[/green] {target_dir} -> {skill_src.resolve()}")
            except OSError:
                shutil.copytree(skill_src, target_dir)
                console.print(f"[green]Copied skill to[/green] {target_dir}")
        else:
            shutil.copytree(skill_src, target_dir)
            console.print(f"[green]Copied skill to[/green] {target_dir}")

    console.print("[bold cyan]Showdown skill ready for agent invocation via /showdown![/bold cyan]")


@app.command()
def judge(
    tournament_id: str = typer.Argument(..., help="Tournament ID to judge"),
    backend: str = typer.Option("auto", "--backend", "-b", help="LLM backend: 'auto', 'omp', 'claude', 'codex', 'openai'"),
    rounds: int = typer.Option(5, "--rounds", "-r", help="Number of match rounds to evaluate"),
    rubric: Optional[Path] = typer.Option(None, "--rubric", help="Optional file path or custom rubric criteria"),
    swap: bool = typer.Option(True, "--swap/--no-swap", help="Mitigate position bias by evaluating swapped pairs"),
    voter: Optional[str] = typer.Option(None, "--voter", "-v", help="Custom voter identifier (defaults to judge:<backend>)"),
    mode: str = typer.Option("active", "--mode", "-m", help="Matchup selection strategy: 'active', 'info_gain', 'controversial', or 'close'"),
    stop_on_convergence: bool = typer.Option(False, "--stop-on-convergence", help="Stop evaluation early if tournament converges"),
):
    """Run automated LLM-as-a-judge tournament rounds with position-bias mitigation."""
    storage = Storage()
    t = storage.load_tournament(tournament_id)
    if not t:
        console.print(f"[red]Error: Tournament '{tournament_id}' not found.[/red]")
        raise typer.Exit(code=1)

    rubric_text = None
    if rubric:
        if rubric.exists():
            rubric_text = rubric.read_text().strip()
        else:
            rubric_text = str(rubric)

    from showdown.client import run_judge

    console.print(f"[bold cyan]Starting LLM-as-a-judge runner on '{tournament_id}'[/bold cyan] (backend={backend}, rounds={rounds}, swap={swap}, mode={mode})")
    res = run_judge(
        tournament_id=tournament_id,
        rounds=rounds,
        backend=backend,
        rubric=rubric_text,
        swap_positions=swap,
        voter=voter,
        mode=mode,
        stop_on_convergence=stop_on_convergence,
    )

    console.print(f"[bold green]Evaluated {res['matches_evaluated']} matches[/bold green] (consistent={res['consistent_matches']}, contradictions={res['contradictions']})")
    if res["converged"]:
        console.print(f"[bold green]Tournament converged![/bold green] Confidence: {res['confidence'] * 100:.0f}%")
    else:
        console.print(f"Convergence confidence: {res['confidence'] * 100:.0f}%")

    table = Table(title=f"Judge Results ({res['voter']})")
    table.add_column("Candidate A", style="cyan")
    table.add_column("Candidate B", style="magenta")
    table.add_column("Winner", style="bold yellow")
    table.add_column("Consistent", style="green")
    table.add_column("Critique", style="white", max_width=60)

    for r in res["results"]:
        consistent_label = "Yes" if r["swapped_consistent"] else "No (Bias/Abstain)"
        if not swap:
            consistent_label = "N/A"
        table.add_row(
            r["id_a"],
            r["id_b"],
            r["winner"].upper(),
            consistent_label,
            r["critique"] or "",
        )
    console.print(table)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport protocol ('stdio', 'sse')"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Custom storage directory"),
):
    """Run the Showdown Model Context Protocol (MCP) server for agent IDE and CLI integration."""
    from showdown.mcp_server import run_mcp_server
    run_mcp_server(transport=transport, data_dir=data_dir)


if __name__ == "__main__":
    app()

