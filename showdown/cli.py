"""Command-line interface for Showdown."""

import json
from pathlib import Path
from typing import Optional
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
    port: int = typer.Option(8091, "--port", "-p", help="Port number"),
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
    task_type: TaskType = typer.Option(TaskType.TEXT, "--type", help="Task type (text, markdown, code, image, json)"),
    file: Path = typer.Option(..., "--file", "-f", help="JSON file containing list of candidates"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Evaluation prompt or task instructions"),
    tournament_id: Optional[str] = typer.Option(None, "--id", help="Custom tournament ID"),
):
    """Create a new tournament from a JSON file of candidates."""
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
    console.print(f"Start ranking at: [underline blue]http://localhost:8091/?t={t.id}[/underline blue]")


@app.command()
def export(
    tournament_id: str = typer.Argument(..., help="Tournament ID"),
    format: str = typer.Option("dpo", "--format", help="Export format: 'dpo', 'leaderboard', or 'raw'"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path to save the export"),
):
    """Export tournament preferences for DPO training or leaderboard stats."""
    storage = Storage()
    t = storage.load_tournament(tournament_id)
    if not t:
        console.print(f"[red]Error: Tournament '{tournament_id}' not found.[/red]")
        raise typer.Exit(code=1)

    from showdown.server import export_tournament

    data = export_tournament(tournament_id=tournament_id, format=format)
    formatted_json = json.dumps(data, indent=2)

    if output:
        output.write_text(formatted_json)
        console.print(f"[green]Exported {format} data to[/green] {output}")
    else:
        console.print(formatted_json)


@app.command()
def demo(port: int = typer.Option(8091, "--port", "-p", help="Port to serve")):
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


if __name__ == "__main__":
    app()
