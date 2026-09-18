---
name: showdown
description: Run interactive pairwise human-in-the-loop ranking and preference convergence on candidate outputs (text, code, markdown, images). Use to rank LLM responses, evaluate taglines or design concepts, capture user likes/dislikes, and iteratively evolve generation N+1 toward user preferences.
allowed-tools: Bash, Read, Write
metadata:
  short-description: Launch Showdown arena to rank, judge, and evolve outputs
---

# Showdown Skill

Use Showdown to present pairwise comparison tournaments to humans or evaluator models, record Elo ratings, gather qualitative triage notes (dislikes, likes, critiques), and actively synthesize next-generation candidate pools.

## Quick Start Commands

### 1. Launch a Demo Tournament
```bash
uv run showdown demo --port 8000
```
Open `http://localhost:8000` to judge candidates.

### 2. Create a Tournament via CLI
```bash
uv run showdown create \
  --title "Tagline Evaluation" \
  --prompt "Choose the punchiest 3-word tagline" \
  --task-type text \
  --item "CODE. SYSTEMS. AGENTS." \
  --item "SYSTEMS. AGENTS. OPS." \
  --item "CODE. INFRA. INTELLIGENCE."
```

### 3. Start the Showdown Server
```bash
uv run showdown serve --port 8000
```

### 4. Create a Tournament & Wait for Winner (Python SDK)
```python
from showdown.client import create_tournament, wait_for_tournament
from showdown.models import TaskType

t = create_tournament(
    title="Model Prompt Variants",
    prompt="Generate an executive summary",
    task_type=TaskType.MARKDOWN,
    items=[
        {"label": "Model A (Claude)", "content": "..."},
        {"label": "Model B (Codex)", "content": "..."},
    ]
)
print(f"Created tournament: {t.id}")

# Autonomous agent blocks until the human accepts a winning candidate:
res = wait_for_tournament(t.id, timeout=120)
if res["completed"]:
    winner = res["accepted_candidate"]
    print(f"Winning output selected: {winner['label']}")
```

### 5. Export Ranked Dataset (DPO / KTO / JSONL)
```bash
# DPO pairs JSONL
uv run showdown export <tournament_id> --format dpo --jsonl --output ./dpo_pairs.jsonl

# KTO binary preference JSONL
uv run showdown export <tournament_id> --format kto --jsonl --output ./kto_pairs.jsonl
```

## Preference Evolution Loop

Showdown supports active preference convergence:
1. User reviews pairs in the Web UI (`1` for Candidate A, `2` for Candidate B, `T` for tie).
2. User provides qualitative feedback:
   - Notes input field: e.g. "dislike ops", "punchy 3 words", "avoid corporate jargon".
   - Star (favorite) or cross (dislike) on individual candidate cards.
3. User or Agent triggers `POST /api/tournaments/{tournament_id}/evolve`:
   - Summarizes winning Elo performers and user critiques.
   - Prompts the configured backend (Claude CLI, Codex CLI, local vLLM, or OpenAI API) to generate Generation N+1.
   - Automatically injects the new candidates into the tournament matchmaking queue.
