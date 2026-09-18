# Showdown

> Minimalist Human-in-the-Loop & LLM Output Ranking Arena.

Showdown is a lightweight, keyboard-driven pairwise ranking and evaluation server for comparing outputs from LLMs, agents, or generative models.

It provides an active Elo matchmaking engine, rapid triage rating, real-time leaderboards, and direct export to standardized DPO (Direct Preference Optimization) training datasets.

---

## Features

- **Multi-Modal Evaluations**: Compares prose, markdown, code, JSON, and images.
- **Active Elo Engine**: Matchmaker actively pairs candidates with similar ratings and fewest evaluations for rapid convergence.
- **Zero-Friction Keyboard UI**: Hotkeys for instant voting (`1` for A, `2` for B, `T` for Tie, `S` to Skip).
- **Agent Integration**: Simple Python SDK and REST API so any autonomous agent can launch a tournament, register candidates, and notify the user.
- **DPO Dataset Export**: Exports pairwise preferences directly to `{prompt, chosen, rejected}` JSONL for model alignment and fine-tuning.

---

## Quickstart

### 1. Installation

Using `uv` (recommended):

```bash
cd showdown
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

### 2. Install as Agent Plugin / Skill

Showdown conforms to the **[Agent Plugins Specification 1.0.0](https://github.com/agentplugins/agent-plugins-spec)** (`plugin.json`) and the universal Agent Skills standard.

**Via Universal Skills Manager (`skills.sh`):**
```bash
npx skills add SgtPooki/showdown
```

**Via Showdown CLI (auto-detects Antigravity, Claude Code, and Cursor):**
```bash
uv run showdown install-skill
```

### 3. Launch the Demo Tournament

```bash
uv run showdown demo
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Agent Usage (Python SDK)

Any autonomous agent or benchmark script can launch a tournament programmatically:

```python
from showdown import create_tournament

tournament = create_tournament(
    title="SQL Generation Comparison",
    prompt="Generate an optimized PostgreSQL query to find the top 10 users by 30-day spend.",
    task_type="code",
    candidates=[
        {"id": "qwen_2.5", "label": "Qwen 2.5 32B", "content": "SELECT ..."},
        {"id": "claude_sonnet", "label": "Claude 3.7 Sonnet", "content": "SELECT ..."},
        {"id": "deepseek_r1", "label": "DeepSeek R1", "content": "SELECT ..."}
    ]
)

print(f"Tournament ready for judging at: http://localhost:8091/?t={tournament.id}")
```

---

## CLI Reference

### Start Server
```bash
showdown serve --port 8091
```

### Create Tournament
```bash
showdown create --title "Summarization Test" --type text --file candidates.json --prompt "Summarize the earnings report."
```

Candidate file format (`candidates.json`):
```json
[
  { "id": "model_a", "label": "Model A", "content": "Summary text..." },
  { "id": "model_b", "label": "Model B", "content": "Alternative summary..." }
]
```

### List Tournaments
```bash
showdown list
```

### Export Preferences (for DPO)
```bash
showdown export <tournament-id> --format dpo --output dpo_dataset.json
```

---

## License

MIT License. Built and maintained by SgtPooki LLC.
