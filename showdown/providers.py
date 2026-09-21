"""Extensible Agent Provider interface and registry with CLI/API auto-detection & multi-agent fan-out."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class AgentProvider(ABC):
    """Abstract base provider for LLM agent CLI runtimes and API endpoints."""

    id: str
    display_name: str
    provider_type: str = "cli"  # "cli" | "api" | "mock"
    model: Optional[str] = None

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider binary or API credentials/endpoint are accessible on host."""
        pass

    @abstractmethod
    def generate(self, prompt: str, timeout: int = 120) -> str:
        """Invoke agent provider with prompt and return raw text output."""
        pass

    def call(self, prompt: str, timeout: int = 120) -> str:
        """Alias for generate() for backwards compatibility."""
        return self.generate(prompt, timeout=timeout)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider_type": self.provider_type,
            "model": self.model,
            "available": self.is_available(),
        }


class ClaudeCliProvider(AgentProvider):
    id = "claude"
    display_name = "Anthropic Claude (CLI)"
    provider_type = "cli"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("SHOWDOWN_CLAUDE_MODEL", "claude-default")

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def generate(self, prompt: str, timeout: int = 120) -> str:
        if not self.is_available():
            raise RuntimeError("Claude CLI ('claude') not found in PATH")
        res = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {res.stderr.strip()}")
        return res.stdout.strip()


ClaudeProvider = ClaudeCliProvider


class CodexCliProvider(AgentProvider):
    id = "codex"
    display_name = "Codex CLI"
    provider_type = "cli"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("SHOWDOWN_CODEX_MODEL", "codex-default")

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    def generate(self, prompt: str, timeout: int = 120) -> str:
        if not self.is_available():
            raise RuntimeError("Codex CLI ('codex') not found in PATH")
        res = subprocess.run(
            ["codex", "exec", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Codex CLI failed: {res.stderr.strip()}")
        return res.stdout.strip()


CodexProvider = CodexCliProvider


class CursorProvider(AgentProvider):
    id = "cursor"
    display_name = "Cursor Agent (CLI)"
    provider_type = "cli"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("SHOWDOWN_CURSOR_MODEL", "cursor-default")

    def is_available(self) -> bool:
        return shutil.which("cursor-agent") is not None

    def generate(self, prompt: str, timeout: int = 120) -> str:
        if not self.is_available():
            raise RuntimeError("Cursor Agent CLI ('cursor-agent') not found in PATH")
        cmd = ["cursor-agent", "-p", "--output-format", "text", prompt]
        if self.model and self.model != "cursor-default":
            cmd.extend(["--model", self.model])
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Cursor Agent CLI failed: {res.stderr.strip()}")
        return res.stdout.strip()


CursorCliProvider = CursorProvider


class OmpProvider(AgentProvider):
    id = "omp"
    display_name = "OMP Homelab (CLI)"
    provider_type = "cli"

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("SHOWDOWN_HOMELAB_MODEL", "homelab-default")

    def is_available(self) -> bool:
        return shutil.which("omp") is not None

    def generate(self, prompt: str, timeout: int = 120) -> str:
        if not self.is_available():
            raise RuntimeError("OMP CLI ('omp') not found in PATH")
        model_name = self.model or "homelab-default"
        res = subprocess.run(
            ["omp", "-p", f"--model={model_name}", "--no-session", "--no-tools", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"OMP Homelab CLI failed: {res.stderr.strip()}")
        return res.stdout.strip()


OmpCliProvider = OmpProvider


class MockProvider(AgentProvider):
    id = "mock"
    display_name = "Mock Provider"
    provider_type = "mock"
    model = "mock-model"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, timeout: int = 120) -> str:
        prompt_lower = prompt.lower()
        if "json array" in prompt_lower or "generate exactly" in prompt_lower or "candidate variations" in prompt_lower:
            return json.dumps([
                {"label": "Mock Candidate", "content": "Mock generated candidate content.", "differs_by": "Mock variation"}
            ])
        if "winner" in prompt_lower or "judge" in prompt_lower or "candidate a:" in prompt_lower:
            return json.dumps({"winner": "a", "critique": "Mock judge preference."})
        return json.dumps([
            {"label": "Mock Candidate", "content": "Mock generated candidate content.", "differs_by": "Mock variation"}
        ])


class OpenAICompatibleProvider(AgentProvider):
    def __init__(
        self,
        provider_id: str = "openai",
        display_name: str = "OpenAI-Compatible API",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.id = provider_id
        self.display_name = display_name
        self.provider_type = "api"
        self._custom_base_url = base_url
        self._custom_api_key = api_key
        self.model = model or os.environ.get("SHOWDOWN_MODEL", "gpt-4o-mini")

    @property
    def base_url(self) -> str:
        if self._custom_base_url:
            return self._custom_base_url.rstrip("/")
        env_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("VLLM_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return env_url.rstrip("/")

    @property
    def api_key(self) -> str:
        if self._custom_api_key is not None:
            return self._custom_api_key
    def is_available(self) -> bool:
        if self.id == "vllm":
            return bool(os.environ.get("VLLM_BASE_URL") or os.environ.get("SHOWDOWN_VLLM_URL") or self._custom_base_url)
        if self.id == "ollama":
            if os.environ.get("OLLAMA_BASE_URL") or self._custom_base_url:
                return True
            # Probe localhost:11434 to check if local Ollama daemon is active
            try:
                with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.2) as resp:
                    return resp.status == 200
            except Exception:
                return False
        if self.id == "openai":
            return bool(self.api_key or os.environ.get("OPENAI_BASE_URL"))
        return bool(self._custom_base_url or self.api_key)

    def generate(self, prompt: str, timeout: int = 120) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise preference optimization assistant that outputs only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class ProviderRegistry:
    """Central registry for discovering, selecting, and invoking agent providers."""

    def __init__(self):
        self._providers: Dict[str, AgentProvider] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(ClaudeCliProvider())
        self.register(CodexCliProvider())
        self.register(CursorProvider())
        self.register(OmpProvider())

        # Generic OpenAI / OpenRouter
        self.register(OpenAICompatibleProvider(
            provider_id="openai",
            display_name="OpenAI API",
        ))

        # Local cluster / vLLM endpoint if configured
        vllm_url = os.environ.get("VLLM_BASE_URL") or os.environ.get("SHOWDOWN_VLLM_URL")
        self.register(OpenAICompatibleProvider(
            provider_id="vllm",
            display_name="vLLM Inference Server",
            base_url=vllm_url,
            api_key=os.environ.get("VLLM_API_KEY"),
            model=os.environ.get("VLLM_MODEL", "vllm-default"),
        ))

        # Local Ollama endpoint if configured or running
        ollama_url = os.environ.get("OLLAMA_BASE_URL")
        self.register(OpenAICompatibleProvider(
            provider_id="ollama",
            display_name="Ollama Local Runtime",
            base_url=ollama_url,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
        ))

        # Test / Mock provider
        self.register(MockProvider())

    def register(self, provider: AgentProvider):
        self._providers[provider.id] = provider

    def get(self, provider_id: str) -> Optional[AgentProvider]:
        clean_id = provider_id.lower().strip()
        # Aliases
        if clean_id in ("homelab", "homelab-default"):
            clean_id = "omp"
        elif clean_id in ("claude-code", "anthropic"):
            clean_id = "claude"
        elif clean_id in ("cursor-agent",):
            clean_id = "cursor"
        return self._providers.get(clean_id)

    def list_available(self, include_mock: bool = False) -> List[AgentProvider]:
        return [
            p for p in self._providers.values()
            if p.is_available() and (include_mock or p.provider_type != "mock")
        ]

    def list_all(self, include_mock: bool = False) -> List[AgentProvider]:
        return [
            p for p in self._providers.values()
            if include_mock or p.provider_type != "mock"
        ]

    def resolve_provider(self, requested: Optional[str] = None) -> AgentProvider:
        """
        Resolve a concrete provider from user preference or environment defaults,
        with graceful fallback following homelab priority.
        """
        if requested and requested not in ("auto", "default", None):
            p = self.get(requested)
            if p and p.is_available():
                return p
            elif p:
                raise RuntimeError(f"Provider '{requested}' is configured but not available on this host.")
            raise ValueError(f"Unknown provider '{requested}'. Available: {[p.id for p in self.list_available()]}")

        # Check global environment override
        env_pref = os.environ.get("SHOWDOWN_BACKEND")
        if env_pref and env_pref != "auto":
            p = self.get(env_pref)
            if p and p.is_available():
                return p

        # Auto-selection priority
        if os.environ.get("SHOWDOWN_PREFER_HOMELAB"):
            omp_p = self.get("omp")
            if omp_p and omp_p.is_available():
                return omp_p

        # Standard priority: claude -> omp -> codex -> cursor -> openai -> vllm -> ollama
        for pid in ("claude", "omp", "codex", "cursor", "openai", "vllm", "ollama"):
            p = self.get(pid)
            if p and p.is_available():
                return p

        available = self.list_available()
        if available:
            return available[0]

        raise RuntimeError("No available agent provider found on host. Ensure claude, omp, codex, cursor-agent, or OPENAI_API_KEY is available.")


# Global default registry instance
registry = ProviderRegistry()


def compute_provider_leaderboard(tournaments: List[Any]) -> List[Dict[str, Any]]:
    """
    Compute win/loss records, win rates, and Elo ratings aggregated by agent provider/model.
    Matches are counted when two candidates from distinct providers/backends competed.
    """
    provider_stats: Dict[str, Dict[str, Any]] = {}

    def get_cand_provider(cand: Any) -> Optional[str]:
        if not hasattr(cand, "metadata") or not cand.metadata:
            return None
        return cand.metadata.get("provider") or cand.metadata.get("backend")

    # Track candidates count and accepted winners
    for t in tournaments:
        cand_map = {c.id: c for c in t.candidates}
        for c in t.candidates:
            p = get_cand_provider(c)
            if p:
                p_clean = p.lower().strip()
                if p_clean in ("homelab", "homelab-default"):
                    p_clean = "omp"
                elif p_clean in ("claude-code", "anthropic"):
                    p_clean = "claude"
                elif p_clean in ("cursor-agent",):
                    p_clean = "cursor"

                if p_clean not in provider_stats:
                    reg_p = registry.get(p_clean)
                    disp_name = reg_p.display_name if reg_p else p_clean.capitalize()
                    provider_stats[p_clean] = {
                        "provider_id": p_clean,
                        "display_name": disp_name,
                        "elo": 1200.0,
                        "wins": 0,
                        "losses": 0,
                        "ties": 0,
                        "matches": 0,
                        "accepted_winners": 0,
                        "candidates_count": 0,
                    }
                provider_stats[p_clean]["candidates_count"] += 1

        if t.accepted_candidate_id and t.accepted_candidate_id in cand_map:
            accepted_cand = cand_map[t.accepted_candidate_id]
            p = get_cand_provider(accepted_cand)
            if p:
                p_clean = p.lower().strip()
                if p_clean in ("homelab", "homelab-default"):
                    p_clean = "omp"
                elif p_clean in ("claude-code", "anthropic"):
                    p_clean = "claude"
                elif p_clean in ("cursor-agent",):
                    p_clean = "cursor"
                if p_clean in provider_stats:
                    provider_stats[p_clean]["accepted_winners"] += 1

        # Process matches between candidates of distinct providers
        for m in t.matches:
            cand_a = cand_map.get(m.id_a)
            cand_b = cand_map.get(m.id_b)
            if not cand_a or not cand_b:
                continue

            prov_a = get_cand_provider(cand_a)
            prov_b = get_cand_provider(cand_b)
            if not prov_a or not prov_b:
                continue

            pa_clean = prov_a.lower().strip()
            if pa_clean in ("homelab", "homelab-default"):
                pa_clean = "omp"
            elif pa_clean in ("claude-code", "anthropic"):
                pa_clean = "claude"
            elif pa_clean in ("cursor-agent",):
                pa_clean = "cursor"

            pb_clean = prov_b.lower().strip()
            if pb_clean in ("homelab", "homelab-default"):
                pb_clean = "omp"
            elif pb_clean in ("claude-code", "anthropic"):
                pb_clean = "claude"
            elif pb_clean in ("cursor-agent",):
                pb_clean = "cursor"

            if pa_clean == pb_clean:
                # Intra-provider comparison, skip for provider Elo
                continue

            # Ensure both exist in stats
            for p_clean in (pa_clean, pb_clean):
                if p_clean not in provider_stats:
                    reg_p = registry.get(p_clean)
                    disp_name = reg_p.display_name if reg_p else p_clean.capitalize()
                    provider_stats[p_clean] = {
                        "provider_id": p_clean,
                        "display_name": disp_name,
                        "elo": 1200.0,
                        "wins": 0,
                        "losses": 0,
                        "ties": 0,
                        "matches": 0,
                        "accepted_winners": 0,
                        "candidates_count": 0,
                    }

            # Update match counts and Elo
            ra = provider_stats[pa_clean]["elo"]
            rb = provider_stats[pb_clean]["elo"]

            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            eb = 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0))

            k = 32.0  # standard K factor for cross-model leaderboard

            if m.winner == "a":
                sa, sb = 1.0, 0.0
                provider_stats[pa_clean]["wins"] += 1
                provider_stats[pb_clean]["losses"] += 1
            elif m.winner == "b":
                sa, sb = 0.0, 1.0
                provider_stats[pa_clean]["losses"] += 1
                provider_stats[pb_clean]["wins"] += 1
            elif m.winner == "tie":
                sa, sb = 0.5, 0.5
                provider_stats[pa_clean]["ties"] += 1
                provider_stats[pb_clean]["ties"] += 1
            else:
                continue

            provider_stats[pa_clean]["matches"] += 1
            provider_stats[pb_clean]["matches"] += 1
            provider_stats[pa_clean]["elo"] += k * (sa - ea)
            provider_stats[pb_clean]["elo"] += k * (sb - eb)

    # Compute win rates and format
    results = []
    for p, data in provider_stats.items():
        matches = data["matches"]
        win_rate = round((data["wins"] / matches) * 100.0, 1) if matches > 0 else 0.0
        results.append({
            "provider_id": data["provider_id"],
            "display_name": data["display_name"],
            "elo": round(data["elo"], 1),
            "wins": data["wins"],
            "losses": data["losses"],
            "ties": data["ties"],
            "matches": matches,
            "win_rate": win_rate,
            "accepted_winners": data["accepted_winners"],
            "candidates_count": data["candidates_count"],
        })

    results.sort(key=lambda r: (r["elo"], r["win_rate"], r["accepted_winners"]), reverse=True)
    return results

