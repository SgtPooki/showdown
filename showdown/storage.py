import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union
from showdown.models import Tournament, CandidateStats


def get_data_dir() -> Path:
    """Return the base storage directory for Showdown."""
    env_dir = os.environ.get("SHOWDOWN_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
    elif (Path.cwd() / ".showdown").is_dir():
        p = Path.cwd() / ".showdown"
    else:
        p = Path.home() / ".showdown"
    p.mkdir(parents=True, exist_ok=True)
    return p


class Storage:
    def __init__(self, data_dir: Optional[Union[Path, str]] = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.data_dir = get_data_dir()

    def _tournament_file(self, tournament_id: str) -> Path:
        clean_id = tournament_id.replace("/", "_").replace("\\", "_")
        return self.data_dir / f"{clean_id}.json"

    def _lock_file(self, tournament_id: str) -> Path:
        clean_id = tournament_id.replace("/", "_").replace("\\", "_")
        return self.data_dir / f"{clean_id}.lock"

    @contextmanager
    def lock_tournament(self, tournament_id: str):
        """Cross-process file lock for tournament mutations."""
        lock_path = self._lock_file(tournament_id)
        with open(lock_path, "a") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def has_tournament(self, tournament_id: str) -> bool:
        return self._tournament_file(tournament_id).exists()

    def save_tournament(self, tournament: Tournament) -> None:
        file_path = self._tournament_file(tournament.id)
        tmp_path = file_path.parent / f"{file_path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp_path.write_text(tournament.model_dump_json(indent=2))
        tmp_path.replace(file_path)

    def load_tournament(self, tournament_id: str) -> Optional[Tournament]:
        file_path = self._tournament_file(tournament_id)
        if not file_path.exists():
            return None
        try:
            data = json.loads(file_path.read_text())
            tournament = Tournament(**data)
            # Ensure every candidate has an entry in stats
            for c in tournament.candidates:
                if c.id not in tournament.stats:
                    tournament.stats[c.id] = CandidateStats()
            return tournament
        except Exception as e:
            print(f"Error loading tournament {tournament_id}: {e}")
            return None

    def list_tournaments(self) -> List[Tournament]:
        tournaments = []
        for file in sorted(self.data_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text())
                tournaments.append(Tournament(**data))
            except Exception:
                continue
        return tournaments

    def delete_tournament(self, tournament_id: str) -> bool:
        file_path = self._tournament_file(tournament_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
