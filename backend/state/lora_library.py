"""Persistent LoRA library — tracks downloaded LoRAs with metadata."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


@dataclass
class LoraEntry:
    """A single LoRA in the local library."""

    id: str
    name: str
    file_path: str
    file_size_bytes: int = 0
    thumbnail_url: str = ""
    trigger_phrase: str = ""
    base_model: str = ""
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None
    description: str = ""


@dataclass
class LoraLibrary:
    entries: list[LoraEntry] = field(default_factory=lambda: list[LoraEntry]())


class LoraLibraryStore:
    """Thread-safe, JSON-backed LoRA catalog."""

    def __init__(self, loras_dir: Path) -> None:
        self._loras_dir = loras_dir
        self._loras_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = loras_dir / "catalog.json"
        self._lock = threading.Lock()
        self._library = self._load()

    @property
    def loras_dir(self) -> Path:
        return self._loras_dir

    def _load(self) -> LoraLibrary:
        if not self._catalog_path.exists():
            return LoraLibrary()
        try:
            raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
            entries = [LoraEntry(**e) for e in raw.get("entries", [])]
            return LoraLibrary(entries=entries)
        except Exception:
            _logger.warning("Failed to load LoRA catalog, starting fresh", exc_info=True)
            return LoraLibrary()

    def _save(self) -> None:
        data = {"entries": [asdict(e) for e in self._library.entries]}
        self._catalog_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_all(self) -> list[LoraEntry]:
        with self._lock:
            return list(self._library.entries)

    def get(self, lora_id: str) -> LoraEntry | None:
        with self._lock:
            for entry in self._library.entries:
                if entry.id == lora_id:
                    return entry
            return None

    def add(self, entry: LoraEntry) -> None:
        with self._lock:
            # Replace if same ID exists
            self._library.entries = [e for e in self._library.entries if e.id != entry.id]
            self._library.entries.append(entry)
            self._save()

    def remove(self, lora_id: str) -> bool:
        with self._lock:
            before = len(self._library.entries)
            self._library.entries = [e for e in self._library.entries if e.id != lora_id]
            if len(self._library.entries) < before:
                self._save()
                return True
            return False

    def update_thumbnail(self, lora_id: str, thumbnail_url: str) -> bool:
        with self._lock:
            for entry in self._library.entries:
                if entry.id == lora_id:
                    entry.thumbnail_url = thumbnail_url
                    self._save()
                    return True
            return False
