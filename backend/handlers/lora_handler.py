"""Handler for LoRA library operations — search CivitAI, download, manage local catalog."""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from state.lora_library import LoraEntry, LoraLibraryStore

_logger = logging.getLogger(__name__)

CIVITAI_API_BASE = "https://civitai.com/api/v1"


class LoraHandler:
    """Search CivitAI, download LoRAs, and manage the local catalog."""

    def __init__(self, store: LoraLibraryStore, civitai_api_key: str = "") -> None:
        self._store = store
        self._civitai_api_key = civitai_api_key

    def set_api_key(self, key: str) -> None:
        self._civitai_api_key = key

    # ── CivitAI Search ──────────────────────────────────────────────

    def search_civitai(
        self,
        query: str = "",
        base_model: str = "",
        sort: str = "Most Downloaded",
        limit: int = 20,
        page: int = 1,
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """Search CivitAI for LORA models. Returns raw API response."""
        params: dict[str, str | int | bool] = {
            "types": "LORA",
            "limit": limit,
            "page": page,
            "sort": sort,
            "nsfw": nsfw,
        }
        if query:
            params["query"] = query
        if base_model:
            params["baseModels"] = base_model

        headers: dict[str, str] = {}
        if self._civitai_api_key:
            headers["Authorization"] = f"Bearer {self._civitai_api_key}"

        resp = requests.get(
            f"{CIVITAI_API_BASE}/models",
            params=params,  # type: ignore[arg-type]
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        # Normalize to a cleaner format for the frontend
        items: list[dict[str, Any]] = []
        for model in data.get("items", []):
            versions = model.get("modelVersions", [])
            if not versions:
                continue
            latest = versions[0]
            files = latest.get("files", [])
            safetensors_file = next(
                (f for f in files if f.get("name", "").endswith(".safetensors")),
                files[0] if files else None,
            )

            # Get preview image
            images = latest.get("images", [])
            thumbnail = images[0].get("url", "") if images else ""

            # Get trigger words
            trigger_words = latest.get("trainedWords", [])

            items.append({
                "civitaiModelId": model.get("id"),
                "civitaiVersionId": latest.get("id"),
                "name": model.get("name", "Unknown"),
                "description": (model.get("description") or "")[:200],
                "thumbnailUrl": thumbnail,
                "triggerPhrase": ", ".join(trigger_words) if trigger_words else "",
                "baseModel": latest.get("baseModel", ""),
                "downloadUrl": safetensors_file.get("downloadUrl", "") if safetensors_file else "",
                "fileSizeBytes": safetensors_file.get("sizeKB", 0) * 1024 if safetensors_file else 0,
                "fileName": safetensors_file.get("name", "") if safetensors_file else "",
                "stats": {
                    "downloadCount": model.get("stats", {}).get("downloadCount", 0),
                    "favoriteCount": model.get("stats", {}).get("favoriteCount", 0),
                    "thumbsUpCount": model.get("stats", {}).get("thumbsUpCount", 0),
                    "rating": model.get("stats", {}).get("rating", 0),
                },
                # Check if already downloaded
                "isDownloaded": self._is_downloaded(model.get("id"), latest.get("id")),
            })

        return {
            "items": items,
            "metadata": data.get("metadata", {}),
        }

    def _is_downloaded(self, model_id: int | None, version_id: int | None) -> bool:
        if model_id is None:
            return False
        for entry in self._store.list_all():
            if entry.civitai_model_id == model_id:
                if version_id is None or entry.civitai_version_id == version_id:
                    return True
        return False

    # ── Download ────────────────────────────────────────────────────

    def download_lora(
        self,
        download_url: str,
        file_name: str,
        name: str,
        thumbnail_url: str = "",
        trigger_phrase: str = "",
        base_model: str = "",
        civitai_model_id: int | None = None,
        civitai_version_id: int | None = None,
        description: str = "",
        on_progress: Any = None,
    ) -> LoraEntry:
        """Download a LoRA file and add it to the catalog."""
        loras_dir = self._store.loras_dir
        dest = loras_dir / file_name

        headers: dict[str, str] = {}
        if self._civitai_api_key:
            headers["Authorization"] = f"Bearer {self._civitai_api_key}"

        _logger.info("Downloading LoRA %s to %s", name, dest)
        resp = requests.get(download_url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress and total > 0:
                    on_progress(downloaded, total)

        # Download thumbnail locally
        local_thumb = ""
        if thumbnail_url:
            try:
                local_thumb = self._download_thumbnail(thumbnail_url, file_name)
            except Exception:
                _logger.warning("Failed to download thumbnail for %s", name, exc_info=True)
                local_thumb = thumbnail_url  # Fall back to remote URL

        lora_id = hashlib.sha256(f"{civitai_model_id}:{civitai_version_id}:{file_name}".encode()).hexdigest()[:16]

        entry = LoraEntry(
            id=lora_id,
            name=name,
            file_path=str(dest),
            file_size_bytes=dest.stat().st_size,
            thumbnail_url=local_thumb,
            trigger_phrase=trigger_phrase,
            base_model=base_model,
            civitai_model_id=civitai_model_id,
            civitai_version_id=civitai_version_id,
            description=description,
        )
        self._store.add(entry)
        _logger.info("LoRA %s downloaded and cataloged (id=%s)", name, lora_id)
        return entry

    def _download_thumbnail(self, url: str, lora_filename: str) -> str:
        """Download thumbnail image to loras/thumbnails/ and return local path."""
        thumb_dir = self._store.loras_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)

        stem = Path(lora_filename).stem
        # Detect extension from URL
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        elif ".webp" in url.lower():
            ext = ".webp"

        thumb_path = thumb_dir / f"{stem}{ext}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        thumb_path.write_bytes(resp.content)
        return str(thumb_path)

    # ── Local Library ───────────────────────────────────────────────

    def get_entry(self, lora_id: str) -> dict[str, Any] | None:
        entry = self._store.get(lora_id)
        if entry is None:
            return None
        return asdict(entry)

    def list_library(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._store.list_all()]

    def delete_lora(self, lora_id: str) -> bool:
        entry = self._store.get(lora_id)
        if entry is None:
            return False

        # Delete the file
        lora_path = Path(entry.file_path)
        if lora_path.exists():
            lora_path.unlink()

        # Delete thumbnail if local
        if entry.thumbnail_url and Path(entry.thumbnail_url).exists():
            try:
                Path(entry.thumbnail_url).unlink()
            except Exception:
                pass

        return self._store.remove(lora_id)

    def import_local_lora(
        self,
        file_path: str,
        name: str = "",
        trigger_phrase: str = "",
        thumbnail_path: str = "",
    ) -> LoraEntry:
        """Import a LoRA from the local filesystem into the library."""
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"LoRA file not found: {file_path}")

        # Copy to loras dir if not already there
        dest = self._store.loras_dir / src.name
        if dest != src:
            shutil.copy2(src, dest)

        lora_id = hashlib.sha256(f"local:{src.name}".encode()).hexdigest()[:16]

        # Copy thumbnail if provided
        local_thumb = ""
        if thumbnail_path and Path(thumbnail_path).exists():
            thumb_dir = self._store.loras_dir / "thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            thumb_dest = thumb_dir / f"{src.stem}{Path(thumbnail_path).suffix}"
            shutil.copy2(thumbnail_path, thumb_dest)
            local_thumb = str(thumb_dest)

        entry = LoraEntry(
            id=lora_id,
            name=name or src.stem,
            file_path=str(dest),
            file_size_bytes=dest.stat().st_size,
            thumbnail_url=local_thumb,
            trigger_phrase=trigger_phrase,
        )
        self._store.add(entry)
        return entry
