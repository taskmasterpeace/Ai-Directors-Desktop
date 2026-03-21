"""Routes for /api/lora — CivitAI search, download, and local library management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app_handler import AppHandler
from _routes._errors import HTTPError
from state import get_state_service

router = APIRouter(prefix="/api/lora", tags=["lora"])


# ── Request/Response Models ─────────────────────────────────────────


class LoraSearchRequest(BaseModel):
    query: str = ""
    baseModel: str = ""
    sort: str = "Most Downloaded"
    limit: int = 20
    page: int = 1
    nsfw: bool = False


class LoraDownloadRequest(BaseModel):
    downloadUrl: str
    fileName: str
    name: str
    thumbnailUrl: str = ""
    triggerPhrase: str = ""
    baseModel: str = ""
    civitaiModelId: int | None = None
    civitaiVersionId: int | None = None
    description: str = ""


class LoraImportRequest(BaseModel):
    filePath: str
    name: str = ""
    triggerPhrase: str = ""
    thumbnailPath: str = ""


# ── Search ──────────────────────────────────────────────────────────


@router.post("/search")
def route_search_civitai(
    body: LoraSearchRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    try:
        return handler.lora.search_civitai(
            query=body.query,
            base_model=body.baseModel,
            sort=body.sort,
            limit=body.limit,
            page=body.page,
            nsfw=body.nsfw,
        )
    except Exception as exc:
        raise HTTPError(502, f"CivitAI search failed: {exc}") from exc


# ── Download ────────────────────────────────────────────────────────


@router.post("/download")
def route_download_lora(
    body: LoraDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    try:
        entry = handler.lora.download_lora(
            download_url=body.downloadUrl,
            file_name=body.fileName,
            name=body.name,
            thumbnail_url=body.thumbnailUrl,
            trigger_phrase=body.triggerPhrase,
            base_model=body.baseModel,
            civitai_model_id=body.civitaiModelId,
            civitai_version_id=body.civitaiVersionId,
            description=body.description,
        )
        return {"status": "ok", "entry": _entry_to_dict(entry)}
    except Exception as exc:
        raise HTTPError(500, f"Download failed: {exc}") from exc


# ── Library ─────────────────────────────────────────────────────────


@router.get("/library")
def route_list_library(
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    return {"entries": handler.lora.list_library()}


@router.delete("/library/{lora_id}")
def route_delete_lora(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    deleted = handler.lora.delete_lora(lora_id)
    if not deleted:
        raise HTTPError(404, f"LoRA not found: {lora_id}")
    return {"status": "ok"}


@router.post("/import")
def route_import_lora(
    body: LoraImportRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, Any]:
    try:
        entry = handler.lora.import_local_lora(
            file_path=body.filePath,
            name=body.name,
            trigger_phrase=body.triggerPhrase,
            thumbnail_path=body.thumbnailPath,
        )
        return {"status": "ok", "entry": _entry_to_dict(entry)}
    except FileNotFoundError as exc:
        raise HTTPError(404, str(exc)) from exc


# ── Thumbnail serving ───────────────────────────────────────────────


@router.get("/thumbnail/{lora_id}")
def route_serve_thumbnail(
    lora_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> Any:
    from pathlib import Path
    from fastapi.responses import FileResponse

    entry = handler.lora.get_entry(lora_id)
    if entry is None:
        raise HTTPError(404, "LoRA not found")

    thumb = entry.get("thumbnail_url", "")
    if thumb and Path(thumb).exists():
        return FileResponse(path=thumb)

    raise HTTPError(404, "No thumbnail available")


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(entry)
