"""GPU and API job executor implementations for the queue worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from api_types import (
    GenerateImageRequest,
    GenerateVideoRequest,
)
from state.job_queue import QueueJob

if TYPE_CHECKING:
    from handlers.image_generation_handler import ImageGenerationHandler
    from handlers.video_generation_handler import VideoGenerationHandler

logger = logging.getLogger(__name__)


def _str(params: dict[str, Any], key: str, default: str = "") -> str:
    v = params.get(key, default)
    return str(v) if v is not None else default


def _bool(params: dict[str, Any], key: str, default: bool = False) -> bool:
    v = params.get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def _int(params: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(params.get(key, default))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class GpuJobExecutor:
    """Executes video and image jobs on the local GPU."""

    def __init__(
        self,
        video_generation: "VideoGenerationHandler",
        image_generation: "ImageGenerationHandler",
    ) -> None:
        self._video = video_generation
        self._image = image_generation

    def execute(self, job: QueueJob) -> list[str]:
        logger.info("[QueueWorker] Executing GPU job %s (type=%s model=%s)", job.id, job.type, job.model)
        if job.type == "image":
            return self._execute_image(job)
        return self._execute_video(job)

    def _execute_video(self, job: QueueJob) -> list[str]:
        p = job.params
        req = GenerateVideoRequest(
            prompt=_str(p, "prompt"),
            imagePath=_str(p, "imagePath") or None,
            lastFramePath=_str(p, "lastFramePath") or None,
            audioPath=_str(p, "audioPath") or None,
            resolution=_str(p, "resolution", "540p"),
            duration=_str(p, "duration", "5"),
            fps=_str(p, "fps", "24"),
            audio=_str(p, "audio", "false"),
            cameraMotion=_str(p, "cameraMotion", "none"),
            aspectRatio=_str(p, "aspectRatio", "16:9"),
            model=job.model,
            negativePrompt=_str(p, "negativePrompt"),
        )
        result = self._video.generate(req)
        if result.video_path:
            return [result.video_path]
        return []

    def _execute_image(self, job: QueueJob) -> list[str]:
        p = job.params
        req = GenerateImageRequest(
            prompt=_str(p, "prompt"),
            width=_int(p, "width", 1920),
            height=_int(p, "height", 1080),
            numImages=_int(p, "numImages", 1),
            numSteps=_int(p, "numSteps", 4),
        )
        result = self._image.generate(req)
        return list(result.image_paths or [])


class ApiJobExecutor:
    """Executes jobs via the LTX Cloud API (forced-API mode)."""

    def __init__(
        self,
        video_generation: "VideoGenerationHandler",
        image_generation: "ImageGenerationHandler",
    ) -> None:
        self._video = video_generation
        self._image = image_generation

    def execute(self, job: QueueJob) -> list[str]:
        logger.info("[QueueWorker] Executing API job %s (type=%s model=%s)", job.id, job.type, job.model)
        if job.type == "image":
            p = job.params
            req = GenerateImageRequest(
                prompt=_str(p, "prompt"),
                width=_int(p, "width", 1920),
                height=_int(p, "height", 1080),
                numImages=_int(p, "numImages", 1),
                numSteps=_int(p, "numSteps", 4),
            )
            result = self._image.generate(req)
            return list(result.image_paths or [])
        else:
            p = job.params
            req = GenerateVideoRequest(
                prompt=_str(p, "prompt"),
                imagePath=_str(p, "imagePath") or None,
                lastFramePath=_str(p, "lastFramePath") or None,
                audioPath=_str(p, "audioPath") or None,
                resolution=_str(p, "resolution", "540p"),
                duration=_str(p, "duration", "5"),
                fps=_str(p, "fps", "24"),
                audio=_str(p, "audio", "false"),
                cameraMotion=_str(p, "cameraMotion", "none"),
                aspectRatio=_str(p, "aspectRatio", "16:9"),
                model=job.model,
                negativePrompt=_str(p, "negativePrompt"),
            )
            result = self._video.generate(req)
            if result.video_path:
                return [result.video_path]
            return []
