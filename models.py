from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MODEL_VERSIONS = ["3.0", "3.0-prime"]
RESOLUTIONS = ["480P", "720P", "1080P"]
SUPER_RESOLUTIONS = ["Disabled", "2K", "4K"]
AUDIO_GENERATION_OPTIONS = ["Enabled", "Disabled"]
DURATIONS = [-1, *range(2, 31)]
ASPECT_RATIOS = ["16:9", "9:16", "4:3", "3:4", "1:1", "adaptive"]
PROMPT_MAX_CHARS = 20000


@dataclass(frozen=True)
class MediaBlob:
    data: bytes
    content_type: str
    extension: str
    duration: float | None = None


@dataclass(frozen=True)
class UploadedObject:
    object_key: str
    url: str


@dataclass
class WanVideoRequest:
    model_version: str
    prompt: str
    resolution: str
    aspect_ratio: str
    duration: int
    session_id: str
    negative_prompt: str = ""
    enhance_prompt: str = "Disabled"
    seed: int | None = None
    super_resolution: str = "Disabled"
    audio_generation: str = "Enabled"
    file_infos: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskSubmission:
    task_id: str
    request_id: str


@dataclass(frozen=True)
class VideoResult:
    video_url: str
    video_id: str
    task_id: str
    request_id: str
    status: str
    raw: dict[str, Any]


def validate_request(request: WanVideoRequest, *, prompt_required: bool) -> None:
    if request.model_version not in MODEL_VERSIONS:
        raise ValueError(f"model_version must be one of: {', '.join(MODEL_VERSIONS)}")
    if request.resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of: {', '.join(RESOLUTIONS)}")
    if request.super_resolution not in SUPER_RESOLUTIONS:
        raise ValueError(
            f"super_resolution must be one of: {', '.join(SUPER_RESOLUTIONS)}"
        )
    if request.audio_generation not in AUDIO_GENERATION_OPTIONS:
        raise ValueError("audio_generation must be Enabled or Disabled.")
    if request.aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"aspect_ratio must be one of: {', '.join(ASPECT_RATIOS)}")
    if prompt_required and not request.prompt.strip():
        raise ValueError("prompt is required for text-to-video.")
    if len(request.prompt) > PROMPT_MAX_CHARS:
        raise ValueError(f"prompt must be {PROMPT_MAX_CHARS} characters or fewer.")
    if request.duration != -1 and not 2 <= request.duration <= 30:
        raise ValueError("duration must be -1 (smart duration) or between 2 and 30 seconds.")
    if request.enhance_prompt not in {"Enabled", "Disabled"}:
        raise ValueError("enhance_prompt must be Enabled or Disabled.")
    if len(request.session_id) > 50:
        raise ValueError("session_id must be 50 characters or fewer.")
