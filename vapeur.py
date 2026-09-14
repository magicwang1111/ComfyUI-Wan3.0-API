from __future__ import annotations

import time
from urllib.parse import quote

import requests

from .config import VapeurConfig
from .models import TaskSubmission, VideoResult, WanVideoRequest, validate_request


GENERATION_PATH = "/qwen/api/v1/services/aigc/video-generation/generation"
TASK_PATH = "/qwen/api/v1/tasks/video/"
FAILED_STATUSES = {"FAILED", "CANCELED", "CANCELLED"}


class VapeurError(RuntimeError):
    pass


class VapeurSubmissionUncertain(VapeurError):
    pass


class VapeurTaskError(VapeurError):
    def __init__(self, message: str, task_id: str, terminal: bool):
        super().__init__(message)
        self.task_id = task_id
        self.terminal = terminal


def build_payload(config: VapeurConfig, request: WanVideoRequest) -> dict:
    validate_request(request, prompt_required=False)
    if request.super_resolution != "Disabled":
        raise ValueError("Vapeur supports 480P/720P/1080P; set super_resolution to Disabled.")
    if request.seed is not None and not 0 <= request.seed <= 2147483647:
        raise ValueError("Vapeur seed must be between 0 and 2147483647.")
    media_types = {
        ("Image", "FirstFrame"): "first_frame",
        ("Image", "LastFrame"): "last_frame",
        ("Image", "Reference"): "reference_image",
        ("Video", "Reference"): "reference_video",
        ("Audio", "Reference"): "reference_audio",
    }
    media = []
    for item in request.file_infos:
        kind = media_types.get((item.get("Category"), item.get("Usage")))
        if not kind:
            raise ValueError("Unsupported Vapeur media category/usage.")
        media.append({"type": kind, "url": item["Url"]})
    input_data = {"prompt": request.prompt, "media": media}
    if request.negative_prompt.strip():
        input_data["negative_prompt"] = request.negative_prompt.strip()
    parameters = {
        "resolution": request.resolution,
        "ratio": request.aspect_ratio,
        "duration": request.duration,
        "audio": request.audio_generation == "Enabled",
        "prompt_extend": request.enhance_prompt == "Enabled",
        "watermark": False,
    }
    if request.seed is not None:
        parameters["seed"] = request.seed
    prime = "-prime" if request.model_version == "3.0-prime" else ""
    return {"model": f"wan3.0-video{prime}-{config.region}", "input": input_data, "parameters": parameters}


class VapeurClient:
    def __init__(self, config: VapeurConfig):
        self.config = config
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            response = self.session.request(
                method, self.config.base_url + path,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self.config.request_timeout, allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise VapeurError(f"Vapeur {method} failed ({type(exc).__name__}); submission outcome may be unknown. Do not automatically resubmit.") from None
        try:
            data = response.json()
        except ValueError:
            raise VapeurError(f"Vapeur returned non-JSON HTTP {response.status_code}.") from None
        if not isinstance(data, dict):
            raise VapeurError(f"Vapeur returned invalid JSON HTTP {response.status_code}.")
        if not 200 <= response.status_code < 300 or data.get("error") or (data.get("code") and not data.get("output")):
            error = data.get("error") or data
            message = str(error.get("message") or error.get("code") or "Unknown error") if isinstance(error, dict) else str(error)
            message = message.replace(self.config.api_key, "[REDACTED]")
            raise VapeurError(f"Vapeur HTTP {response.status_code}: {message}; request_id={data.get('request_id', '-')}")
        return data

    def create_video(self, request: WanVideoRequest, *, prompt_required: bool) -> TaskSubmission:
        validate_request(request, prompt_required=prompt_required)
        payload = build_payload(self.config, request)
        try:
            data = self.request("POST", GENERATION_PATH, payload)
        except VapeurError as exc:
            raise VapeurSubmissionUncertain(str(exc)) from None
        output = data.get("output") or {}
        task_id = str(output.get("task_id") or "").strip()
        if not task_id:
            raise VapeurSubmissionUncertain("Vapeur submission returned no task_id; do not automatically resubmit.")
        print(f"[Wan 3.0 API] Vapeur task_id={task_id}")
        return TaskSubmission(task_id, str(data.get("request_id") or ""))

    def describe_task(self, task_id: str) -> dict:
        return self.request("GET", TASK_PATH + quote(task_id, safe=""))

    @staticmethod
    def status(detail: dict) -> str:
        return str((detail.get("output") or {}).get("task_status") or "UNKNOWN").upper()

    def check_task(self, detail: dict, task_id: str) -> str:
        status = self.status(detail)
        if status in FAILED_STATUSES or status == "UNKNOWN":
            output = detail.get("output") or {}
            message = str(output.get("message") or output.get("code") or status).replace(self.config.api_key, "[REDACTED]")
            raise VapeurTaskError(f"Vapeur task {task_id}: {message}", task_id, status in FAILED_STATUSES)
        return status

    def wait_for_task(self, task_id: str) -> dict:
        started = time.monotonic()
        last_status = None
        while True:
            try:
                detail = self.describe_task(task_id)
            except VapeurError as exc:
                raise VapeurTaskError(f"Vapeur task {task_id}: {exc}; query this task_id to recover.", task_id, False) from None
            status = self.check_task(detail, task_id)
            if status != last_status:
                print(f"[Wan 3.0 API] Vapeur task {task_id}: {status}")
                last_status = status
            if status == "SUCCEEDED":
                return detail
            if time.monotonic() - started >= self.config.max_wait_seconds:
                raise VapeurTaskError(f"Vapeur task {task_id} timed out; query this task_id to recover.", task_id, False)
            time.sleep(self.config.poll_interval)


def video_result(detail: dict, submission: TaskSubmission) -> VideoResult:
    output = detail.get("output") or {}
    url = str(output.get("video_url") or "")
    if not url:
        raise VapeurTaskError(f"Vapeur task {submission.task_id} completed but returned no video_url.", submission.task_id, True)
    return VideoResult(url, submission.task_id, submission.task_id, submission.request_id, VapeurClient.status(detail), detail)
