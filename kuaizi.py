from __future__ import annotations

import time
from urllib.parse import quote

import requests

from .config import KuaiziConfig
from .models import TaskSubmission, VideoResult, WanVideoRequest, validate_request


GENERATION_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/"
FAILED_STATUSES = {"FAILED", "CANCELED"}


class KuaiziError(RuntimeError):
    pass


class KuaiziSubmissionUncertain(KuaiziError):
    pass


class KuaiziTransientError(KuaiziError):
    pass


class KuaiziTaskError(KuaiziError):
    def __init__(self, message: str, task_id: str, terminal: bool):
        super().__init__(message)
        self.task_id = task_id
        self.terminal = terminal


def build_payload(request: WanVideoRequest) -> dict:
    validate_request(request, prompt_required=False)
    if request.super_resolution != "Disabled":
        raise ValueError("Kuaizi supports 480P/720P/1080P; set super_resolution to Disabled.")
    if request.negative_prompt.strip() or request.enhance_prompt != "Disabled":
        raise ValueError("Kuaizi does not document negative_prompt or enhance_prompt; leave them empty/Disabled.")
    if request.seed is not None and not 0 <= request.seed <= 2147483647:
        raise ValueError("Kuaizi seed must be between 0 and 2147483647.")
    media_types = {
        ("Image", "FirstFrame"): "first_frame",
        ("Image", "LastFrame"): "last_frame",
        ("Image", "Reference"): "reference_image",
        ("Video", "Reference"): "reference_video",
        ("Audio", "Reference"): "reference_audio",
    }
    input_data = {}
    if request.prompt.strip():
        input_data["prompt"] = request.prompt
    if request.file_infos:
        media = []
        for item in request.file_infos:
            kind = media_types.get((item.get("Category"), item.get("Usage")))
            if not kind:
                raise ValueError("Unsupported Kuaizi media category/usage.")
            media.append({"type": kind, "url": item["Url"]})
        input_data["media"] = media
    parameters = {
        "resolution": request.resolution,
        "ratio": request.aspect_ratio,
        "duration": request.duration,
        "audio": request.audio_generation == "Enabled",
        "watermark": False,
    }
    if request.seed is not None:
        parameters["seed"] = request.seed
    model = "wan3.0-video-prime" if request.model_version == "3.0-prime" else "wan3.0-video"
    return {"model": model, "input": input_data, "parameters": parameters}


class KuaiziClient:
    def __init__(self, config: KuaiziConfig):
        self.config = config
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.session.close()
        return False

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        if method == "POST":
            headers["X-DashScope-Async"] = "enable"
        try:
            response = self.session.request(
                method, self.config.base_url + path, headers=headers,
                json=payload, timeout=self.config.request_timeout, allow_redirects=False,
            )
        except requests.RequestException as exc:
            if method == "POST":
                raise KuaiziSubmissionUncertain(
                    f"Kuaizi submission failed ({type(exc).__name__}); outcome unknown. Do not automatically resubmit."
                ) from None
            error_type = KuaiziTransientError if isinstance(exc, (requests.Timeout, requests.ConnectionError)) else KuaiziError
            raise error_type(f"Kuaizi query failed ({type(exc).__name__}).") from None
        if method == "GET" and response.status_code in {408, 429, 500, 502, 503, 504}:
            raise KuaiziTransientError(f"Kuaizi query returned HTTP {response.status_code}.")
        error_type = KuaiziSubmissionUncertain if method == "POST" else KuaiziError
        try:
            data = response.json()
        except ValueError:
            raise error_type(f"Kuaizi returned non-JSON HTTP {response.status_code}.") from None
        if not isinstance(data, dict):
            raise error_type(f"Kuaizi returned invalid JSON HTTP {response.status_code}.")
        if not 200 <= response.status_code < 300 or data.get("code"):
            message = str(data.get("message") or "Unknown error").replace(self.config.api_key, "[REDACTED]")
            error_type = KuaiziSubmissionUncertain if method == "POST" and response.status_code >= 500 else KuaiziError
            raise error_type(
                f"Kuaizi HTTP {response.status_code}: {data.get('code', '-')}: {message}; request_id={data.get('request_id', '-')}"
            )
        return data

    def create_video(self, request: WanVideoRequest, *, prompt_required: bool) -> TaskSubmission:
        validate_request(request, prompt_required=prompt_required)
        payload = build_payload(request)
        if not payload["input"]:
            raise ValueError("Kuaizi requires prompt or media.")
        data = self.request("POST", GENERATION_PATH, payload)
        output = data.get("output")
        task_id = str(output.get("task_id") or "").strip() if isinstance(output, dict) else ""
        if not task_id:
            raise KuaiziSubmissionUncertain("Kuaizi returned no task_id; do not automatically resubmit.")
        print(f"[Wan 3.0 API] Kuaizi task_id={task_id}")
        return TaskSubmission(task_id, str(data.get("request_id") or ""))

    def describe_task(self, task_id: str) -> dict:
        return self.request("GET", TASK_PATH + quote(task_id, safe=""))

    @staticmethod
    def status(detail: dict) -> str:
        return str((detail.get("output") or {}).get("task_status") or "UNKNOWN").upper()

    def check_task(self, detail: dict, task_id: str) -> str:
        status = self.status(detail)
        if status not in {"PENDING", "RUNNING", "SUCCEEDED"}:
            output = detail.get("output") or {}
            message = str(output.get("message") or status).replace(self.config.api_key, "[REDACTED]")
            raise KuaiziTaskError(
                f"Kuaizi task {task_id}: {output.get('code', status)}: {message}; request_id={detail.get('request_id', '-')}",
                task_id, status in FAILED_STATUSES,
            )
        return status

    def wait_for_task(self, task_id: str) -> dict:
        deadline = time.monotonic() + self.config.max_wait_seconds
        consecutive_errors = 0
        while True:
            try:
                detail = self.describe_task(task_id)
            except KuaiziTransientError as exc:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    raise KuaiziTaskError(f"Kuaizi task {task_id}: {exc}; query this task_id to recover.", task_id, False) from None
                delay = min(self.config.poll_interval * 2 ** (consecutive_errors - 1), 60)
            except KuaiziError as exc:
                raise KuaiziTaskError(f"Kuaizi task {task_id}: {exc}; query this task_id to recover.", task_id, False) from None
            else:
                consecutive_errors = 0
                if self.check_task(detail, task_id) == "SUCCEEDED":
                    return detail
                delay = self.config.poll_interval
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KuaiziTaskError(f"Kuaizi task {task_id} timed out; query this task_id to recover.", task_id, False)
            time.sleep(min(delay, remaining))
            if time.monotonic() >= deadline:
                raise KuaiziTaskError(f"Kuaizi task {task_id} timed out; query this task_id to recover.", task_id, False)


def video_result(detail: dict, submission: TaskSubmission) -> VideoResult:
    url = str((detail.get("output") or {}).get("video_url") or "")
    if not url:
        raise KuaiziTaskError(f"Kuaizi task {submission.task_id} completed but returned no video_url.", submission.task_id, True)
    return VideoResult(url, submission.task_id, submission.task_id, submission.request_id, KuaiziClient.status(detail), detail)
