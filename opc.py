from __future__ import annotations

import base64
import io
import time
from urllib.parse import quote, urlsplit

import requests
from PIL import Image

from .config import OpcConfig
from .models import MediaBlob, TaskSubmission, VideoResult, WanVideoRequest, validate_request


GENERATION_PATH = "/v1/videos/generations"
TASK_PATH = "/v1/tasks/"
MODELS = {"3.0": "qwen/wan3.0-video/v1", "3.0-prime": "qwen/wan3.0-video-prime/v1"}
FAILED_STATUSES = {"FAILED", "CANCELED", "CANCELLED", "ERROR"}


class OpcError(RuntimeError):
    pass


class OpcSubmissionUncertain(OpcError):
    pass


class OpcTransientError(OpcError):
    pass


class OpcTaskError(OpcError):
    def __init__(self, message: str, task_id: str, terminal: bool):
        super().__init__(message)
        self.task_id = task_id
        self.terminal = terminal


def build_payload(request: WanVideoRequest, media: list[tuple[MediaBlob, str, str]] | None = None) -> dict:
    validate_request(request, prompt_required=True)
    if request.super_resolution != "Disabled" or request.enhance_prompt != "Disabled" or request.negative_prompt.strip():
        raise ValueError("OPC requires super_resolution/enhance_prompt Disabled and an empty negative_prompt.")
    if request.seed is not None:
        raise ValueError("OPC does not document seed; set seed to -1.")
    if request.audio_generation != "Enabled":
        raise ValueError("OPC does not expose audio control; leave audio_generation Enabled (provider default).")
    if request.duration < 1:
        raise ValueError("OPC requires a fixed duration; smart duration (-1) is not supported.")
    media = media or []
    if len(media) + len(request.file_infos) > 1:
        raise ValueError("OPC supports only one image, without a last frame, video, or audio reference.")
    ratio = request.aspect_ratio
    payload = {"model": MODELS[request.model_version], "prompt": request.prompt,
               "duration": request.duration, "watermark": False}
    dimensions = None
    if media:
        blob, category, usage = media[0]
        if category != "Image" or usage not in {"FirstFrame", "Reference"}:
            raise ValueError("OPC supports only a first frame or single reference image; no last frame/video/audio.")
        payload["image"] = f"data:{blob.content_type};base64," + base64.b64encode(blob.data).decode("ascii")
        if ratio == "adaptive":
            with Image.open(io.BytesIO(blob.data)) as image:
                dimensions = image.size
    elif request.file_infos:
        item = request.file_infos[0]
        if item.get("Category") != "Image" or item.get("Usage") not in {"FirstFrame", "Reference"}:
            raise ValueError("OPC supports only a first frame or single reference image; no last frame/video/audio.")
        image_url = str(item.get("Url") or "")
        parsed = urlsplit(image_url)
        if not ((parsed.scheme == "https" and parsed.netloc) or image_url.startswith("data:image/")):
            raise ValueError("OPC image must be an HTTPS URL or image Data URL.")
        payload["image"] = image_url
    if ratio != "adaptive":
        dimensions = tuple(int(value) for value in ratio.split(":"))
    if dimensions:
        width, height = dimensions
        short_side = int(request.resolution[:-1])
        scale = short_side / min(width, height)
        payload["size"] = f"{round(width * scale / 2) * 2}x{round(height * scale / 2) * 2}"
    return payload


def task_output(detail: dict) -> dict:
    # Accept the router envelope and the upstream Wan output envelope.
    data = detail.get("data", detail)
    if not isinstance(data, dict):
        raise OpcError("OPC returned invalid task data.")
    output = data.get("output", data)
    if not isinstance(output, dict):
        raise OpcError("OPC returned invalid task output.")
    return output


class OpcClient:
    def __init__(self, config: OpcConfig):
        self.config = config
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.session.close()
        return False

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            response = self.session.request(
                method, self.config.base_url + path,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self.config.request_timeout, allow_redirects=False,
            )
        except requests.RequestException as exc:
            if method == "POST":
                raise OpcSubmissionUncertain(
                    f"OPC submission failed ({type(exc).__name__}); outcome unknown. Do not automatically resubmit."
                ) from None
            error_type = OpcTransientError if isinstance(exc, (requests.Timeout, requests.ConnectionError)) else OpcError
            raise error_type(f"OPC query failed ({type(exc).__name__}).") from None
        transient = method == "GET" and response.status_code in {408, 429, 500, 502, 503, 504}
        error_type = OpcSubmissionUncertain if method == "POST" else OpcTransientError if transient else OpcError
        try:
            data = response.json()
        except ValueError:
            raise error_type(f"OPC returned non-JSON HTTP {response.status_code}.") from None
        if not isinstance(data, dict):
            raise error_type(f"OPC returned invalid JSON HTTP {response.status_code}.")
        if not 200 <= response.status_code < 300 or data.get("error") or data.get("error_code"):
            if method == "POST" and response.status_code < 500:
                error_type = OpcError
            error = data.get("error") or data.get("message") or "Unknown error"
            message = (f"OPC HTTP {response.status_code}: {data.get('error_code', '-')}: {error}; "
                       f"request_id={data.get('mr_req_id') or data.get('request_id', '-')}")
            raise error_type(message.replace(self.config.api_key, "[REDACTED]"))
        return data

    def create_video(self, request: WanVideoRequest, *, prompt_required: bool, media=None) -> TaskSubmission:
        data = self.request("POST", GENERATION_PATH, build_payload(request, media))
        try:
            output = task_output(data)
        except OpcError:
            raise OpcSubmissionUncertain("OPC returned invalid submission data; do not automatically resubmit.") from None
        envelope = data.get("data", data)
        task_id = str(envelope.get("task_id") or envelope.get("id") or output.get("task_id") or output.get("id") or "").strip()
        if not task_id:
            raise OpcSubmissionUncertain("OPC returned no task_id; do not automatically resubmit.")
        print(f"[Wan 3.0 API] OPC task_id={task_id}")
        return TaskSubmission(task_id, str(data.get("mr_req_id") or data.get("request_id") or ""))

    def describe_task(self, task_id: str) -> dict:
        return self.request("GET", TASK_PATH + quote(task_id, safe=""))

    @staticmethod
    def status(detail: dict) -> str:
        output = task_output(detail)
        envelope = detail.get("data", detail)
        status = str(envelope.get("task_status") or envelope.get("status") or output.get("task_status") or output.get("status") or "UNKNOWN").upper()
        return {"QUEUED": "PENDING", "PROCESSING": "RUNNING", "IN_PROGRESS": "RUNNING",
                "COMPLETED": "SUCCEEDED", "SUCCESS": "SUCCEEDED"}.get(status, status)

    def check_task(self, detail: dict, task_id: str) -> str:
        status = self.status(detail)
        if status not in {"PENDING", "RUNNING", "SUCCEEDED"}:
            output = task_output(detail)
            message = str(output.get("error") or output.get("message") or status).replace(self.config.api_key, "[REDACTED]")
            raise OpcTaskError(f"OPC task {task_id}: {status}: {message}", task_id, status in FAILED_STATUSES)
        return status

    def wait_for_task(self, task_id: str) -> dict:
        deadline = time.monotonic() + self.config.max_wait_seconds
        consecutive_errors = 0
        while True:
            try:
                detail = self.describe_task(task_id)
                status = self.check_task(detail, task_id)
            except OpcTransientError as exc:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    raise OpcTaskError(f"OPC task {task_id}: {exc}; query this task_id to recover.", task_id, False) from None
                delay = min(self.config.poll_interval * 2 ** (consecutive_errors - 1), 30)
            except OpcTaskError:
                raise
            except OpcError as exc:
                raise OpcTaskError(f"OPC task {task_id}: {exc}; query this task_id to recover.", task_id, False) from None
            else:
                consecutive_errors = 0
                if status == "SUCCEEDED":
                    return detail
                delay = self.config.poll_interval
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpcTaskError(f"OPC task {task_id} timed out; query this task_id to recover.", task_id, False)
            time.sleep(min(delay, remaining))
            if time.monotonic() >= deadline:
                raise OpcTaskError(f"OPC task {task_id} timed out; query this task_id to recover.", task_id, False)


def video_result(detail: dict, submission: TaskSubmission) -> VideoResult:
    output = task_output(detail)
    url = str(output.get("video_url") or output.get("url") or "")
    if not url:
        raise OpcTaskError(f"OPC task {submission.task_id} completed but returned no video_url.", submission.task_id, True)
    return VideoResult(url, submission.task_id, submission.task_id, submission.request_id, OpcClient.status(detail), detail)
