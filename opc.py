from __future__ import annotations

import time
from collections import Counter
from urllib.parse import quote, urlsplit

import requests

from .config import OpcConfig
from .models import TaskSubmission, VideoResult, WanVideoRequest, validate_request


GENERATION_PATH = "/v1/videos/generations"
TASK_PATH = "/v1/tasks/"
MODELS = {"3.0": "qwen/wan3.0-video/v1", "3.0-prime": "qwen/wan3.0-video-prime/v1"}
FAILED_STATUSES = {"FAILED", "CANCELED", "CANCELLED", "ERROR"}
MEDIA_TYPES = {
    ("Image", "FirstFrame"): "first_frame",
    ("Image", "LastFrame"): "last_frame",
    ("Image", "Reference"): "reference_image",
    ("Video", "Reference"): "reference_video",
    ("Audio", "Reference"): "reference_audio",
    ("File", "Reference"): "file",
    ("Link", "Reference"): "link",
}


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


def validate_media(categories: list[tuple[str, str]]) -> None:
    if any(item not in MEDIA_TYPES for item in categories):
        raise ValueError("Unsupported OPC media category/usage.")
    counts = Counter(MEDIA_TYPES[item] for item in categories)
    for kind, maximum in {"first_frame": 1, "last_frame": 1, "reference_image": 10,
                          "reference_video": 5, "reference_audio": 5, "file": 1, "link": 1}.items():
        if counts[kind] > maximum:
            raise ValueError(f"OPC supports at most {maximum} {kind} inputs.")
    frames = counts["first_frame"] + counts["last_frame"]
    if frames and frames != len(categories):
        raise ValueError("OPC first/last frames cannot be mixed with reference media, files, or links.")
    if counts["last_frame"] and not counts["first_frame"]:
        raise ValueError("OPC last_frame requires first_frame.")
    if counts["file"] and counts["link"]:
        raise ValueError("OPC file and link inputs are mutually exclusive.")


def build_payload(request: WanVideoRequest) -> dict:
    validate_request(request, prompt_required=False)
    if request.super_resolution != "Disabled" or request.negative_prompt.strip():
        raise ValueError("OPC requires super_resolution Disabled and an empty negative_prompt.")
    if request.seed is not None and not 0 <= request.seed <= 2147483647:
        raise ValueError("OPC seed must be between 0 and 2147483647, or -1 in the node to omit it.")
    validate_media([(item.get("Category"), item.get("Usage")) for item in request.file_infos])
    input_data = {}
    if request.prompt.strip():
        input_data["prompt"] = request.prompt
    media = []
    for item in request.file_infos:
        kind = MEDIA_TYPES[(item["Category"], item["Usage"])]
        url = str(item.get("Url") or "").strip()
        parsed = urlsplit(url)
        if not ((parsed.scheme in {"http", "https"} and parsed.netloc)
                or (item["Category"] == "Image" and url.startswith("data:image/"))):
            raise ValueError("OPC media must use a public HTTP(S) URL; images also accept Data URLs.")
        if kind in {"file", "link"} and request.enhance_prompt != "Enabled":
            raise ValueError("OPC file/link inputs require enhance_prompt Enabled.")
        media.append({"type": kind, "url": url})
    if media:
        input_data["media"] = media
    parameters = {
        "resolution": request.resolution, "ratio": request.aspect_ratio,
        "duration": request.duration, "audio": request.audio_generation == "Enabled",
        "prompt_extend": request.enhance_prompt == "Enabled", "watermark": False,
    }
    if request.seed is not None:
        parameters["seed"] = request.seed
    return {"model": MODELS[request.model_version], "input": input_data, "parameters": parameters}


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
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        if method == "POST":
            headers["X-DashScope-Async"] = "enable"
        try:
            response = self.session.request(
                method, self.config.base_url + path,
                headers=headers,
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
        if not 200 <= response.status_code < 300 or data.get("error") or data.get("error_code") or data.get("code"):
            if method == "POST" and response.status_code < 500:
                error_type = OpcError
            error = data.get("error") or data.get("message") or "Unknown error"
            message = (f"OPC HTTP {response.status_code}: {data.get('error_code') or data.get('code', '-')}: {error}; "
                       f"request_id={data.get('request_id', '-')}; mr_req_id={data.get('mr_req_id', '-')}")
            raise error_type(message.replace(self.config.api_key, "[REDACTED]"))
        return data

    def create_video(self, request: WanVideoRequest, *, prompt_required: bool) -> TaskSubmission:
        validate_request(request, prompt_required=prompt_required)
        payload = build_payload(request)
        if not payload["input"]:
            raise ValueError("OPC requires prompt or media.")
        data = self.request("POST", GENERATION_PATH, payload)
        try:
            output = task_output(data)
        except OpcError:
            raise OpcSubmissionUncertain("OPC returned invalid submission data; do not automatically resubmit.") from None
        envelope = data.get("data", data)
        task_id = str(envelope.get("task_id") or envelope.get("id") or output.get("task_id") or output.get("id") or "").strip()
        if not task_id:
            raise OpcSubmissionUncertain("OPC returned no task_id; do not automatically resubmit.")
        print(f"[Wan 3.0 API] OPC task_id={task_id}")
        return TaskSubmission(task_id, str(data.get("request_id") or data.get("mr_req_id") or ""))

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
            message = (f"OPC task {task_id}: {status}: {output.get('code', '-')}: "
                       f"{output.get('error') or output.get('message') or status}; "
                       f"request_id={detail.get('request_id', '-')}; mr_req_id={detail.get('mr_req_id', '-')}")
            raise OpcTaskError(message.replace(self.config.api_key, "[REDACTED]"), task_id, status in FAILED_STATUSES)
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
