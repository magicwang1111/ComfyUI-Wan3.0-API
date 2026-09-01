from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any

import requests

from .config import TencentConfig
from .models import TaskSubmission, VideoResult, WanVideoRequest, validate_request


VOD_ENDPOINT = "vod.tencentcloudapi.com"
VOD_URL = f"https://{VOD_ENDPOINT}/"
VOD_SERVICE = "vod"
VOD_VERSION = "2018-07-17"
CREATE_ACTION = "CreateAigcVideoTask"
DESCRIBE_ACTION = "DescribeTaskDetail"
TERMINAL_STATUSES = {"FINISH", "SUCCESS", "FAIL", "FAILED", "ABORTED"}
FAILED_STATUSES = {"FAIL", "FAILED", "ABORTED"}


class TencentVodError(RuntimeError):
    pass


class TencentVodTaskError(TencentVodError):
    def __init__(self, message: str, task_id: str, terminal: bool):
        super().__init__(message)
        self.task_id = task_id
        self.terminal = terminal


class TencentVodClient:
    def __init__(self, config: TencentConfig):
        self.config = config
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @staticmethod
    def canonical_json(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _signature(self, date: str, string_to_sign: str) -> str:
        secret_date = hmac.new(
            ("TC3" + self.config.secret_key).encode("utf-8"),
            date.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        secret_service = hmac.new(secret_date, VOD_SERVICE.encode("utf-8"), hashlib.sha256).digest()
        secret_signing = hmac.new(secret_service, b"tc3_request", hashlib.sha256).digest()
        return hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    def headers(self, action: str, payload: dict, timestamp: int | None = None) -> dict[str, str]:
        timestamp = int(time.time()) if timestamp is None else timestamp
        date = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).strftime("%Y-%m-%d")
        body = self.canonical_json(payload)
        canonical_request = "\n".join([
            "POST",
            "/",
            "",
            f"content-type:application/json; charset=utf-8\nhost:{VOD_ENDPOINT}\nx-tc-action:{action.lower()}\n",
            "content-type;host;x-tc-action",
            hashlib.sha256(body.encode("utf-8")).hexdigest(),
        ])
        scope = f"{date}/{VOD_SERVICE}/tc3_request"
        string_to_sign = "\n".join([
            "TC3-HMAC-SHA256",
            str(timestamp),
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])
        authorization = (
            f"TC3-HMAC-SHA256 Credential={self.config.secret_id}/{scope}, "
            f"SignedHeaders=content-type;host;x-tc-action, Signature={self._signature(date, string_to_sign)}"
        )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": VOD_ENDPOINT,
            "X-TC-Action": action,
            "X-TC-Version": VOD_VERSION,
            "X-TC-Region": self.config.region,
            "X-TC-Timestamp": str(timestamp),
        }

    def request(self, action: str, payload: dict) -> dict:
        body = self.canonical_json(payload)
        response = self.session.post(
            VOD_URL,
            headers=self.headers(action, payload),
            data=body.encode("utf-8"),
            timeout=self.config.request_timeout,
        )
        try:
            decoded = response.json()
        except ValueError as exc:
            raise TencentVodError(f"{action} returned non-JSON HTTP {response.status_code}.") from exc
        top = decoded.get("Response")
        if response.status_code != 200 or not isinstance(top, dict):
            raise TencentVodError(f"{action} HTTP {response.status_code} returned an invalid response.")
        if "Error" in top:
            error = top.get("Error") or {}
            raise TencentVodError(
                f"{action} Code={error.get('Code')}, Message={error.get('Message')}, "
                f"RequestId={top.get('RequestId')}"
            )
        return top

    def create_video(self, request: WanVideoRequest, *, prompt_required: bool) -> TaskSubmission:
        validate_request(request, prompt_required=prompt_required)
        payload = build_payload(self.config, request)
        response = self.request(CREATE_ACTION, payload)
        task_id = str(response.get("TaskId") or "").strip()
        request_id = str(response.get("RequestId") or "").strip()
        if not task_id:
            raise TencentVodError(f"{CREATE_ACTION} returned no TaskId; RequestId={request_id or '-'}")
        return TaskSubmission(task_id=task_id, request_id=request_id)

    def describe_task(self, task_id: str) -> dict:
        return self.request(DESCRIBE_ACTION, {
            "TaskId": task_id,
            "SubAppId": self.config.sub_app_id,
        })

    @staticmethod
    def extract_task(detail: dict) -> dict | None:
        preferred = detail.get("AigcVideoTask")
        if isinstance(preferred, dict):
            return preferred
        for value in detail.values():
            if isinstance(value, dict) and ("TaskId" in value or "Status" in value):
                return value
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and ("TaskId" in item or "Status" in item):
                        return item
        return detail if "Status" in detail else None

    @staticmethod
    def status(task: dict | None) -> str:
        return str((task or {}).get("Status") or "UNKNOWN").strip().upper()

    def wait_for_task(self, task_id: str) -> dict:
        started = time.monotonic()
        last_status = None
        while True:
            detail = self.describe_task(task_id)
            task = self.extract_task(detail)
            status = self.status(task)
            if status != last_status:
                print(f"[Wan 3.0 API] task {task_id}: {status}")
                last_status = status
            if status in TERMINAL_STATUSES:
                if task is None:
                    raise TencentVodTaskError(
                        f"Tencent VOD task {task_id} returned no task details.", task_id, True
                    )
                error = task_error(task)
                if status in FAILED_STATUSES or error:
                    raise TencentVodTaskError(
                        f"Tencent VOD task {task_id} failed with status={status}: {error or '-'}",
                        task_id,
                        True,
                    )
                return task
            if time.monotonic() - started > self.config.max_wait_seconds:
                raise TencentVodTaskError(
                    f"Tencent VOD task {task_id} did not finish within "
                    f"{self.config.max_wait_seconds} seconds.",
                    task_id,
                    False,
                )
            time.sleep(self.config.poll_interval)


def build_payload(config: TencentConfig, request: WanVideoRequest) -> dict:
    output_resolution = (
        request.resolution
        if request.super_resolution == "Disabled"
        else request.super_resolution
    )
    payload = {
        "SubAppId": config.sub_app_id,
        "ModelName": "Wan",
        "ModelVersion": request.model_version,
        "Prompt": request.prompt,
        "OutputConfig": {
            "StorageMode": config.storage_mode,
            "Resolution": output_resolution,
            "AspectRatio": request.aspect_ratio,
            "Duration": request.duration,
            "InputComplianceCheck": config.input_compliance_check,
            "OutputComplianceCheck": config.output_compliance_check,
        },
        "SessionId": request.session_id,
    }
    if request.file_infos:
        payload["FileInfos"] = request.file_infos
    if request.negative_prompt.strip():
        payload["NegativePrompt"] = request.negative_prompt.strip()
    if request.enhance_prompt:
        payload["EnhancePrompt"] = request.enhance_prompt
    if request.seed is not None and request.seed >= 0:
        payload["Seed"] = request.seed
    return payload


def task_error(task: dict) -> str | None:
    err_code = task.get("ErrCode")
    err_code_ext = str(task.get("ErrCodeExt") or "").strip()
    if err_code not in (None, 0, "0") or err_code_ext:
        return (
            f"ErrCode={err_code}, ErrCodeExt={err_code_ext or '-'}, "
            f"Message={task.get('Message') or '-'}"
        )
    return None


def video_result(task: dict, submission: TaskSubmission) -> VideoResult:
    output = task.get("Output") or {}
    file_infos = output.get("FileInfos") or []
    first = next(
        (item for item in file_infos if isinstance(item, dict) and (item.get("FileUrl") or item.get("Url"))),
        None,
    )
    if first is None:
        raise TencentVodTaskError(
            f"Tencent VOD task {submission.task_id} completed but returned no file URL.",
            submission.task_id,
            True,
        )
    return VideoResult(
        video_url=str(first.get("FileUrl") or first.get("Url")),
        video_id=str(first.get("FileId") or first.get("VideoId") or submission.task_id),
        task_id=submission.task_id,
        request_id=submission.request_id,
        status=TencentVodClient.status(task),
        raw=task,
    )


def sanitize_task(value):
    if isinstance(value, dict):
        return {
            key: sanitize_task(item)
            for key, item in value.items()
            if key.lower() not in {"authorization", "secretkey", "secret_key"}
        }
    if isinstance(value, list):
        return [sanitize_task(item) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(value)
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return value
