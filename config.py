from __future__ import annotations

import json
import os
from urllib.parse import urlsplit
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.local.json"


@dataclass(frozen=True)
class VapeurConfig:
    api_key: str
    base_url: str
    region: str
    poll_interval: float
    request_timeout: int
    max_wait_seconds: int


def load_provider(data: dict | None = None) -> str:
    data = load_json_config() if data is None else data
    provider = str(_value(data, "provider", "WAN3_PROVIDER", "tencent")).strip().lower()
    if provider not in {"tencent", "vapeur"}:
        raise ValueError("provider must be tencent or vapeur.")
    return provider


def load_vapeur_config(data: dict | None = None) -> VapeurConfig:
    data = load_json_config() if data is None else data
    api_key = str(_value(data, "vapeur_api_key", "VAPEUR_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("Vapeur requires vapeur_api_key in config.local.json or VAPEUR_API_KEY.")
    base_url = str(_value(data, "vapeur_base_url", "VAPEUR_BASE_URL", "https://api.vapeur.ai")).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("vapeur_base_url must be an HTTPS origin without a path (do not append /v1).")
    region = str(_value(data, "vapeur_region", "VAPEUR_REGION", "cn")).strip().lower()
    if region not in {"cn", "glb"}:
        raise ValueError("vapeur_region must be cn or glb.")
    return VapeurConfig(
        api_key=api_key, base_url=base_url, region=region,
        poll_interval=_positive_float(_value(data, "vapeur_poll_interval", "VAPEUR_POLL_INTERVAL", 5), "vapeur_poll_interval"),
        request_timeout=_positive_int(_value(data, "vapeur_request_timeout", "VAPEUR_REQUEST_TIMEOUT", 120), "vapeur_request_timeout", 5),
        max_wait_seconds=_positive_int(_value(data, "vapeur_max_wait_seconds", "VAPEUR_MAX_WAIT_SECONDS", 3600), "vapeur_max_wait_seconds", 30),
    )


@dataclass(frozen=True)
class OssConfig:
    endpoint: str
    access_key_id: str
    access_key_secret: str
    bucket: str
    prefix: str
    signed_url_expires: int
    cleanup_after_task: bool


@dataclass(frozen=True)
class TencentConfig:
    secret_id: str
    secret_key: str
    region: str
    sub_app_id: int
    poll_interval: float
    request_timeout: int
    max_wait_seconds: int
    storage_mode: str
    input_compliance_check: str
    output_compliance_check: str


def load_json_config(path: Path | None = None) -> dict:
    path = path or CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a top-level JSON object.")
    return data


def _present(data: dict, key: str) -> bool:
    value = data.get(key)
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _value(data: dict, key: str, env: str, default=None):
    if _present(data, key):
        return data[key]
    env_value = os.getenv(env, "").strip()
    return env_value if env_value else default


def _positive_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return parsed


def _positive_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _boolean(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _oss_prefix(value) -> str:
    prefix = str(value or "").strip().strip("/")
    if not prefix or prefix.startswith((".", "\\")) or ".." in prefix.split("/"):
        raise ValueError("oss_prefix must be a non-empty relative object prefix.")
    return prefix


def load_oss_config(data: dict | None = None) -> OssConfig:
    data = load_json_config() if data is None else data
    endpoint = str(_value(data, "oss_endpoint", "OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")).strip()
    endpoint = endpoint.removeprefix("https://").removeprefix("http://").strip("/")
    bucket = str(_value(data, "oss_bucket", "OSS_BUCKET", "")).strip()
    access_key_id = str(_value(data, "oss_access_key_id", "OSS_ACCESS_KEY_ID", "")).strip()
    access_key_secret = str(_value(data, "oss_access_key_secret", "OSS_ACCESS_KEY_SECRET", "")).strip()
    if not endpoint or not bucket or not access_key_id or not access_key_secret:
        raise ValueError(
            "OSS input upload requires oss_endpoint, oss_bucket, oss_access_key_id, and "
            "oss_access_key_secret in config.local.json or environment variables."
        )
    if any(char in endpoint for char in "/?# "):
        raise ValueError("oss_endpoint must be a hostname without a scheme or path.")
    if any(char in bucket for char in "/\\"):
        raise ValueError("oss_bucket must be a bucket name, not a path.")
    return OssConfig(
        endpoint=endpoint,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        bucket=bucket,
        prefix=_oss_prefix(_value(data, "oss_prefix", "OSS_PREFIX", "ComfyUI-tmp")),
        signed_url_expires=_positive_int(
            _value(data, "oss_signed_url_expires", "OSS_SIGNED_URL_EXPIRES", 86400),
            "oss_signed_url_expires",
            60,
        ),
        cleanup_after_task=_boolean(
            _value(data, "oss_cleanup_after_task", "OSS_CLEANUP_AFTER_TASK", True),
            "oss_cleanup_after_task",
        ),
    )


def load_tencent_config(data: dict | None = None) -> TencentConfig:
    data = load_json_config() if data is None else data
    secret_id = str(_value(data, "tencent_secret_id", "TENCENTCLOUD_SECRET_ID", "")).strip()
    secret_key = str(_value(data, "tencent_secret_key", "TENCENTCLOUD_SECRET_KEY", "")).strip()
    sub_app_id = _positive_int(
        _value(data, "tencent_sub_app_id", "TENCENT_SUB_APP_ID", 0),
        "tencent_sub_app_id",
        1,
    )
    if not secret_id or not secret_key:
        raise ValueError(
            "Tencent VOD requires tencent_secret_id and tencent_secret_key in "
            "config.local.json or environment variables."
        )
    storage_mode = str(data.get("tencent_storage_mode") or "Temporary").strip()
    if storage_mode != "Temporary":
        raise ValueError("tencent_storage_mode must be Temporary.")
    checks = {}
    for key in ("tencent_input_compliance_check", "tencent_output_compliance_check"):
        value = str(data.get(key) or "Enabled").strip()
        if value not in {"Enabled", "Disabled"}:
            raise ValueError(f"{key} must be Enabled or Disabled.")
        checks[key] = value
    return TencentConfig(
        secret_id=secret_id,
        secret_key=secret_key,
        region=str(_value(data, "tencent_region", "TENCENT_REGION", "ap-guangzhou")).strip(),
        sub_app_id=sub_app_id,
        poll_interval=_positive_float(
            _value(data, "tencent_poll_interval", "TENCENT_POLL_INTERVAL", 5.0),
            "tencent_poll_interval",
        ),
        request_timeout=_positive_int(
            _value(data, "tencent_request_timeout", "TENCENT_REQUEST_TIMEOUT", 120),
            "tencent_request_timeout",
            5,
        ),
        max_wait_seconds=_positive_int(
            _value(data, "tencent_max_wait_seconds", "TENCENT_MAX_WAIT_SECONDS", 3600),
            "tencent_max_wait_seconds",
            30,
        ),
        storage_mode=storage_mode,
        input_compliance_check=checks["tencent_input_compliance_check"],
        output_compliance_check=checks["tencent_output_compliance_check"],
    )
