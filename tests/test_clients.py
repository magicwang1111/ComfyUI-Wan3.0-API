import json
from types import SimpleNamespace
from unittest import mock

import pytest

from wan3_api.config import OssConfig, TencentConfig
from wan3_api.models import MediaBlob, TaskSubmission, WanVideoRequest
from wan3_api.oss_client import OssClient, OssError
from wan3_api.tencent_vod import (
    TencentVodClient,
    TencentVodError,
    TencentVodTaskError,
    build_payload,
    sanitize_task,
    video_result,
)


def oss_config():
    return OssConfig(
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        access_key_id="test-id",
        access_key_secret="test-secret",
        bucket="bucket",
        prefix="tmp",
        signed_url_expires=3600,
        cleanup_after_task=True,
    )


def tencent_config():
    return TencentConfig(
        secret_id="test-id",
        secret_key="test-secret",
        region="ap-guangzhou",
        sub_app_id=123,
        poll_interval=0.001,
        request_timeout=120,
        max_wait_seconds=30,
        storage_mode="Temporary",
        input_compliance_check="Enabled",
        output_compliance_check="Enabled",
    )


def response(status=200, payload=None, request_id="request-1"):
    result = mock.Mock()
    result.status_code = status
    result.headers = {"x-oss-request-id": request_id}
    result.json.return_value = payload
    return result


def test_oss_upload_signed_url_and_delete():
    client = OssClient(oss_config())
    client.session.put = mock.Mock(return_value=response())
    client.session.delete = mock.Mock(return_value=response(status=204))
    uploaded = client.upload("tmp/wan3/day/session/file.jpg", MediaBlob(b"x", "image/jpeg", "jpg"))
    assert uploaded.object_key.startswith("tmp/wan3/")
    assert "OSSAccessKeyId=test-id" in uploaded.url
    assert "Signature=" in uploaded.url
    assert client.session.put.call_args.kwargs["headers"]["Authorization"].startswith("OSS test-id:")
    client.delete(uploaded.object_key)
    assert client.session.delete.call_count == 1


def test_oss_error_does_not_include_secret():
    client = OssClient(oss_config())
    client.session.put = mock.Mock(return_value=response(status=403))
    with pytest.raises(OssError) as raised:
        client.upload("tmp/file.jpg", MediaBlob(b"x", "image/jpeg", "jpg"))
    assert "test-secret" not in str(raised.value)
    assert "RequestId=request-1" in str(raised.value)


def test_tencent_payload_and_signature_headers():
    config = tencent_config()
    request = WanVideoRequest(
        model_version="3.0-prime",
        prompt="demo",
        resolution="720P",
        aspect_ratio="9:16",
        duration=5,
        session_id="wan3-session",
        negative_prompt="blur",
        enhance_prompt="Enabled",
        seed=7,
        super_resolution="4K",
        file_infos=[{"Type": "Url", "Category": "Image", "Usage": "FirstFrame", "Url": "https://x"}],
    )
    payload = build_payload(config, request)
    assert payload["ModelName"] == "Wan"
    assert payload["ModelVersion"] == "3.0-prime"
    assert payload["OutputConfig"]["StorageMode"] == "Temporary"
    assert payload["OutputConfig"]["Resolution"] == "4K"
    assert payload["FileInfos"][0]["Type"] == "Url"
    headers = TencentVodClient(config).headers("CreateAigcVideoTask", payload, timestamp=1700000000)
    assert headers["Host"] == "vod.tencentcloudapi.com"
    assert headers["Authorization"].startswith("TC3-HMAC-SHA256 Credential=test-id/")
    assert "test-secret" not in json.dumps(headers)


def test_tencent_payload_uses_native_resolution_when_super_resolution_is_disabled():
    request = WanVideoRequest(
        model_version="3.0",
        prompt="demo",
        resolution="1080P",
        aspect_ratio="16:9",
        duration=5,
        session_id="wan3-session",
    )
    payload = build_payload(tencent_config(), request)
    assert payload["OutputConfig"]["Resolution"] == "1080P"


def test_tencent_api_error_is_actionable_and_secret_free():
    client = TencentVodClient(tencent_config())
    client.session.post = mock.Mock(return_value=response(payload={
        "Response": {
            "Error": {"Code": "InvalidParameter", "Message": "bad input"},
            "RequestId": "tc-request",
        }
    }))
    with pytest.raises(TencentVodError) as raised:
        client.request("CreateAigcVideoTask", {})
    message = str(raised.value)
    assert "InvalidParameter" in message
    assert "tc-request" in message
    assert "test-secret" not in message


def test_wait_for_task_success_and_failure():
    client = TencentVodClient(tencent_config())
    client.describe_task = mock.Mock(side_effect=[
        {"AigcVideoTask": {"TaskId": "task", "Status": "PROCESSING"}},
        {"AigcVideoTask": {"TaskId": "task", "Status": "FINISH", "ErrCode": 0}},
    ])
    assert client.wait_for_task("task")["Status"] == "FINISH"
    client.describe_task = mock.Mock(return_value={
        "AigcVideoTask": {
            "TaskId": "failed",
            "Status": "FAIL",
            "ErrCode": 1201,
            "ErrCodeExt": "InvalidParameter",
            "Message": "bad input",
        }
    })
    with pytest.raises(TencentVodTaskError, match="InvalidParameter") as raised:
        client.wait_for_task("failed")
    assert raised.value.terminal is True


def test_video_result_requires_url_and_sanitizes_queries():
    submission = TaskSubmission("task", "request")
    task = {
        "Status": "FINISH",
        "Output": {"FileInfos": [{"FileId": "file", "FileUrl": "https://host/video.mp4?token=secret"}]},
    }
    result = video_result(task, submission)
    assert result.video_id == "file"
    assert sanitize_task(task)["Output"]["FileInfos"][0]["FileUrl"] == "https://host/video.mp4"
    with pytest.raises(TencentVodTaskError, match="no file URL"):
        video_result({"Status": "FINISH", "Output": {}}, submission)
