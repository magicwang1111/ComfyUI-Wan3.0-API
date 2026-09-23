import io
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import requests
from PIL import Image

from wan3_api import config, nodes
from wan3_api.models import MediaBlob, TaskSubmission, WanVideoRequest
from wan3_api.opc import OpcClient, OpcError, OpcSubmissionUncertain, OpcTaskError, build_payload, video_result


def settings(**kwargs):
    return config.load_opc_config({"opc_api_key": "test-secret", **kwargs})


def request(**kwargs):
    return replace(WanVideoRequest("3.0", "demo", "720P", "16:9", 5, "session"), **kwargs)


def response(body, status=200):
    result = mock.Mock(status_code=status)
    result.json.return_value = body
    return result


def test_config_and_credential_isolation():
    with mock.patch.dict("os.environ", {}, clear=True):
        assert config.load_provider({"provider": "OPC"}) == "opc"
        assert settings().base_url == "https://model-router.edu-aliyun.com"
        with pytest.raises(ValueError, match="opc_api_key"):
            config.load_opc_config({"vapeur_api_key": "other-key"})
    with mock.patch.dict("os.environ", {"OPC_API_KEY": "env-key", "WAN3_PROVIDER": "opc"}, clear=True):
        assert config.load_provider({}) == "opc"
        assert config.load_opc_config({}).api_key == "env-key"
        assert settings().api_key == "test-secret"


@pytest.mark.parametrize("values", [
    {"opc_base_url": "http://host"}, {"opc_base_url": "https://host/v1"},
    {"opc_base_url": "https://user:secret@host"}, {"opc_base_url": "https://host?key=secret"},
    {"opc_poll_interval": 0}, {"opc_request_timeout": 0}, {"opc_max_wait_seconds": 0},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        settings(**values)


def test_text_payload_uses_documented_fields():
    assert build_payload(request()) == {
        "model": "qwen/wan3.0-video/v1", "prompt": "demo", "duration": 5,
        "size": "1280x720", "watermark": False,
    }
    payload = build_payload(request(model_version="3.0-prime", aspect_ratio="9:16", resolution="1080P"))
    assert payload["model"] == "qwen/wan3.0-video-prime/v1"
    assert payload["size"] == "1080x1920"


def test_local_image_data_url_and_adaptive_size():
    buffer = io.BytesIO()
    Image.new("RGB", (600, 800)).save(buffer, format="PNG")
    payload = build_payload(request(aspect_ratio="adaptive"), [(MediaBlob(buffer.getvalue(), "image/png", "png"), "Image", "FirstFrame")])
    assert payload["image"].startswith("data:image/png;base64,iVBOR")
    assert payload["size"] == "720x960"
    assert "media" not in payload


def test_https_image_input():
    image = nodes._file_info("https://host/input.jpg", "Image", "Reference")
    assert build_payload(request(file_infos=[image]))["image"] == image["Url"]


@pytest.mark.parametrize("values", [
    {"prompt": ""}, {"duration": -1}, {"seed": 0}, {"audio_generation": "Disabled"},
    {"negative_prompt": "bad"}, {"enhance_prompt": "Enabled"}, {"super_resolution": "4K"},
])
def test_unsupported_controls_fail_before_submission(values):
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock()
        with pytest.raises(ValueError):
            client.create_video(request(**values), prompt_required=False)
        client.session.request.assert_not_called()


@pytest.mark.parametrize("media", [
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "LastFrame")],
    [(MediaBlob(b"x", "video/mp4", "mp4"), "Video", "Reference")],
    [(MediaBlob(b"x", "audio/wav", "wav"), "Audio", "Reference")],
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "Reference")] * 2,
])
def test_unsupported_media_never_uploads_or_submits(media):
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch.object(nodes, "OssClient") as oss,
        mock.patch("wan3_api.opc.requests.Session.request") as http,
        pytest.raises(ValueError),
    ):
        nodes._generate(request(), media, prompt_required=False)
    oss.assert_not_called()
    http.assert_not_called()


def test_submit_and_query_http_contract():
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({"task_id": "qwen_task", "mr_req_id": "req"}))
        assert client.create_video(request(), prompt_required=True) == TaskSubmission("qwen_task", "req")
        call = client.session.request.call_args
        assert call.args == ("POST", "https://model-router.edu-aliyun.com/v1/videos/generations")
        assert call.kwargs["json"] == build_payload(request())
        assert call.kwargs["headers"] == {"Authorization": "Bearer test-secret", "Content-Type": "application/json"}
        assert call.kwargs["allow_redirects"] is False
        client.describe_task("qwen_task/part")
        assert client.session.request.call_args.args == ("GET", "https://model-router.edu-aliyun.com/v1/tasks/qwen_task%2Fpart")
        query = client.session.request.call_args.kwargs
        assert query["headers"] == {"Authorization": "Bearer test-secret", "Content-Type": "application/json"}
        assert query["json"] is None


def test_router_task_id_and_status_take_precedence_over_upstream_output():
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({
            "task_id": "qwen_task", "output": {"task_id": "raw-task"},
        }))
        assert client.create_video(request(), prompt_required=True).task_id == "qwen_task"
    detail = {"status": "completed", "output": {"video_url": "https://host/v.mp4"}}
    assert OpcClient.status(detail) == "SUCCEEDED"
    assert video_result(detail, TaskSubmission("qwen_task", "req")).video_url == "https://host/v.mp4"


def test_query_server_error_retains_router_diagnostics():
    with OpcClient(settings()) as client, mock.patch("wan3_api.opc.time.sleep"):
        client.session.request = mock.Mock(return_value=response({
            "error": {"message": "unsupported model_type"},
            "error_code": "B.Upstream.BuildRequestException", "mr_req_id": "req",
        }, 500))
        with pytest.raises(OpcTaskError, match="B.Upstream.BuildRequestException") as exc:
            client.wait_for_task("qwen_task")
        assert "request_id=req" in str(exc.value)


@pytest.mark.parametrize("failure", [requests.ReadTimeout(), requests.ConnectionError(), response({}, 500), response({}), response([])])
def test_uncertain_submission_is_never_retried(failure):
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(side_effect=[failure])
        with pytest.raises(OpcSubmissionUncertain):
            client.create_video(request(), prompt_required=True)
        client.session.request.assert_called_once()


def test_router_error_keeps_code_and_request_id_without_secret():
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({
            "error": {"message": "bad test-secret"}, "error_code": "B.Auth.InvalidKey", "mr_req_id": "req",
        }, 401))
        with pytest.raises(OpcError) as exc:
            client.create_video(request(), prompt_required=True)
        assert "B.Auth.InvalidKey" in str(exc.value)
        assert "request_id=req" in str(exc.value)
        assert "test-secret" not in str(exc.value)
        client.session.request.assert_called_once()


def test_query_retries_then_completes_without_resubmission():
    with OpcClient(settings()) as client, mock.patch("wan3_api.opc.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=[
            requests.ReadTimeout(), response({"status": "queued"}), response({}, 429),
            response({"status": "in_progress"}), response({"status": "completed", "video_url": "https://host/v.mp4"}),
        ])
        detail = client.wait_for_task("qwen_task")
        assert video_result(detail, TaskSubmission("qwen_task", "req")).video_url == "https://host/v.mp4"
        assert all(call.args[0] == "GET" for call in client.session.request.call_args_list)
        assert sleep.call_count == 4


def test_query_retry_limit_and_deadline():
    with OpcClient(settings()) as client, mock.patch("wan3_api.opc.time.sleep"):
        client.session.request = mock.Mock(side_effect=requests.ConnectionError())
        with pytest.raises(OpcTaskError) as exc:
            client.wait_for_task("qwen_task")
        assert not exc.value.terminal
        assert client.session.request.call_count == 6
        client.describe_task = mock.Mock(return_value={"status": "running"})
        with mock.patch("wan3_api.opc.time.monotonic", side_effect=[0, 3599, 3600]):
            with pytest.raises(OpcTaskError, match="timed out"):
                client.wait_for_task("qwen_task")


@pytest.mark.parametrize("status,terminal", [("FAILED", True), ("CANCELED", True), ("UNKNOWN", False)])
def test_failed_or_unknown_task(status, terminal):
    with OpcClient(settings()) as client:
        with pytest.raises(OpcTaskError) as exc:
            client.check_task({"status": status, "message": "test-secret"}, "qwen_task")
        assert exc.value.terminal is terminal
        assert "test-secret" not in str(exc.value)


@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("envelope", ["top", "data", "output"])
def test_query_node_and_result_url(envelope, wait):
    detail = {"task_status": "SUCCEEDED", "video_url": "https://host/v.mp4?token=secret"}
    if envelope != "top":
        detail = {envelope: detail}
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch.object(OpcClient, "describe_task", return_value=detail),
        mock.patch.object(nodes, "TencentVodClient") as tencent,
    ):
        result = nodes.WanQueryTask().query("qwen_task", wait)
    assert result[:4] == ("SUCCEEDED", "https://host/v.mp4?token=secret", "qwen_task", "qwen_task")
    assert "token=secret" not in result[4]
    tencent.assert_not_called()


@pytest.mark.parametrize("mode", ["text", "frame", "reference"])
def test_generation_nodes_send_opc_payload_without_oss(mode):
    image = np.zeros((1, 256, 256, 3), dtype=np.float32)
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch("wan3_api.opc.requests.Session.request", side_effect=[
            response({"task_id": "qwen_task"}), response({"status": "completed", "video_url": "https://host/v.mp4"}),
        ]) as http,
        mock.patch.object(nodes, "OssClient") as oss,
    ):
        if mode == "text":
            result = nodes.WanTextToVideo().generate("3.0", "demo", "720P", "16:9", 5)
        elif mode == "frame":
            result = nodes.WanFrameToVideo().generate("3.0", "demo", "720P", 5, first_frame=image)
        else:
            result = nodes.WanReferenceToVideo().generate("3.0", "demo", "720P", "16:9", 5, reference_images=image)
    assert result == ("https://host/v.mp4", "qwen_task", "qwen_task")
    assert ("image" in http.call_args_list[0].kwargs["json"]) == (mode != "text")
    oss.assert_not_called()


def test_success_without_url_keeps_task_id():
    with pytest.raises(OpcTaskError, match="qwen_task"):
        video_result({"status": "completed"}, TaskSubmission("qwen_task", "req"))
