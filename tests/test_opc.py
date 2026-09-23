from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import requests

from wan3_api import config, nodes
from wan3_api.models import MediaBlob, TaskSubmission, UploadedObject, WanVideoRequest
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
        "model": "qwen/wan3.0-video/v1", "input": {"prompt": "demo"},
        "parameters": {"resolution": "720P", "ratio": "16:9", "duration": 5,
                       "audio": True, "prompt_extend": False, "watermark": False},
    }
    payload = build_payload(request(model_version="3.0-prime", aspect_ratio="adaptive", resolution="1080P",
                                    duration=-1, seed=0, audio_generation="Disabled", enhance_prompt="Enabled"))
    assert payload["model"] == "qwen/wan3.0-video-prime/v1"
    assert payload["parameters"] == {"resolution": "1080P", "ratio": "adaptive", "duration": -1,
                                     "seed": 0, "audio": False, "prompt_extend": True, "watermark": False}


@pytest.mark.parametrize("category,usage,kind", [
    ("Image", "FirstFrame", "first_frame"), ("Image", "Reference", "reference_image"),
    ("Video", "Reference", "reference_video"), ("Audio", "Reference", "reference_audio"),
    ("File", "Reference", "file"), ("Link", "Reference", "link"),
])
def test_native_media_without_prompt(category, usage, kind):
    payload = build_payload(request(prompt="", enhance_prompt="Enabled", file_infos=[
        nodes._file_info("https://host/input", category, usage),
    ]))
    assert payload["input"] == {"media": [{"type": kind, "url": "https://host/input"}]}


def test_https_image_input():
    image = nodes._file_info("https://host/input.jpg", "Image", "Reference")
    assert build_payload(request(file_infos=[image]))["input"]["media"][0]["url"] == image["Url"]


def test_frame_pair_and_multimodal_order():
    frames = [nodes._file_info("https://host/" + usage, "Image", usage) for usage in ("FirstFrame", "LastFrame")]
    assert [item["type"] for item in build_payload(request(file_infos=frames))["input"]["media"]] == ["first_frame", "last_frame"]
    references = [nodes._file_info("https://host/" + str(i), category, "Reference")
                  for i, category in enumerate(["Image", "Video", "Image", "Audio"])]
    media = build_payload(request(file_infos=references))["input"]["media"]
    assert [item["type"] for item in media] == ["reference_image", "reference_video", "reference_image", "reference_audio"]
    assert [item["url"] for item in media] == [item["Url"] for item in references]


@pytest.mark.parametrize("values", [
    {"prompt": ""}, {"duration": 0}, {"seed": 2147483648},
    {"negative_prompt": "bad"}, {"super_resolution": "4K"},
])
def test_unsupported_controls_fail_before_submission(values):
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock()
        with pytest.raises(ValueError):
            client.create_video(request(**values), prompt_required=False)
        client.session.request.assert_not_called()


@pytest.mark.parametrize("media", [
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "LastFrame")],
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")] * 2,
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "Reference")] * 11,
    [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame"),
     (MediaBlob(b"x", "audio/wav", "wav"), "Audio", "Reference")],
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
        assert call.kwargs["headers"] == {"Authorization": "Bearer test-secret", "Content-Type": "application/json",
                                          "X-DashScope-Async": "enable"}
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
        assert "mr_req_id=req" in str(exc.value)


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
        assert "mr_req_id=req" in str(exc.value)
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
def test_generation_nodes_send_native_opc_payload(mode):
    image = np.zeros((1, 256, 256, 3), dtype=np.float32)
    oss = mock.MagicMock()
    oss.__enter__.return_value = oss
    oss.upload.return_value = UploadedObject("tmp/image", "https://host/image.jpg")
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch.object(nodes, "load_oss_config", return_value=config.OssConfig("host", "id", "secret", "bucket", "tmp", 86400, True)),
        mock.patch("wan3_api.opc.requests.Session.request", side_effect=[
            response({"output": {"task_id": "qwen_task", "task_status": "PENDING"}}),
            response({"output": {"task_status": "SUCCEEDED", "video_url": "https://host/v.mp4"}}),
        ]) as http,
        mock.patch.object(nodes, "OssClient", return_value=oss) as oss_type,
    ):
        if mode == "text":
            result = nodes.WanTextToVideo().generate("3.0", "demo", "720P", "16:9", 5)
        elif mode == "frame":
            result = nodes.WanFrameToVideo().generate("3.0", "demo", "720P", 5, first_frame=image)
        else:
            result = nodes.WanReferenceToVideo().generate("3.0", "demo", "720P", "16:9", 5, reference_images=image)
    assert result == ("https://host/v.mp4", "qwen_task", "qwen_task")
    payload = http.call_args_list[0].kwargs["json"]
    assert ("media" in payload["input"]) == (mode != "text")
    if mode == "text":
        oss_type.assert_not_called()
    else:
        expected_type = "first_frame" if mode == "frame" else "reference_image"
        assert payload["input"]["media"] == [{"type": expected_type, "url": "https://host/image.jpg"}]
        oss.delete.assert_called_once_with("tmp/image")


def test_success_without_url_keeps_task_id():
    with pytest.raises(OpcTaskError, match="qwen_task"):
        video_result({"status": "completed"}, TaskSubmission("qwen_task", "req"))


@pytest.mark.parametrize("status", [200, 400])
def test_upstream_error_envelope(status):
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({
            "code": "InvalidParameter", "message": "bad test-secret", "request_id": "upstream-req",
        }, status))
        with pytest.raises(OpcError, match="InvalidParameter") as exc:
            client.create_video(request(), prompt_required=True)
        assert not isinstance(exc.value, OpcSubmissionUncertain)
        assert "request_id=upstream-req" in str(exc.value)
        assert "test-secret" not in str(exc.value)
        client.session.request.assert_called_once()


def test_observed_router_submission_and_failed_task():
    with OpcClient(settings()) as client:
        client.session.request = mock.Mock(side_effect=[
            response({"mr_req_id": "router-submit", "request_id": "upstream-submit",
                      "output": {"task_id": "qwen_native-task", "task_status": "PENDING"}}),
            response({"mr_req_id": "router-query", "request_id": "upstream-query",
                      "output": {"task_id": "qwen_native-task", "task_status": "FAILED",
                                 "code": "InvalidParameter", "message": "media.url scheme must be http/https, got: ''"}}),
        ])
        submission = client.create_video(request(), prompt_required=True)
        assert submission == TaskSubmission("qwen_native-task", "upstream-submit")
        with pytest.raises(OpcTaskError, match="InvalidParameter") as exc:
            client.wait_for_task(submission.task_id)
        assert exc.value.terminal
        assert "request_id=upstream-query" in str(exc.value)
        assert "mr_req_id=router-query" in str(exc.value)


@pytest.mark.parametrize("file_url,link_url,enhance,match", [
    ("https://host/a.pdf", "https://host/page", "Enabled", "mutually exclusive"),
    ("https://host/a.pdf", "", "Disabled", "enhance_prompt"),
    ("", "https://host/page", "Disabled", "enhance_prompt"),
    ("D:/private/a.pdf", "", "Enabled", "HTTP"),
])
def test_invalid_document_inputs_fail_before_network(file_url, link_url, enhance, match):
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch.object(nodes, "OssClient") as oss,
        mock.patch("wan3_api.opc.requests.Session.request") as http,
        pytest.raises(ValueError, match=match),
    ):
        nodes.WanReferenceToVideo().generate("3.0", "", "720P", "adaptive", 5,
                                            reference_file_url=file_url, reference_link_url=link_url, enhance_prompt=enhance)
    oss.assert_not_called()
    http.assert_not_called()


@pytest.mark.parametrize("field,kind", [("reference_file_url", "file"), ("reference_link_url", "link")])
def test_document_only_node_uses_url_without_oss(field, kind):
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch("wan3_api.opc.requests.Session.request", side_effect=[
            response({"output": {"task_id": "qwen_task"}}),
            response({"output": {"task_status": "SUCCEEDED", "video_url": "https://host/v.mp4"}}),
        ]) as http,
        mock.patch.object(nodes, "OssClient") as oss,
    ):
        result = nodes.WanReferenceToVideo().generate("3.0", "", "720P", "adaptive", -1,
                                                     enhance_prompt="Enabled", **{field: "https://host/input"})
    assert result[2] == "qwen_task"
    payload = http.call_args_list[0].kwargs["json"]
    assert payload["input"] == {"media": [{"type": kind, "url": "https://host/input"}]}
    assert payload["parameters"]["prompt_extend"] is True
    oss.assert_not_called()


def test_audio_only_reference_node_for_opc():
    audio = {"waveform": np.zeros((1, 1, 16000), dtype=np.float32), "sample_rate": 8000}
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc"}),
        mock.patch.object(nodes, "_generate") as generate,
    ):
        nodes.WanReferenceToVideo().generate("3.0", "", "720P", "adaptive", 5, reference_audio_1=audio)
    request_arg, media = generate.call_args.args
    assert request_arg.prompt == ""
    assert [item[1:] for item in media] == [("Audio", "Reference")]


@pytest.mark.parametrize("error,deletes", [
    (None, 3), (OpcTaskError("failed", "qwen_task", True), 3),
    (OpcTaskError("timeout", "qwen_task", False), 0),
    (OpcSubmissionUncertain("unknown"), 0), (OpcError("rejected"), 3),
])
def test_multimodal_oss_cleanup_and_retention(error, deletes):
    oss = mock.MagicMock()
    oss.__enter__.return_value = oss
    oss.upload.side_effect = [UploadedObject(f"tmp/{i}", f"https://host/{i}") for i in range(3)]
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.create_video.return_value = TaskSubmission("qwen_task", "req")
    client.wait_for_task.return_value = {"output": {"task_status": "SUCCEEDED", "video_url": "https://host/v.mp4"}}
    if isinstance(error, OpcTaskError):
        client.wait_for_task.side_effect = error
    elif error:
        client.create_video.side_effect = error
    media = [(MediaBlob(b"x", content_type, extension), category, "Reference") for content_type, extension, category
             in [("image/jpeg", "jpg", "Image"), ("video/mp4", "mp4", "Video"), ("audio/wav", "wav", "Audio")]]
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "opc", "opc_api_key": "test"}),
        mock.patch.object(nodes, "load_oss_config", return_value=config.OssConfig("host", "id", "secret", "bucket", "tmp", 86400, True)),
        mock.patch.object(nodes, "OssClient", return_value=oss),
        mock.patch.object(nodes, "OpcClient", return_value=client),
    ):
        if error:
            with pytest.raises(type(error)):
                nodes._generate(request(), media, prompt_required=False)
        else:
            assert nodes._generate(request(), media, prompt_required=False).video_url == "https://host/v.mp4"
    submitted = client.create_video.call_args.args[0]
    assert build_payload(submitted)["input"]["media"] == [
        {"type": kind, "url": f"https://host/{i}"}
        for i, kind in enumerate(["reference_image", "reference_video", "reference_audio"])
    ]
    assert oss.delete.call_count == deletes
