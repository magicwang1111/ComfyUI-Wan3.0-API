from dataclasses import replace
from unittest import mock

import pytest
import requests

from wan3_api import config, nodes
from wan3_api.kuaizi import (
    KuaiziClient, KuaiziError, KuaiziSubmissionUncertain, KuaiziTaskError,
    build_payload, video_result,
)
from wan3_api.models import MediaBlob, TaskSubmission, UploadedObject, WanVideoRequest


def settings(**kwargs):
    return config.load_kuaizi_config({"kuaizi_api_key": "test-secret", **kwargs})


def request(**kwargs):
    return replace(WanVideoRequest("3.0", "demo", "480P", "16:9", 5, "session"), **kwargs)


def response(body, status=200):
    result = mock.Mock(status_code=status)
    result.json.return_value = body
    return result


def test_config_defaults_environment_and_isolation():
    with mock.patch.dict("os.environ", {}, clear=True):
        assert config.load_provider({"provider": "Kuaizi"}) == "kuaizi"
        assert settings().base_url == "https://aiopenapi.kuaizi.cn/ai-open-platform-api"
        assert settings().poll_interval == 15
        with pytest.raises(ValueError, match="kuaizi_api_key"):
            config.load_kuaizi_config({"vapeur_api_key": "other-provider"})
    with mock.patch.dict("os.environ", {"KUAIZI_API_KEY": "env-key", "WAN3_PROVIDER": "kuaizi"}, clear=True):
        assert config.load_provider({}) == "kuaizi"
        assert config.load_kuaizi_config({}).api_key == "env-key"
        assert settings().api_key == "test-secret"


@pytest.mark.parametrize("base_url", [
    "https://aiopenapi.kuaizi.cn", "https://aiopenapi.kuaizi.cn/",
    "https://aiopenapi.kuaizi.cn/ai-open-platform-api",
    "https://aiopenapi.kuaizi.cn/ai-open-platform-api/",
])
def test_base_url_normalization(base_url):
    assert settings(kuaizi_base_url=base_url).base_url == "https://aiopenapi.kuaizi.cn/ai-open-platform-api"


@pytest.mark.parametrize("values", [
    {"kuaizi_base_url": "http://aiopenapi.kuaizi.cn"},
    {"kuaizi_base_url": "https://aiopenapi.kuaizi.cn/api/v1"},
    {"kuaizi_base_url": "https://user:secret@aiopenapi.kuaizi.cn"},
    {"kuaizi_base_url": "https://aiopenapi.kuaizi.cn?key=secret"},
    {"kuaizi_poll_interval": 5}, {"kuaizi_request_timeout": 0}, {"kuaizi_max_wait_seconds": 0},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        settings(**values)


@pytest.mark.parametrize("version,model", [("3.0", "wan3.0-video"), ("3.0-prime", "wan3.0-video-prime")])
def test_payload_exact_contract(version, model):
    assert build_payload(request(model_version=version, duration=-1, seed=0, audio_generation="Disabled")) == {
        "model": model, "input": {"prompt": "demo"},
        "parameters": {"resolution": "480P", "ratio": "16:9", "duration": -1,
                       "seed": 0, "audio": False, "watermark": False},
    }


@pytest.mark.parametrize("category,usage,kind", [
    ("Image", "FirstFrame", "first_frame"), ("Image", "LastFrame", "last_frame"),
    ("Image", "Reference", "reference_image"), ("Video", "Reference", "reference_video"),
    ("Audio", "Reference", "reference_audio"),
])
def test_media_mapping_without_prompt(category, usage, kind):
    payload = build_payload(request(prompt="", file_infos=[nodes._file_info("https://signed", category, usage)]))
    assert payload["input"] == {"media": [{"type": kind, "url": "https://signed"}]}
    assert "seed" not in payload["parameters"]


@pytest.mark.parametrize("values", [
    {"negative_prompt": "blur"}, {"enhance_prompt": "Enabled"},
    {"super_resolution": "4K"}, {"seed": 2147483648},
])
def test_unsupported_parameters_rejected_before_upload(values):
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "kuaizi", "kuaizi_api_key": "test"}),
        mock.patch.object(nodes, "OssClient") as oss,
        pytest.raises(ValueError),
    ):
        nodes._generate(request(**values), [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")], prompt_required=True)
    oss.assert_not_called()


def test_submit_and_query_wire_contract():
    with KuaiziClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({
            "output": {"task_id": "kz-cgt-task", "task_status": "PENDING"}, "request_id": "req",
        }))
        assert client.create_video(request(), prompt_required=True) == TaskSubmission("kz-cgt-task", "req")
        call = client.session.request.call_args
        assert call.args == ("POST", "https://aiopenapi.kuaizi.cn/ai-open-platform-api/api/v1/services/aigc/video-generation/video-synthesis")
        assert call.kwargs["headers"] == {
            "Authorization": "Bearer test-secret", "Content-Type": "application/json", "X-DashScope-Async": "enable",
        }
        assert call.kwargs["json"] == build_payload(request())
        assert call.kwargs["allow_redirects"] is False
        assert call.kwargs["timeout"] == 120
        client.describe_task("kz-cgt/task")
        assert client.session.request.call_args.args == ("GET", "https://aiopenapi.kuaizi.cn/ai-open-platform-api/api/v1/tasks/kz-cgt%2Ftask")


@pytest.mark.parametrize("status,code", [(400, "InvalidParameter"), (401, "InvalidApiKey"), (429, "Throttling"), (429, "InsufficientBalance")])
def test_sync_errors_are_not_resubmitted(status, code):
    with KuaiziClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({"code": code, "message": "bad test-secret", "request_id": "req"}, status))
        with pytest.raises(KuaiziError, match=code) as exc:
            client.create_video(request(), prompt_required=True)
        assert not isinstance(exc.value, KuaiziSubmissionUncertain)
        assert "test-secret" not in str(exc.value)
        assert "request_id=req" in str(exc.value)
        client.session.request.assert_called_once()


@pytest.mark.parametrize("failure", [
    requests.ReadTimeout(), requests.ConnectionError(), response({"code": "InternalError"}, 500),
    response({"output": {}}), response({"output": []}), response([]),
])
def test_uncertain_submission_is_not_retried(failure):
    with KuaiziClient(settings()) as client:
        client.session.request = mock.Mock(side_effect=[failure])
        with pytest.raises(KuaiziSubmissionUncertain):
            client.create_video(request(), prompt_required=True)
        client.session.request.assert_called_once()


@pytest.mark.parametrize("status,terminal", [("FAILED", True), ("CANCELED", True), ("UNKNOWN", False)])
def test_http_200_task_failure(status, terminal):
    with KuaiziClient(settings()) as client:
        client.session.request = mock.Mock(return_value=response({
            "output": {"task_status": status, "code": "InvalidParameter", "message": "bad test-secret"}, "request_id": "req",
        }))
        with pytest.raises(KuaiziTaskError, match="InvalidParameter") as exc:
            client.wait_for_task("kz-task")
        assert exc.value.terminal is terminal
        assert exc.value.task_id == "kz-task"
        assert "test-secret" not in str(exc.value)
        assert "request_id=req" in str(exc.value)


def test_pending_running_and_success():
    with KuaiziClient(settings()) as client, mock.patch("wan3_api.kuaizi.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=[response({"output": {"task_status": status}}) for status in ["PENDING", "RUNNING", "SUCCEEDED"]])
        assert client.wait_for_task("kz-task")["output"]["task_status"] == "SUCCEEDED"
        assert [call.args[0] for call in sleep.call_args_list] == [15, 15]


@pytest.mark.parametrize("failure", [requests.ReadTimeout(), requests.ConnectionError(), 408, 429, 500, 502, 503, 504])
def test_transient_queries_retry_without_resubmitting(failure):
    failed = response({}, failure) if isinstance(failure, int) else failure
    with KuaiziClient(settings()) as client, mock.patch("wan3_api.kuaizi.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=[failed, response({"output": {"task_status": "SUCCEEDED"}})])
        client.wait_for_task("existing-task")
        assert [call.args[0] for call in client.session.request.call_args_list] == ["GET", "GET"]
        sleep.assert_called_once_with(15)


def test_query_retries_bounded_and_nonterminal():
    with KuaiziClient(settings()) as client, mock.patch("wan3_api.kuaizi.time.sleep") as sleep:
        client.session.request = mock.Mock(side_effect=requests.ReadTimeout())
        with pytest.raises(KuaiziTaskError) as exc:
            client.wait_for_task("existing-task")
        assert not exc.value.terminal
        assert client.session.request.call_count == 6
        assert [call.args[0] for call in sleep.call_args_list] == [15, 30, 60, 60, 60]


def test_wait_deadline_keeps_task_recoverable():
    with KuaiziClient(settings()) as client, mock.patch("wan3_api.kuaizi.time.sleep") as sleep:
        client.describe_task = mock.Mock(return_value={"output": {"task_status": "RUNNING"}})
        with mock.patch("wan3_api.kuaizi.time.monotonic", side_effect=[0, 3599, 3600]):
            with pytest.raises(KuaiziTaskError, match="timed out") as exc:
                client.wait_for_task("task")
        assert not exc.value.terminal
        sleep.assert_called_once_with(1)


@pytest.mark.parametrize("wait,status", [(False, "RUNNING"), (False, "SUCCEEDED"), (True, "SUCCEEDED")])
def test_query_node_routes_to_kuaizi(wait, status):
    detail = {"output": {"task_status": status}}
    if status == "SUCCEEDED":
        detail["output"]["video_url"] = "https://host/video.mp4?token=secret"
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "kuaizi", "kuaizi_api_key": "test"}),
        mock.patch.object(KuaiziClient, "describe_task", return_value=detail),
        mock.patch.object(nodes, "TencentVodClient") as tencent,
        mock.patch.object(nodes, "VapeurClient") as vapeur,
    ):
        result = nodes.WanQueryTask().query("kz-task", wait)
    assert result[0] == status
    assert "token=secret" not in result[4]
    tencent.assert_not_called()
    vapeur.assert_not_called()


def test_text_generation_routes_without_oss():
    detail = {"output": {"task_status": "SUCCEEDED", "video_url": "https://video"}}
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "kuaizi", "kuaizi_api_key": "test"}),
        mock.patch.object(KuaiziClient, "create_video", return_value=TaskSubmission("kz-task", "req")),
        mock.patch.object(KuaiziClient, "wait_for_task", return_value=detail),
        mock.patch.object(nodes, "OssClient") as oss,
    ):
        result = nodes._generate(request(), [], prompt_required=True)
    assert result.video_url == "https://video"
    assert result.video_id == result.task_id == "kz-task"
    oss.assert_not_called()


@pytest.mark.parametrize("error,deletes", [
    (None, 1), (KuaiziTaskError("failed", "task", True), 1),
    (KuaiziTaskError("timeout", "task", False), 0),
    (KuaiziSubmissionUncertain("unknown"), 0), (KuaiziError("Throttling"), 1),
])
def test_oss_cleanup_and_retention(error, deletes):
    oss = mock.MagicMock()
    oss.__enter__.return_value = oss
    oss.upload.return_value = UploadedObject("tmp/x", "https://signed")
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.create_video.return_value = TaskSubmission("task", "req")
    client.wait_for_task.return_value = {"output": {"task_status": "SUCCEEDED", "video_url": "https://video"}}
    if isinstance(error, KuaiziTaskError):
        client.wait_for_task.side_effect = error
    elif error:
        client.create_video.side_effect = error
    with (
        mock.patch.object(nodes, "load_json_config", return_value={"provider": "kuaizi", "kuaizi_api_key": "test"}),
        mock.patch.object(nodes, "load_oss_config", return_value=config.OssConfig("host", "id", "secret", "bucket", "tmp", 86400, True)),
        mock.patch.object(nodes, "OssClient", return_value=oss),
        mock.patch.object(nodes, "KuaiziClient", return_value=client),
    ):
        if error:
            with pytest.raises(type(error)):
                nodes._generate(request(), [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")], prompt_required=True)
        else:
            result = nodes._generate(request(), [(MediaBlob(b"x", "image/jpeg", "jpg"), "Image", "FirstFrame")], prompt_required=True)
            assert result.video_url == "https://video"
        assert client.create_video.call_args.args[0].file_infos[0]["Url"] == "https://signed"
    assert oss.delete.call_count == deletes


def test_missing_success_url_is_terminal():
    with pytest.raises(KuaiziTaskError) as exc:
        video_result({"output": {"task_status": "SUCCEEDED"}}, TaskSubmission("task", "req"))
    assert exc.value.terminal
